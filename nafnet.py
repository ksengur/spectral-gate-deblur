"""Lightweight NAFNet with an order token injected by FiLM in every block.

Every model in the 2x2 study has exactly the same architecture and parameter
count. Blind models always receive the UNKNOWN token; oracle models receive the
true formation order (0 = blur->noise, 1 = noise->blur).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BLUR_THEN_NOISE, NOISE_THEN_BLUR, UNKNOWN = 0, 1, 2


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, cond_dim):
        super().__init__()
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, 2 * c, 1)
        self.conv2 = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.conv3 = nn.Conv2d(c, c, 1)
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, 2 * c, 1)
        self.conv5 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        # FiLM: zero init -> starts as an unconditioned NAFBlock.
        self.film = nn.Linear(cond_dim, 2 * c)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, inp, cond):
        scale, shift = self.film(cond)[:, :, None, None].chunk(2, dim=1)
        x = self.norm1(inp) * (1 + scale) + shift
        x = self.sg(self.conv2(self.conv1(x)))
        x = x * self.sca(x)
        y = inp + self.conv3(x) * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, width=32, enc_blocks=(1, 2, 4), middle_blocks=4,
                 dec_blocks=(1, 1, 1), cond_dim=16, n_params=0, in_ch=3):
        super().__init__()
        self.embed = nn.Embedding(3, cond_dim)
        # Optional continuous degradation descriptor (0 keeps old checkpoints loadable).
        self.n_params = n_params
        if n_params:
            self.param_mlp = nn.Sequential(nn.Linear(n_params, cond_dim), nn.SiLU(),
                                           nn.Linear(cond_dim, cond_dim))
        # in_ch > 3: extra channels (e.g. a Wiener estimate); the residual base is the last 3
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()
        c = width
        for n in enc_blocks:
            self.encoders.append(nn.ModuleList(NAFBlock(c, cond_dim) for _ in range(n)))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2
        self.middle = nn.ModuleList(NAFBlock(c, cond_dim) for _ in range(middle_blocks))
        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.ModuleList(NAFBlock(c, cond_dim) for _ in range(n)))
        self.pad_to = 2 ** len(enc_blocks)

    @staticmethod
    def _run(blocks, x, cond):
        for block in blocks:
            x = block(x, cond)
        return x

    def forward(self, inp, token, params=None):
        h, w = inp.shape[-2:]
        ph, pw = (-h) % self.pad_to, (-w) % self.pad_to
        x0 = F.pad(inp, (0, pw, 0, ph), mode="reflect") if ph or pw else inp
        cond = self.embed(token)
        if self.n_params:
            cond = cond + self.param_mlp(params)
        x = self.intro(x0)
        skips = []
        for blocks, down in zip(self.encoders, self.downs):
            x = self._run(blocks, x, cond)
            skips.append(x)
            x = down(x)
        x = self._run(self.middle, x, cond)
        for blocks, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = self._run(blocks, up(x) + skip, cond)
        x = self.ending(x) + x0[:, -3:]
        return x[..., :h, :w]
