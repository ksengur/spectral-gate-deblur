"""USRNet-style unrolled deblurring network (scale factor 1) for the conditioning study.

Half-quadratic splitting with T stages, as in USRNet (Zhang et al., CVPR 2020):
  data step  : x_k = argmin ||k * x - y||^2 + a_k ||x - z_{k-1}||^2   (closed form, FFT)
  prior step : z_k = D(x_k, b_k)   shared NAFNet denoiser, noise-level map b_k as 4th channel
  a_k, b_k   : predicted from sigma by a small MLP (USRNet's HyPa-Net)
The kernel enters only through the physics of the data step, never as a learned embedding,
which is the design usually credited with generalising to unseen kernels.

Interface matches NAFNet: forward(inp, token, params) with params = kernel_descriptor
[present, sigma/40, sqrt(k) flattened], so train/eval code is shared with --cond kernel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from degrade_torch import KSIZE, N_KDESC
from nafnet import NAFNet


def _fft_kernel(kernels, hp, wp):
    b, ks, _ = kernels.shape
    kp = torch.zeros(b, hp, wp, device=kernels.device)
    kp[:, :ks, :ks] = kernels.float()
    return torch.fft.rfft2(torch.roll(kp, (-(ks // 2), -(ks // 2)), dims=(1, 2)))[:, None]


class Unrolled(nn.Module):
    def __init__(self, width=16, stages=4, floor=0.29, enc_blocks=(1, 1, 2), middle_blocks=2,
                 dec_blocks=(1, 1, 1), data=True, emb=False):
        super().__init__()
        # 2x2 factorial on how the kernel enters: data=True -> FFT data step uses k (else k = delta,
        # i.e. no kernel in the physics); emb=True -> kernel descriptor also FiLM-conditions the prior.
        self.stages, self.floor, self.data, self.emb = stages, floor, data, emb
        self.prior = NAFNet(width=width, enc_blocks=enc_blocks, middle_blocks=middle_blocks,
                            dec_blocks=dec_blocks, in_ch=4, n_params=N_KDESC if emb else 0)
        self.hypa = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                  nn.Linear(64, 2 * stages), nn.Softplus())

    def forward(self, inp, token, params):
        y = inp.float()
        b, c, h, w = y.shape
        sigma = params[:, 1] * 40  # 0-255 units
        k = (params[:, 2:].float() ** 2).reshape(b, KSIZE, KSIZE)
        k = k / k.sum((1, 2), keepdim=True).clamp_min(1e-8)
        s = ((sigma ** 2 + self.floor ** 2) ** 0.5 / 255)[:, None]
        hyp = self.hypa(s * 10)
        # alpha scaled by noise variance (USRNet: alpha_k = mu_k * sigma^2), beta as noise level
        alpha = hyp[:, :self.stages] * (s ** 2) * 100 + 1e-6
        beta = hyp[:, self.stages:] * s
        p = 2 * KSIZE
        yp = F.pad(y, [p] * 4, mode="reflect")
        hp, wp = yp.shape[-2:]
        if not self.data:  # delta kernel: same computation, no kernel information
            k = torch.zeros_like(k)
            k[:, KSIZE // 2, KSIZE // 2] = 1
        H = _fft_kernel(k, hp, wp)
        HY, H2 = H.conj() * torch.fft.rfft2(yp), H.abs() ** 2
        z = y
        tok = torch.full((b,), 2, device=y.device, dtype=torch.long)  # UNKNOWN: order not used
        for i in range(self.stages):
            a = alpha[:, i, None, None, None]
            zp = F.pad(z.float(), [p] * 4, mode="reflect")
            x = torch.fft.irfft2((HY + a * torch.fft.rfft2(zp)) / (H2 + a), s=(hp, wp))[..., p:p + h, p:p + w]
            bmap = beta[:, i, None, None, None].expand(b, 1, h, w)
            z = self.prior(torch.cat([bmap, x], 1).to(inp.dtype), tok, params if self.emb else None)
        return z


UNROLLED = {"unrolled": dict(data=True, emb=False), "unrolled_emb": dict(data=False, emb=True),
            "unrolled_both": dict(data=True, emb=True), "unrolled_blind": dict(data=False, emb=False)}


def build_model(cond, width, n_params, in_ch, floor):
    """Factory shared by train/eval: 'unrolled' gets the unrolled net, everything else NAFNet."""
    if cond in UNROLLED:
        return Unrolled(floor=floor, **UNROLLED[cond])
    return NAFNet(width=width, n_params=n_params, in_ch=in_ch)
