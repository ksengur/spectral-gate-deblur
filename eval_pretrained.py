"""Pretrained unfolding / plug-and-play deblurrers under the locked protocol + the two-sided gate.

  usrnet : USRNet (Zhang et al., CVPR 2020), official KAIR weights usrnet.pth, scale factor 1.
           8 HQS iterations, ResUNet prior, HyPaNet predicts alpha/beta from (sigma, sf).
  dpir   : DPIR (Zhang et al., TPAMI 2022) deblurring = HQS with the DRUNet denoiser drunet_color.pth,
           8 iterations, denoiser level log-spaced from 49 to sigma, alpha_k = 0.23 sigma^2 / sigma_k^2
           (the schedule of the paper); no x8 self-ensemble.
Both use the closed-form FFT data step re-implemented with torch.fft (the KAIR DataNet uses the removed
torch.rfft API); for scale factor 1 it is X = (conj(K) Y + a Z) / (|K|^2 + a). Images are reflect-padded by
2 x kernel size, exactly as for our unrolled networks, and cropped back.

The gate4 pipeline is reused unchanged (same images, noise seeds, calibration, real captures, outputs):
gate4.load / gate4.predict are redirected for the two pretrained models only.
    python eval_pretrained.py --models usrnet dpir --out-root outputs/gate4
"""

from paths import GATE_OUT, DIV2K_DIR, KAIR_DIR, PRETRAINED_DIR  # noqa: E402
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KAIR = KAIR_DIR
WEIGHTS = PRETRAINED_DIR
sys.path.insert(0, str(KAIR))
import models.basicblock as B          # noqa: E402  (KAIR building blocks, reviewed: torch only)
from models.network_unet import UNetRes  # noqa: E402

import gate4                            # noqa: E402
import eval_shift                       # noqa: E402
from degrade_torch import KSIZE         # noqa: E402

PRETRAINED = ("usrnet", "dpir")


# ---------------------------------------------------------------- USRNet modules (copied from KAIR, sf = 1)
class ResUNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=(64, 128, 256, 512), nb=2):
        super().__init__()
        self.m_head = B.conv(in_nc, nc[0], bias=False, mode="C")
        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode="CRC") for _ in range(nb)],
                                    B.downsample_strideconv(nc[0], nc[1], bias=False, mode="2"))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode="CRC") for _ in range(nb)],
                                    B.downsample_strideconv(nc[1], nc[2], bias=False, mode="2"))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode="CRC") for _ in range(nb)],
                                    B.downsample_strideconv(nc[2], nc[3], bias=False, mode="2"))
        self.m_body = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode="CRC") for _ in range(nb)])
        self.m_up3 = B.sequential(B.upsample_convtranspose(nc[3], nc[2], bias=False, mode="2"),
                                  *[B.ResBlock(nc[2], nc[2], bias=False, mode="CRC") for _ in range(nb)])
        self.m_up2 = B.sequential(B.upsample_convtranspose(nc[2], nc[1], bias=False, mode="2"),
                                  *[B.ResBlock(nc[1], nc[1], bias=False, mode="CRC") for _ in range(nb)])
        self.m_up1 = B.sequential(B.upsample_convtranspose(nc[1], nc[0], bias=False, mode="2"),
                                  *[B.ResBlock(nc[0], nc[0], bias=False, mode="CRC") for _ in range(nb)])
        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode="C")

    def forward(self, x):
        h, w = x.shape[-2:]
        x = nn.ReplicationPad2d((0, -w % 8, 0, -h % 8))(x)
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x[..., :h, :w]


class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=16, channel=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(in_nc, channel, 1), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, 1),
                                 nn.ReLU(inplace=True), nn.Conv2d(channel, out_nc, 1), nn.Softplus())

    def forward(self, x):
        return self.mlp(x) + 1e-6


class USRNet(nn.Module):
    def __init__(self, n_iter=8):
        super().__init__()
        self.p = ResUNet()
        self.h = HyPaNet(out_nc=2 * n_iter)
        self.n = n_iter


# ---------------------------------------------------------------- shared FFT data step
def _otf(k, hp, wp):
    b, ks, _ = k.shape
    kp = torch.zeros(b, hp, wp, device=k.device)
    kp[:, :ks, :ks] = k.float()
    return torch.fft.rfft2(torch.roll(kp, (-(ks // 2), -(ks // 2)), dims=(1, 2)))[:, None]


def unfold(y, k, alphas, prior):
    """HQS with closed-form data step; alphas: list of (B,) tensors; prior(x, i) -> x."""
    p = 2 * KSIZE
    h, w = y.shape[-2:]
    yp = F.pad(y.float(), [p] * 4, mode="reflect")
    hp, wp = yp.shape[-2:]
    H = _otf(k, hp, wp)
    HY, H2 = H.conj() * torch.fft.rfft2(yp), H.abs() ** 2
    x = yp
    for i, a in enumerate(alphas):
        a = a.float().view(-1, 1, 1, 1)
        x = torch.fft.irfft2((HY + a * torch.fft.rfft2(x)) / (H2 + a), s=(hp, wp))
        x = prior(x, i)
    return x[..., p:p + h, p:p + w]


def noise_level(s, floor):
    return torch.sqrt(s.float() ** 2 + floor ** 2) / 255


# ---------------------------------------------------------------- models
class Pretrained:
    def __init__(self, name, device):
        self.name, self.device = name, device
        if name == "usrnet":
            self.net = USRNet()
            sd = torch.load(WEIGHTS / "usrnet.pth", map_location=device)
            self.net.load_state_dict({k: v for k, v in sd.items() if not k.startswith("d.")}, strict=True)
        else:
            self.net = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode="R",
                               downsample_mode="strideconv", upsample_mode="convtranspose", bias=False)
            self.net.load_state_dict(torch.load(WEIGHTS / "drunet_color.pth", map_location=device), strict=True)
        self.net.to(device).eval()

    @torch.no_grad()
    def __call__(self, y, k, s, floor):
        sig = noise_level(s, floor)                              # (B,), 0-1 scale
        b = len(y)
        if self.name == "usrnet":
            ab = self.net.h(torch.stack([sig, torch.ones_like(sig)], 1)[..., None, None])[:, :, 0, 0]
            n = self.net.n

            def prior(x, i):
                beta = ab[:, n + i].view(-1, 1, 1, 1).expand(b, 1, *x.shape[-2:])
                return self.net.p(torch.cat([x, beta], 1))
            return unfold(y, k, [ab[:, i] for i in range(n)], prior)
        # DPIR schedule (all samples of a batch share sigma in our protocol; computed per sample anyway)
        n = 8
        s0 = torch.clamp(sig, min=0.255 / 255)
        lo, hi = np.log10(49.0), torch.log10(s0 * 255)
        levels = [10 ** (lo + (hi - lo) * i / (n - 1)) / 255 for i in range(n)]   # (B,) each
        alphas = [0.23 * s0 ** 2 / lv ** 2 for lv in levels]

        def prior(x, i):
            hh, ww = x.shape[-2:]
            xp = F.pad(x, (0, -ww % 8, 0, -hh % 8), mode="replicate")
            m = levels[i].view(-1, 1, 1, 1).expand(b, 1, *xp.shape[-2:])
            return self.net(torch.cat([xp, m], 1))[..., :hh, :ww]
        return unfold(y, k, alphas, prior)


_orig_load, _orig_predict = eval_shift.load, eval_shift.predict
_cache = {}


def load(run, device):
    run = Path(run)
    if run.name in PRETRAINED:
        if run.name not in _cache:
            _cache[run.name] = Pretrained(run.name, device)
        return _cache[run.name], {"cond": run.name, "seed": 0, "mode": "matched_effective_energy", "floor": 0.29}
    return _orig_load(run, device)


def predict(m, t, y, k, s):
    if t["cond"] in PRETRAINED:
        return m(y, k, s, t["floor"]).float()
    return _orig_predict(m, t, y, k, s)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(PRETRAINED), choices=PRETRAINED)
    p.add_argument("--out-root", default=str(GATE_OUT / "gate4"))
    p.add_argument("--ksets", nargs="*", default=None)
    p.add_argument("--smoke", action="store_true", help="load models and restore one image, then exit")
    a = p.parse_args()
    if a.smoke:
        from degrade_torch import degrade_pair, eval_kernel_sets
        from train_2x2 import build_cache, center_crops, psnr
        dev = torch.device("cuda")
        x = center_crops(build_cache((DIV2K_DIR / "DIV2K_valid_HR"), Path("cache/valid"))[:4], 256).to(dev)
        for name in a.models:
            m, t = load(Path(name), dev)
            for kset, kern in eval_kernel_sets().items():
                k = torch.from_numpy(kern[2]).float().to(dev).expand(4, -1, -1)
                for sigma in (2, 25):
                    s = torch.full((4,), float(sigma), device=dev)
                    y = degrade_pair(x, k, s, t["mode"], t["floor"], torch.Generator(device=dev).manual_seed(0))[0]
                    est = predict(m, t, y, k, s)
                    sl = (..., slice(10, -10), slice(10, -10))
                    print(f"{name} {kset} sigma={sigma}: input {psnr(y[sl].clamp(0, 1), x[sl]).mean():.2f} "
                          f"-> {psnr(est[sl].clamp(0, 1), x[sl]).mean():.2f} dB", flush=True)
        return
    gate4.load, gate4.predict = load, predict
    sys.argv = ["gate4.py", "--runs", *a.models, "--train-dir", str(DIV2K_DIR / "DIV2K_train_HR"),
                "--test-dir", str(DIV2K_DIR / "DIV2K_valid_HR"), "--out-root", a.out_root]
    if a.ksets:
        sys.argv += ["--ksets", *a.ksets]
    gate4.main()


if __name__ == "__main__":
    main()
