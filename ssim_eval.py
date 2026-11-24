"""SSIM for every model of the gate study, on exactly the gate4 rows (same crops, kernels, noise seeds, orders).

Writes <out-root>/<run>/ssim.csv  (image, kset, kid, sigma, order, ssim_net) and ssim_real.csv (pair, ssim_net),
and once <out-root>/_ref/ssim.csv / ssim_real.csv with the blind NAFNet (seed 0) and Wiener SSIM.
Gate SSIM = SSIM of the output the gate chose; the decisions are read from gate4.csv by the table script, so
the thresholds are the calibrated ones of the gate study (no re-calibration).

SSIM: Gaussian window 11x11, sigma 1.5, K1 0.01, K2 0.03, data range 1, mean over RGB channels, computed on the
same 10-pixel-border interior as PSNR. Real captures: grey SSIM at the integer shift (within +-8 px) that maximises
PSNR, 15-pixel border, as for the PSNR protocol.
    python ssim_eval.py --runs runs_cond/unrolled_s0 ... [--pretrained usrnet dpir]
"""

from paths import GATE_OUT, DIV2K_DIR  # noqa: E402
import argparse
import csv
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from degrade_torch import _blur, degrade_pair, eval_kernel_sets, wiener_deconv
from eval_2x2 import EVAL_SIGMAS
from ext_kernels import levin_real_pairs, shift_kernel_sets
from train_2x2 import build_cache, center_crops

OUT = (GATE_OUT / "ssim")


def _window(device, size=11, sigma=1.5):
    r = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-r ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a, b, border=10):
    """a, b: (B,C,H,W) in [0,1]; returns (B,) mean SSIM over channels and the interior."""
    if border:
        a, b = a[..., border:-border, border:-border], b[..., border:-border, border:-border]
    a, b = a.float(), b.float()
    c = a.shape[1]
    w = _window(a.device).expand(c, 1, -1, -1)
    mu_a, mu_b = F.conv2d(a, w, groups=c), F.conv2d(b, w, groups=c)
    saa = F.conv2d(a * a, w, groups=c) - mu_a ** 2
    sbb = F.conv2d(b * b, w, groups=c) - mu_b ** 2
    sab = F.conv2d(a * b, w, groups=c) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return m.mean((1, 2, 3))


def aligned_ssim(est, x, device, border=15, max_shift=8):
    """Grey SSIM at the PSNR-optimal integer shift (same alignment rule as the PSNR protocol)."""
    h, w = x.shape
    xc = x[border:h - border, border:w - border]
    best, arg = -1.0, (0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            e = est[border + dy:h - border + dy, border + dx:w - border + dx]
            p = -float(((e - xc) ** 2).mean())
            if p > best:
                best, arg = p, (dy, dx)
    dy, dx = arg
    e = est[border + dy:h - border + dy, border + dx:w - border + dx]
    t = lambda z: torch.from_numpy(np.ascontiguousarray(z)).float().to(device)[None, None]
    return float(ssim(t(e), t(xc), border=0)[0])


def rows_for(model_fn, images, ksets, device, mode, floor):
    sl_out = []
    with torch.no_grad():
        for kset, kernels in ksets.items():
            for kid, kern in enumerate(kernels):
                for sigma in EVAL_SIGMAS:
                    for i in range(0, len(images), 10):
                        x = images[i:i + 10].to(device)
                        n = len(x)
                        k = torch.from_numpy(kern).float().to(device).expand(n, -1, -1)
                        s = torch.full((n,), float(sigma), device=device)
                        gen = torch.Generator(device=device).manual_seed(zlib.crc32(f"{kset}|{kid}|{sigma}|{i}".encode()))
                        for order, y in enumerate(degrade_pair(x, k, s, mode, floor, gen)):
                            vals = model_fn(y, k, s, x)
                            for j in range(n):
                                sl_out.append((i + j, kset, kid, sigma, "bn" if order == 0 else "nb",
                                               *[v[j] for v in vals]))
            print(f"  {kset} done", flush=True)
    return sl_out


def real_rows(model_fn, device):
    out = []
    with torch.no_grad():
        for name, x, y, kern in levin_real_pairs():
            k = torch.from_numpy(kern).float().to(device)[None]
            yt = torch.from_numpy(y).to(device)[None, None].expand(1, 3, -1, -1).contiguous()
            xt = torch.from_numpy(x).to(device)[None, None].expand(1, 3, -1, -1).contiguous()
            bb = _blur(xt, k)[0, 0].cpu().numpy()
            sigma = min(float((np.roll(bb, (dy, dx), (0, 1)) - y)[25:-25, 25:-25].std() * 255)
                        for dy in range(-8, 9) for dx in range(-8, 9))
            s = torch.tensor([max(sigma, 0.5)], device=device)
            ests = model_fn(yt, k, s, None, real=True)
            out.append((name, *[aligned_ssim(e.clamp(0, 1).mean(1)[0].cpu().numpy(), x, device) for e in ests]))
    return out


def write(path, header, rows, files=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            r = list(r)
            if files is not None:
                r[0] = files[r[0]].stem
            w.writerow([f"{v:.5f}" if isinstance(v, float) else v for v in r])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="*", default=[])
    p.add_argument("--pretrained", nargs="*", default=[])
    p.add_argument("--ref", action="store_true", help="also write blind NAFNet (s0) and Wiener SSIM")
    p.add_argument("--limit", type=int, default=None, help="first N test images only (smoke test)")
    a = p.parse_args()
    device = torch.device("cuda")
    files = build_cache((DIV2K_DIR / "DIV2K_valid_HR"), Path("cache/valid"))
    images = center_crops(files, 256)
    if a.limit:
        images = images[:a.limit]
    ksets = dict(eval_kernel_sets(), **shift_kernel_sets())
    if a.pretrained:
        import eval_pretrained as ep
        load, predict = ep.load, ep.predict
    else:
        from eval_shift import load, predict
    floor, mode = 0.29, "matched_effective_energy"

    if a.ref:
        bm, bt = load(Path("runs/mix_none_s0"), device)

        def ref_fn(y, k, s, x, real=False):
            outs = [predict(bm, bt, y, k, s), wiener_deconv(y, k, s, floor)]
            return outs if real else [ssim(o.clamp(0, 1), x).tolist() for o in outs]
        print("reference", flush=True)
        write(OUT / "_ref" / "ssim.csv", ["image", "kset", "kid", "sigma", "order", "ssim_blind", "ssim_wiener"],
              rows_for(ref_fn, images, ksets, device, mode, floor), files)
        write(OUT / "_ref" / "ssim_real.csv", ["pair", "ssim_blind", "ssim_wiener"], real_rows(ref_fn, device))

    for run in list(a.runs) + [Path(n) for n in a.pretrained]:
        m, t = load(run, device)
        print(run.name, flush=True)

        def fn(y, k, s, x, real=False):
            o = predict(m, t, y, k, s)
            return [o] if real else [ssim(o.clamp(0, 1), x).tolist()]
        write(OUT / run.name / "ssim.csv", ["image", "kset", "kid", "sigma", "order", "ssim_net"],
              rows_for(fn, images, ksets, device, t["mode"], t["floor"]), files)
        write(OUT / run.name / "ssim_real.csv", ["pair", "ssim_net"], real_rows(fn, device))
        print(f"{run.name}: wrote {OUT / run.name}", flush=True)


if __name__ == "__main__":
    main()
