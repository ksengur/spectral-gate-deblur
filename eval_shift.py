"""Extra-shift evaluation (no retraining): defocus and Levin kernels on the locked DIV2K test crops,
and the real captured Levin et al. (2009) blurred photographs.

Synthetic part: identical to eval_2x2.py (same crops, sigmas, both orders, crc32 noise seeds keyed on
the kernel-set name) but with the kernel families from ext_kernels.py, true descriptor only.
Writes <run>/eval_shift.csv (same columns as eval.csv).

Real part (--real): 28 real camera-shake captures (4 images x 7 kernels, grey). The grey image is
replicated to RGB, the network gets the measured kernel and sigma estimated from the ground-truth
residual (std of y - k*x, oracle like the synthetic setting). PSNR on the grey mean after the best
integer shift within +-8 px (standard for this data set), 15 px border.
Writes <run>/eval_levin_real.csv and saves restored images for the qualitative figure.
"""

import argparse
import csv
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from degrade_torch import DESCRIPTORS, IN_CH, _blur, degrade_pair, model_input, wiener_deconv
from eval_2x2 import EVAL_SIGMAS
from ext_kernels import levin_real_pairs, shift_kernel_sets
from nafnet import UNKNOWN
from train_2x2 import build_cache, center_crops, psnr
from unrolled import build_model


def load(run, device):
    ckpt = torch.load(run / "last.pt", map_location=device)
    t = ckpt["args"]
    cond = t["cond"]
    m = build_model(cond, t["width"], DESCRIPTORS[cond][0] if cond in DESCRIPTORS else 0,
                    IN_CH.get(cond, 3), t["floor"]).to(device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m, t


def predict(m, t, y, k, s):
    """True-descriptor prediction; blind models get the UNKNOWN token."""
    cond, n = t["cond"], len(y)
    if cond == "oracle":
        raise ValueError("order-oracle runs are not part of the shift study")
    params = DESCRIPTORS[cond][1](k, s) if cond in DESCRIPTORS else None
    tok = torch.full((n,), UNKNOWN, device=y.device, dtype=torch.long)
    return m(model_input(cond, y, k, s, t["floor"]), tok, params).float()


def synthetic(run, m, t, images, files, device, border=10, batch=10):
    sl = (..., slice(border, -border), slice(border, -border))
    out = run / "eval_shift.csv"
    tmp = out.with_suffix(".partial")
    with open(tmp, "w", newline="", encoding="utf-8") as fh, torch.no_grad():
        w = csv.writer(fh)
        w.writerow(["run", "train_order", "cond", "token_dropout", "seed", "mode", "floor", "token", "image",
                    "kset", "kid", "sigma", "order", "psnr"])
        for kset, kernels in shift_kernel_sets().items():
            for kid, kern in enumerate(kernels):
                for sigma in EVAL_SIGMAS:
                    for i in range(0, len(images), batch):
                        x = images[i:i + batch].to(device)
                        n = len(x)
                        k = torch.from_numpy(kern).float().to(device).expand(n, -1, -1)
                        s = torch.full((n,), float(sigma), device=device)
                        gen = torch.Generator(device=device).manual_seed(
                            zlib.crc32(f"{kset}|{kid}|{sigma}|{i}".encode()))
                        for order, y in enumerate(degrade_pair(x, k, s, t["mode"], t["floor"], gen)):
                            est = predict(m, t, y, k, s) if m is not None else wiener_deconv(y, k, s, t["floor"])
                            for j, v in enumerate(psnr(est[sl].clamp(0, 1), x[sl]).tolist()):
                                w.writerow([run.name, t["train_order"], t["cond"], t.get("token_dropout", 0.0),
                                            t["seed"], t["mode"], t["floor"],
                                            "true" if t["cond"] != "none" else "unknown", files[i + j].stem,
                                            kset, kid, sigma, "bn" if order == 0 else "nb", f"{v:.4f}"])
            print(f"{run.name}: {kset} done", flush=True)
    tmp.replace(out)


def aligned_psnr(est, x, border=15, max_shift=8):
    """Best PSNR over integer shifts of est (grey, 0-1) against x."""
    best = -1.0
    h, w = x.shape
    xc = x[border:h - border, border:w - border]
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            e = est[border + dy:h - border + dy, border + dx:w - border + dx]
            mse = float(((e - xc) ** 2).mean())
            best = max(best, 10 * np.log10(1.0 / max(mse, 1e-12)))
    return best


def real(run, m, t, device, save_dir=None):
    out = run / "eval_levin_real.csv"
    rows = []
    with torch.no_grad():
        for name, x, y, kern in levin_real_pairs():
            k = torch.from_numpy(kern).float().to(device)[None]
            yt = torch.from_numpy(y).to(device)[None, None].expand(1, 3, -1, -1).contiguous()
            xt = torch.from_numpy(x).to(device)[None, None].expand(1, 3, -1, -1).contiguous()
            # sigma (0-255) from the ground-truth residual at the best integer alignment (the fitted
            # kernel is centroid-centred, so k*x and y can be offset by a few pixels)
            b = _blur(xt, k)[0, 0].cpu().numpy()
            sigma = min(float((np.roll(b, (dy, dx), (0, 1)) - y)[25:-25, 25:-25].std() * 255)
                        for dy in range(-8, 9) for dx in range(-8, 9))
            s = torch.tensor([max(sigma, 0.5)], device=device)
            est = predict(m, t, yt, k, s) if m is not None else wiener_deconv(yt, k, s, t["floor"])
            g = est.clamp(0, 1).mean(1)[0].cpu().numpy()
            rows.append([run.name, t["cond"], t["seed"], name, f"{sigma:.3f}", f"{aligned_psnr(g, x):.4f}",
                         f"{aligned_psnr(y, x):.4f}"])
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                np.save(save_dir / f"{name}.npy", g.astype(np.float16))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "cond", "seed", "pair", "sigma_est", "psnr", "psnr_input"])
        w.writerows(rows)
    print(f"{run.name}: levin real done, mean {np.mean([float(r[5]) for r in rows]):.2f} dB", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--wiener-ref", type=Path, default=None,
                   help="also evaluate plain Wiener (alpha=300) using this run's degradation settings; "
                        "results go to <dir>/eval_shift.csv")
    p.add_argument("--skip-synthetic", action="store_true")
    p.add_argument("--save-real", action="store_true", help="save restored real images (reports/levin_real)")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = build_cache(args.test_dir, Path("cache/valid"))
    images = center_crops(files, 256)
    jobs = [(r, *load(r, device)) for r in args.runs]
    if args.wiener_ref:
        _, t = load(args.runs[0], device)
        wt = dict(t, cond="wiener_only", seed=0, token_dropout=0.0)
        args.wiener_ref.mkdir(parents=True, exist_ok=True)
        jobs.append((args.wiener_ref, None, wt))
    for run, m, t in jobs:
        if not args.skip_synthetic and not (run / "eval_shift.csv").exists():
            synthetic(run, m, t, images, files, device)
        real(run, m, t, device, Path("reports/levin_real") / run.name if args.save_real else None)


if __name__ == "__main__":
    main()
