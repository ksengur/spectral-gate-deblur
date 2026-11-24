"""Calibrated data-consistency gate for every model and every kernel shift (no retraining).

Generalises gate2.py (NAFNet only, locked kernels) to any run, incl. the unrolled arms, and to the
extra shifts of ext_kernels.py and the real Levin captures.

  rho(x_hat) = mean((k * x_hat - y)^2) / s^2,  s^2 = sigma^2 + floor^2 (0-1 scale), eval region only
  tau(sigma) = 99 % quantile of rho of THIS model on dev images 0751-0800 with 12 fresh random
               training-distribution (Gaussian) kernels, both orders. Calibrated on a sigma grid and
               interpolated in log-sigma, so the real captures (estimated sigma) get a threshold too.
               No test image and no OOD kernel is used for calibration (pre-registered QUANTILE 0.99).
  gate     : keep the network output if rho <= tau(sigma), else fall back to
             (a) plain Wiener (alpha 300, as degrade_torch.wiener_deconv) or (b) the blind NAFNet.

Synthetic test: locked protocol (same crops, sigmas, both orders, crc32 noise seeds keyed on the
kernel-set name) for kernel sets gauss, motion (eval_kernel_sets) and defocus, levin (ext_kernels).
Writes <out>/gate3.csv (per image) and <out>/gate3_real.csv (28 real Levin captures), where <out> is
the run directory or <out-root>/<run name>.
"""

import argparse
import csv
import zlib
from pathlib import Path

import numpy as np
import torch

from degrade_torch import _blur, degrade_pair, eval_kernel_sets, random_gaussian_kernel, wiener_deconv
from eval_2x2 import EVAL_SIGMAS
from eval_shift import aligned_psnr, load, predict
from ext_kernels import levin_real_pairs, shift_kernel_sets
from train_2x2 import build_cache, center_crops, psnr

QUANTILE = 0.99
CALIB_SIGMAS = (1, 2, 3, 5, 10, 15, 25)


def rho_fn(est, y, k, sigma, floor, sl):
    s2 = (sigma ** 2 + floor ** 2) / 255 ** 2
    return (_blur(est, k) - y)[sl].pow(2).mean((1, 2, 3)) / s2


def calibrate(m, t, dev, device, sl, n_kernels=12):
    rng = np.random.default_rng(777)
    ckern = [random_gaussian_kernel(rng) for _ in range(n_kernels)]
    tau = {}
    with torch.no_grad():
        for sigma in CALIB_SIGMAS:
            vals = []
            for kid, kern in enumerate(ckern):
                for i in range(0, len(dev), 10):
                    x = dev[i:i + 10].to(device)
                    n = len(x)
                    k = torch.from_numpy(kern).float().to(device).expand(n, -1, -1)
                    s = torch.full((n,), float(sigma), device=device)
                    gen = torch.Generator(device=device).manual_seed(zlib.crc32(f"calib|{kid}|{sigma}|{i}".encode()))
                    for y in degrade_pair(x, k, s, t["mode"], t["floor"], gen):
                        est = predict(m, t, y, k, s) if m is not None else wiener_deconv(y, k, s, t["floor"])
                        vals += rho_fn(est, y, k, sigma, t["floor"], sl).tolist()
            tau[sigma] = float(np.quantile(vals, QUANTILE))
    return tau


def tau_at(tau, sigma):
    xs = np.log(list(tau))
    return float(np.interp(np.log(max(sigma, CALIB_SIGMAS[0])), xs, list(tau.values())))


def real_psnr(est, x):
    return aligned_psnr(est.clamp(0, 1).mean(1)[0].cpu().numpy(), x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--blind", type=Path, default=Path("runs/mix_none_s0"))
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--border", type=int, default=10)
    p.add_argument("--out-root", type=Path, default=None,
                   help="write <out-root>/<run name>/gate3*.csv instead of into the run directory")
    p.add_argument("--real-only", action="store_true", help="skip the synthetic sets (redo gate3_real.csv)")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b = args.border
    sl = (..., slice(b, -b), slice(b, -b))
    dev = center_crops(build_cache(args.train_dir, Path("cache/train"))[-50:], 256)
    files = build_cache(args.test_dir, Path("cache/valid"))
    images = center_crops(files, 256)
    ksets = {} if args.real_only else dict(eval_kernel_sets(), **shift_kernel_sets())
    blind_m, bt = load(args.blind, device)
    floor = bt["floor"]

    # reference outputs (blind NAFNet, Wiener) do not depend on the gated model: compute once
    print("reference pass", flush=True)
    ref = {}
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
                        for order, y in enumerate(degrade_pair(x, k, s, bt["mode"], floor, gen)):
                            bl, wi = predict(blind_m, bt, y, k, s), wiener_deconv(y, k, s, floor)
                            ref[(kset, kid, sigma, i, order)] = (
                                psnr(bl[sl].clamp(0, 1), x[sl]).cpu(), psnr(wi[sl].clamp(0, 1), x[sl]).cpu())
        real = []
        for name, x, y, kern in levin_real_pairs():
            k = torch.from_numpy(kern).float().to(device)[None]
            yt = torch.from_numpy(y).to(device)[None, None].expand(1, 3, -1, -1).contiguous()
            xt = torch.from_numpy(x).to(device)[None, None].expand(1, 3, -1, -1).contiguous()
            bb = _blur(xt, k)[0, 0].cpu().numpy()
            sigma = min(float((np.roll(bb, (dy, dx), (0, 1)) - y)[25:-25, 25:-25].std() * 255)
                        for dy in range(-8, 9) for dx in range(-8, 9))
            s = torch.tensor([max(sigma, 0.5)], device=device)
            real.append((name, x, yt, k, s, sigma, real_psnr(predict(blind_m, bt, yt, k, s), x),
                         real_psnr(wiener_deconv(yt, k, s, floor), x)))

    for run in args.runs:
        m, t = load(run, device)
        od = args.out_root / run.name if args.out_root else run
        od.mkdir(parents=True, exist_ok=True)
        tau = calibrate(m, t, dev, device, sl)
        print(f"{run.name}: tau " + " ".join(f"{s}:{v:.3f}" for s, v in tau.items()), flush=True)
        if ksets:
            with open(od / "gate3.csv", "w", newline="", encoding="utf-8") as fh, torch.no_grad():
                w = csv.writer(fh)
                w.writerow(["run", "cond", "seed", "image", "kset", "kid", "sigma", "order", "tau", "rho",
                            "psnr_net", "psnr_blind", "psnr_wiener"])
                for kset, kernels in ksets.items():
                    for kid, kern in enumerate(kernels):
                        for sigma in EVAL_SIGMAS:
                            ts = tau_at(tau, sigma)
                            for i in range(0, len(images), 10):
                                x = images[i:i + 10].to(device)
                                n = len(x)
                                k = torch.from_numpy(kern).float().to(device).expand(n, -1, -1)
                                s = torch.full((n,), float(sigma), device=device)
                                gen = torch.Generator(device=device).manual_seed(
                                    zlib.crc32(f"{kset}|{kid}|{sigma}|{i}".encode()))
                                for order, y in enumerate(degrade_pair(x, k, s, t["mode"], t["floor"], gen)):
                                    est = predict(m, t, y, k, s)
                                    rh = rho_fn(est, y, k, sigma, t["floor"], sl).tolist()
                                    pn = psnr(est[sl].clamp(0, 1), x[sl]).tolist()
                                    pb, pw = ref[(kset, kid, sigma, i, order)]
                                    for j in range(n):
                                        w.writerow([run.name, t["cond"], t["seed"], files[i + j].stem, kset, kid,
                                                    sigma, "bn" if order == 0 else "nb", f"{ts:.4f}", f"{rh[j]:.4f}",
                                                    f"{pn[j]:.4f}", f"{pb[j]:.4f}", f"{pw[j]:.4f}"])
                    print(f"{run.name}: {kset} done", flush=True)
        with open(od / "gate3_real.csv", "w", newline="", encoding="utf-8") as fr, torch.no_grad():
            wr = csv.writer(fr)
            wr.writerow(["run", "cond", "seed", "pair", "sigma_est", "tau", "rho", "psnr_net", "psnr_blind",
                         "psnr_wiener"])
            for name, x, yt, k, s, sigma, pbl, pwi in real:
                est = predict(m, t, yt, k, s)
                rh = float(rho_fn(est, yt, k, float(s), t["floor"], (..., slice(15, -15), slice(15, -15))))
                wr.writerow([run.name, t["cond"], t["seed"], name, f"{sigma:.3f}", f"{tau_at(tau, float(s)):.4f}",
                             f"{rh:.4f}", f"{real_psnr(est, x):.4f}", f"{pbl:.4f}", f"{pwi:.4f}"])
        print(f"{run.name}: wrote {od}", flush=True)


if __name__ == "__main__":
    main()
