"""Two-sided gate: range-space consistency (rho) + null-space plausibility (nu). No retraining.

gate3 showed the residual test rho = ||k*x_hat - y||^2 / s^2 catches embedding-conditioned failures
(errors the blur can see) but not data-step failures: an unrolled data step enforces k*x_hat ~ y by
construction, so its errors sit where |H| is small (the near-null space of k, e.g. the spectral
zero rings of a defocus disk) and vanish under re-blurring.

nu measures exactly that part: the deviation of x_hat from the regularised Wiener estimate x_W,
restricted to the frequencies the Wiener prior controls,
    W(f)  = lam |G(f)|^2 / (|H(f)|^2 + lam |G(f)|^2)  in [0, 1],  lam = alpha s^2 (alpha 300)
    nu    = mean_f |W(f) FFT(x_hat - x_W)(f)|^2 / (h w)     (mean pixel energy, Parseval)
W ~ 1 where the kernel carries no information and ~ 0 where the data determine the image, so nu
ignores detail that is supported by the data and penalises only unsupported content.

Both thresholds are calibrated exactly like gate3 (99 % quantile of THIS model on dev images 0751-0800
with 12 fresh Gaussian training-distribution kernels, both orders, sigma grid, log-sigma
interpolation). Accept the network iff rho <= tau_rho AND nu <= tau_nu, else fall back to Wiener.
Writes <out>/gate4.csv and <out>/gate4_real.csv (same rows as gate3 plus nu, tau_nu).
"""

import argparse
import csv
import zlib
from pathlib import Path

import numpy as np
import torch

from degrade_torch import degrade_pair, eval_kernel_sets, motion_kernel, random_gaussian_kernel, wiener_deconv
from eval_2x2 import EVAL_SIGMAS
from eval_shift import load, predict
from ext_kernels import levin_real_pairs, shift_kernel_sets
from gate3 import CALIB_SIGMAS, QUANTILE, real_psnr, rho_fn, tau_at
from train_2x2 import build_cache, center_crops, psnr

ALPHA = 300.0


def null_weight(k, h, w, sigma, floor):
    """W(f) on an h x w grid for kernels k (B,K,K); returns (B,1,h,w//2+1)."""
    b, ks, _ = k.shape
    kp = torch.zeros(b, h, w, device=k.device)
    kp[:, :ks, :ks] = k.float()
    kp = torch.roll(kp, (-(ks // 2), -(ks // 2)), dims=(1, 2))
    H2 = torch.fft.rfft2(kp).abs()[:, None] ** 2
    gx = torch.zeros(h, w, device=k.device)
    gx[0, 0], gx[0, 1] = -1, 1
    gy = torch.zeros(h, w, device=k.device)
    gy[0, 0], gy[1, 0] = -1, 1
    G2 = torch.fft.rfft2(gx).abs() ** 2 + torch.fft.rfft2(gy).abs() ** 2
    lam = ALPHA * (sigma ** 2 + floor ** 2) / 255 ** 2
    return lam * G2 / (H2 + lam * G2 + 1e-12)


def nu_fn(est, xw, k, sigma, floor, sl):
    d = (est.float() - xw.float())[sl]
    h, w = d.shape[-2:]
    Wf = null_weight(k, h, w, sigma, floor)
    D = torch.fft.rfft2(d)
    # rfft2 halves the spectrum; weight the non-redundant columns twice (Parseval for real signals)
    cw = torch.full((D.shape[-1],), 2.0, device=d.device)
    cw[0] = 1.0
    if w % 2 == 0:
        cw[-1] = 1.0
    return ((Wf * D).abs() ** 2 * cw).sum((1, 2, 3)) / (h * w) ** 2 / d.shape[1]


def calibrate(m, t, dev, device, sl, n_kernels=12, kernels="gauss"):
    """kernels='mix': half Gaussian, half motion, i.e. the training distribution of --train-kernels mix models."""
    rng = np.random.default_rng(777)
    if kernels == "mix":
        ckern = ([random_gaussian_kernel(rng) for _ in range(n_kernels // 2)]
                 + [motion_kernel(rng) for _ in range(n_kernels - n_kernels // 2)])
    else:
        ckern = [random_gaussian_kernel(rng) for _ in range(n_kernels)]
    tau_r, tau_n = {}, {}
    with torch.no_grad():
        for sigma in CALIB_SIGMAS:
            vr, vn = [], []
            for kid, kern in enumerate(ckern):
                for i in range(0, len(dev), 10):
                    x = dev[i:i + 10].to(device)
                    n = len(x)
                    k = torch.from_numpy(kern).float().to(device).expand(n, -1, -1)
                    s = torch.full((n,), float(sigma), device=device)
                    gen = torch.Generator(device=device).manual_seed(zlib.crc32(f"calib|{kid}|{sigma}|{i}".encode()))
                    for y in degrade_pair(x, k, s, t["mode"], t["floor"], gen):
                        est = predict(m, t, y, k, s)
                        xw = wiener_deconv(y, k, s, t["floor"], alpha=ALPHA)
                        vr += rho_fn(est, y, k, sigma, t["floor"], sl).tolist()
                        vn += nu_fn(est, xw, k, sigma, t["floor"], sl).tolist()
            tau_r[sigma] = float(np.quantile(vr, QUANTILE))
            tau_n[sigma] = float(np.quantile(vn, QUANTILE))
    return tau_r, tau_n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--blind", type=Path, default=Path("runs/mix_none_s0"))
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--border", type=int, default=10)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--calib-kernels", choices=("gauss", "mix"), default="gauss",
                   help="calibration kernels: the model's training distribution")
    p.add_argument("--real-sigma", choices=("oracle", "mad", "immerkaer"), default="oracle",
                   help="noise level used on the real captures (blind estimates from y alone: real_sigma.py); "
                        "non-oracle runs skip the synthetic sets, use a separate --out-root")
    p.add_argument("--nu-alpha", type=float, default=300.0,
                   help="Wiener weight of the nu score (W and its reference); the fallback keeps 300")
    p.add_argument("--ksets", nargs="*", default=None, help="subset of kernel sets (default all)")
    args = p.parse_args()
    global ALPHA
    ALPHA = args.nu_alpha
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b = args.border
    sl = (..., slice(b, -b), slice(b, -b))
    dev = center_crops(build_cache(args.train_dir, Path("cache/train"))[-50:], 256)
    files = build_cache(args.test_dir, Path("cache/valid"))
    images = center_crops(files, 256)
    ksets = dict(eval_kernel_sets(), **shift_kernel_sets())
    if args.real_sigma != "oracle":
        ksets = {}
    if args.ksets is not None:
        ksets = {k: v for k, v in ksets.items() if k in args.ksets}
    blind_m, bt = load(args.blind, device)
    floor = bt["floor"]

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
            from degrade_torch import _blur
            bb = _blur(xt, k)[0, 0].cpu().numpy()
            sigma = min(float((np.roll(bb, (dy, dx), (0, 1)) - y)[25:-25, 25:-25].std() * 255)
                        for dy in range(-8, 9) for dx in range(-8, 9))
            if args.real_sigma != "oracle":
                from real_sigma import immerkaer, mad
                sigma = (mad if args.real_sigma == "mad" else immerkaer)(y)
            s = torch.tensor([max(sigma, 0.5)], device=device)
            xw = wiener_deconv(yt, k, s, floor, alpha=ALPHA)
            real.append((name, x, yt, k, s, sigma, xw, real_psnr(predict(blind_m, bt, yt, k, s), x),
                         real_psnr(wiener_deconv(yt, k, s, floor), x)))

    for run in args.runs:
        m, t = load(run, device)
        od = args.out_root / run.name if args.out_root else run
        od.mkdir(parents=True, exist_ok=True)
        tau_r, tau_n = calibrate(m, t, dev, device, sl, kernels=args.calib_kernels)
        print(f"{run.name}: tau_rho " + " ".join(f"{s}:{v:.3f}" for s, v in tau_r.items())
              + " | tau_nu " + " ".join(f"{s}:{v:.2e}" for s, v in tau_n.items()), flush=True)
        with open(od / "gate4.csv", "w", newline="", encoding="utf-8") as fh, torch.no_grad():
            w = csv.writer(fh)
            w.writerow(["run", "cond", "seed", "image", "kset", "kid", "sigma", "order", "tau", "rho", "tau_nu", "nu",
                        "psnr_net", "psnr_blind", "psnr_wiener"])
            for kset, kernels in ksets.items():
                for kid, kern in enumerate(kernels):
                    for sigma in EVAL_SIGMAS:
                        tr, tn = tau_at(tau_r, sigma), tau_at(tau_n, sigma)
                        for i in range(0, len(images), 10):
                            x = images[i:i + 10].to(device)
                            n = len(x)
                            k = torch.from_numpy(kern).float().to(device).expand(n, -1, -1)
                            s = torch.full((n,), float(sigma), device=device)
                            gen = torch.Generator(device=device).manual_seed(
                                zlib.crc32(f"{kset}|{kid}|{sigma}|{i}".encode()))
                            for order, y in enumerate(degrade_pair(x, k, s, t["mode"], t["floor"], gen)):
                                est = predict(m, t, y, k, s)
                                xw = wiener_deconv(y, k, s, t["floor"], alpha=ALPHA)
                                rh = rho_fn(est, y, k, sigma, t["floor"], sl).tolist()
                                nu = nu_fn(est, xw, k, sigma, t["floor"], sl).tolist()
                                pn = psnr(est[sl].clamp(0, 1), x[sl]).tolist()
                                pb, pw = ref[(kset, kid, sigma, i, order)]
                                for j in range(n):
                                    w.writerow([run.name, t["cond"], t["seed"], files[i + j].stem, kset, kid, sigma,
                                                "bn" if order == 0 else "nb", f"{tr:.4f}", f"{rh[j]:.4f}",
                                                f"{tn:.4e}", f"{nu[j]:.4e}", f"{pn[j]:.4f}", f"{pb[j]:.4f}",
                                                f"{pw[j]:.4f}"])
                print(f"{run.name}: {kset} done", flush=True)
        rs = (..., slice(15, -15), slice(15, -15))
        with open(od / "gate4_real.csv", "w", newline="", encoding="utf-8") as fr, torch.no_grad():
            wr = csv.writer(fr)
            wr.writerow(["run", "cond", "seed", "pair", "sigma_est", "tau", "rho", "tau_nu", "nu", "psnr_net",
                         "psnr_blind", "psnr_wiener"])
            for name, x, yt, k, s, sigma, xw, pbl, pwi in real:
                est = predict(m, t, yt, k, s)
                sv = float(s)
                wr.writerow([run.name, t["cond"], t["seed"], name, f"{sigma:.3f}", f"{tau_at(tau_r, sv):.4f}",
                             f"{float(rho_fn(est, yt, k, sv, t['floor'], rs)):.4f}", f"{tau_at(tau_n, sv):.4e}",
                             f"{float(nu_fn(est, xw, k, sv, t['floor'], rs)):.4e}", f"{real_psnr(est, x):.4f}",
                             f"{pbl:.4f}", f"{pwi:.4f}"])
        print(f"{run.name}: wrote {od}", flush=True)


if __name__ == "__main__":
    main()
