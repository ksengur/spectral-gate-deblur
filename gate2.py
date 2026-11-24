"""High-sigma gate: calibrate the data-consistency test in-distribution, add a blind fallback.

dc_gate.py used a fixed rho threshold (1.5). At sigma 15-25 a bad output still has
rho ~ 1 because the noise term dominates, so the gate never fires there. Here:
  1. Calibration (no test data, no OOD kernels): on dev images 0751-0800 with fresh
     random Gaussian training-distribution kernels, record rho of the conditioned
     network per sigma. Threshold tau(sigma) = 99th percentile -> about 1 % false alarms
     in distribution, by construction.
  2. Test (locked protocol, same seeds as eval_2x2.py): per image store PSNR and rho of
     the conditioned net, the blind net (runs/mix_none_s0) and the Wiener estimate, so
     every gate variant can be scored offline without rerunning networks.
rho = mean((k * x_hat - y)^2) / s^2 on the evaluation region, s^2 = sigma^2 + floor^2.
"""

import argparse
import csv
import zlib
from pathlib import Path

import numpy as np
import torch

from degrade_torch import (DESCRIPTORS, IN_CH, _blur, degrade_pair, eval_kernel_sets, model_input,
                           random_gaussian_kernel, wiener_deconv)
from eval_2x2 import EVAL_SIGMAS
from nafnet import UNKNOWN, NAFNet
from train_2x2 import build_cache, center_crops, psnr

QUANTILE = 0.99  # pre-registered calibration level


def load(run, device):
    ckpt = torch.load(run / "last.pt", map_location=device)
    t = ckpt["args"]
    m = NAFNet(width=t["width"], n_params=DESCRIPTORS[t["cond"]][0] if t["cond"] in DESCRIPTORS else 0,
               in_ch=IN_CH.get(t["cond"], 3)).to(device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    cond = t["cond"]

    def run_model(y, k, s, floor):
        tok = torch.full((len(y),), UNKNOWN, device=device, dtype=torch.long)
        params = DESCRIPTORS[cond][1](k, s) if cond in DESCRIPTORS else None
        return m(model_input(cond, y, k, s, floor), tok, params).float()
    return run_model, t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="conditioned run")
    p.add_argument("--blind", type=Path, default=Path("runs/mix_none_s0"))
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--n-calib-kernels", type=int, default=12)
    p.add_argument("--border", type=int, default=10)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond_net, t = load(args.run, device)
    blind_net, _ = load(args.blind, device)
    mode, floor = t["mode"], t["floor"]
    b = args.border
    sl = (..., slice(b, -b), slice(b, -b))
    s2 = lambda sigma: (sigma ** 2 + floor ** 2) / 255 ** 2
    rho = lambda est, y, k, sigma: ((_blur(est, k) - y)[sl].pow(2).mean((1, 2, 3)) / s2(sigma))

    # 1. calibration on dev images with in-distribution kernels
    dev_files = build_cache(args.train_dir, Path("cache/train"))[-50:]
    dev = center_crops(dev_files, 256)
    rng = np.random.default_rng(777)
    ckern = [random_gaussian_kernel(rng) for _ in range(args.n_calib_kernels)]
    tau = {}
    with torch.no_grad():
        for sigma in EVAL_SIGMAS:
            vals = []
            for kid, kern in enumerate(ckern):
                for i in range(0, len(dev), 10):
                    x = dev[i:i + 10].to(device)
                    n = len(x)
                    k = torch.from_numpy(kern).float().to(device).expand(n, -1, -1)
                    s = torch.full((n,), float(sigma), device=device)
                    gen = torch.Generator(device=device).manual_seed(zlib.crc32(f"calib|{kid}|{sigma}|{i}".encode()))
                    for y in degrade_pair(x, k, s, mode, floor, gen):
                        vals += rho(cond_net(y, k, s, floor), y, k, sigma).tolist()
            tau[sigma] = float(np.quantile(vals, QUANTILE))
            print(f"calib sigma={sigma}: median rho {np.median(vals):.4f}, tau {tau[sigma]:.4f}", flush=True)

    # 2. locked test protocol
    files = build_cache(args.test_dir, Path("cache/valid"))
    images = center_crops(files, 256)
    out = args.run / "gate2.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh, torch.no_grad():
        w = csv.writer(fh)
        w.writerow(["run", "image", "kset", "kid", "sigma", "order", "tau",
                    "psnr_cond", "psnr_blind", "psnr_wiener", "rho_cond", "rho_blind", "rho_wiener"])
        for kset, kernels in eval_kernel_sets().items():
            for kid, kern in enumerate(kernels):
                for sigma in EVAL_SIGMAS:
                    for i in range(0, len(images), 10):
                        x = images[i:i + 10].to(device)
                        n = len(x)
                        k = torch.from_numpy(kern).float().to(device).expand(n, -1, -1)
                        s = torch.full((n,), float(sigma), device=device)
                        gen = torch.Generator(device=device).manual_seed(
                            zlib.crc32(f"{kset}|{kid}|{sigma}|{i}".encode()))
                        for order, y in enumerate(degrade_pair(x, k, s, mode, floor, gen)):
                            ests = (cond_net(y, k, s, floor), blind_net(y, k, s, floor),
                                    wiener_deconv(y, k, s, floor))
                            ps = [psnr(e[sl].clamp(0, 1), x[sl]).tolist() for e in ests]
                            rh = [rho(e, y, k, sigma).tolist() for e in ests]
                            for j in range(n):
                                w.writerow([args.run.name, files[i + j].stem, kset, kid, sigma,
                                            "bn" if order == 0 else "nb", f"{tau[sigma]:.5f}"]
                                           + [f"{v[j]:.4f}" for v in ps] + [f"{v[j]:.5f}" for v in rh])
            print(f"{args.run.name}: {kset} done", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
