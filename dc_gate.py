"""Test-time data-consistency remedy, no retraining.

Finding that motivates it: on OOD motion kernels the conditioned networks are worse
than the plain Wiener estimate they could have used (e.g. wiener_s0 21.7 dB vs its own
Wiener input 29.2 dB at sigma=2). Because the kernel is known at test time, the output
can be checked against the observation:
    rho = mean((k * x_hat - y)^2) / s^2,   s^2 = sigma^2 + floor^2  (0-1 scale)
A faithful estimate has rho ~ 1 (both orders have total noise variance s^2 under
matched effective energy). Variants written per image:
  net    : network output
  wiener : Wiener estimate alone (same as the wiener run's input channel)
  gate   : net if rho <= TAU else wiener   (TAU fixed in advance, not tuned on test)
  dc     : net + one Wiener step on the residual: x_hat + W(y - k * x_hat)
Locked eval protocol: same crops, kernels, sigmas and noise seeds as eval_2x2.py.
"""

import argparse
import csv
import zlib
from pathlib import Path

import torch

from degrade_torch import (DESCRIPTORS, IN_CH, _blur, degrade_pair, eval_kernel_sets, model_input,
                           wiener_deconv)
from eval_2x2 import EVAL_SIGMAS
from nafnet import UNKNOWN, NAFNet
from train_2x2 import build_cache, center_crops, psnr

TAU = 1.5  # pre-registered gate threshold


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("cache/valid"))
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--border", type=int, default=10)
    p.add_argument("--batch", type=int, default=10)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.run / "last.pt", map_location=device)
    targs = ckpt["args"]
    cond, floor = targs["cond"], targs["floor"]
    model = NAFNet(width=targs["width"], n_params=DESCRIPTORS[cond][0] if cond in DESCRIPTORS else 0,
                   in_ch=IN_CH.get(cond, 3)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    files = build_cache(args.test_dir, args.cache)
    images = center_crops(files, args.crop)
    b = args.border
    sl = (..., slice(b, -b or None), slice(b, -b or None))
    out = args.run / "dc_gate.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh, torch.no_grad():
        w = csv.writer(fh)
        w.writerow(["run", "cond", "image", "kset", "kid", "sigma", "order", "rho", "variant", "psnr"])
        for kset, kernels in eval_kernel_sets().items():
            for kid, kern in enumerate(kernels):
                for sigma in EVAL_SIGMAS:
                    for i in range(0, len(images), args.batch):
                        x = images[i:i + args.batch].to(device)
                        n = len(x)
                        k = torch.from_numpy(kern).float().to(device).expand(n, -1, -1)
                        gen = torch.Generator(device=device).manual_seed(
                            zlib.crc32(f"{kset}|{kid}|{sigma}|{i}".encode()))
                        s = torch.full((n,), float(sigma), device=device)
                        pair = degrade_pair(x, k, s, targs["mode"], floor, gen)
                        for order, y in enumerate(pair):
                            tok = order if cond == "oracle" else UNKNOWN
                            params = DESCRIPTORS[cond][1](k, s) if cond in DESCRIPTORS else None
                            net = model(model_input(cond, y, k, s, floor),
                                        torch.full((n,), tok, device=device, dtype=torch.long), params).float()
                            wien = wiener_deconv(y, k, s, floor)
                            s2 = (sigma ** 2 + floor ** 2) / 255 ** 2
                            rho = (_blur(net, k) - y)[sl].pow(2).mean((1, 2, 3)) / s2
                            gate = torch.where((rho <= TAU)[:, None, None, None], net, wien)
                            dc = net + wiener_deconv(y - _blur(net, k), k, s, floor)
                            for name, est in (("net", net), ("wiener", wien), ("gate", gate), ("dc", dc)):
                                for j, v in enumerate(psnr(est[sl].clamp(0, 1), x[sl]).tolist()):
                                    w.writerow([args.run.name, cond, files[i + j].stem, kset, kid, sigma,
                                                "bn" if order == 0 else "nb", f"{rho[j]:.3f}", name, f"{v:.4f}"])
                print(f"{args.run.name}: {kset} done", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
