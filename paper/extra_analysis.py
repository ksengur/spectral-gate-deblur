"""Supplementary analyses for the paper (read-only w.r.t. runs; writes reports/ and outputs/calib).

  calib   : dump the in-distribution calibration scores (rho, nu) of every gated model, so the gate
            quantile can be varied post hoc (ablation: q in {0.90, 0.95, 0.99, 0.995, 0.999}).
  hypa    : learned HQS weights alpha_k of the data-step models at the test noise levels.
  alpha   : test-time scaling of the learned alpha (x1, x10, x100) for the data-step models
            (mechanism check: does stronger data-step regularisation remove the defocus failure?).
  runtime : wall-clock time of network, Wiener and gate scores on 256x256 and 512x512 inputs.
"""

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from paths import GATE_OUT, DIV2K_DIR  # noqa: E402
import argparse
import csv
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from degrade_torch import degrade_pair, eval_kernel_sets, random_gaussian_kernel, wiener_deconv
from eval_2x2 import EVAL_SIGMAS
from eval_shift import load, predict
from ext_kernels import shift_kernel_sets
from gate3 import CALIB_SIGMAS, rho_fn
from gate4 import nu_fn
from train_2x2 import build_cache, center_crops, psnr

UNROLLED = [f"runs_cond/{a}_s{s}" for a in ("unrolled", "unrolled_emb", "unrolled_both", "unrolled_blind")
            for s in (0, 1, 2)]
NAF = ["runs/mix_none_s0", "runs_cond/kernel_s0", "runs_cond/params_s0", "runs_cond/wiener_s0"]
SL = (..., slice(10, -10), slice(10, -10))


def calib(runs, dev, device, out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(777)
    ckern = [random_gaussian_kernel(rng) for _ in range(12)]
    for run in runs:
        m, t = load(Path(run), device)
        rows = []
        with torch.no_grad():
            for sigma in CALIB_SIGMAS:
                for kid, kern in enumerate(ckern):
                    for i in range(0, len(dev), 10):
                        x = dev[i:i + 10].to(device)
                        n = len(x)
                        k = torch.from_numpy(kern).float().to(device).expand(n, -1, -1)
                        s = torch.full((n,), float(sigma), device=device)
                        gen = torch.Generator(device=device).manual_seed(
                            zlib.crc32(f"calib|{kid}|{sigma}|{i}".encode()))
                        for y in degrade_pair(x, k, s, t["mode"], t["floor"], gen):
                            est = predict(m, t, y, k, s)
                            xw = wiener_deconv(y, k, s, t["floor"])
                            r = rho_fn(est, y, k, sigma, t["floor"], SL).tolist()
                            v = nu_fn(est, xw, k, sigma, t["floor"], SL).tolist()
                            rows += [(sigma, a, b) for a, b in zip(r, v)]
        with open(out / f"{Path(run).name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sigma", "rho", "nu"])
            w.writerows(rows)
        print(f"calib {run} done", flush=True)


def hypa(device, out):
    rows = []
    for run in [r for r in UNROLLED if "/unrolled_s" in r or "unrolled_both" in r]:
        m, t = load(Path(run), device)
        for sigma in EVAL_SIGMAS:
            s = torch.tensor([[((sigma ** 2 + m.floor ** 2) ** 0.5) / 255]], device=device)
            hyp = m.hypa(s * 10)
            alpha = (hyp[:, :m.stages] * s ** 2 * 100 + 1e-6)[0].tolist()
            rows.append(dict(run=Path(run).name, sigma=sigma, **{f"alpha{i + 1}": a for i, a in enumerate(alpha)},
                             wiener_lam=300 * float(s) ** 2))
    with open(out / "hypa_alpha.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("hypa done", flush=True)


def alpha_scaling(images, device, out, scales=(1, 10, 100)):
    ksets = dict(eval_kernel_sets(), **shift_kernel_sets())
    rows = []
    for run in [r for r in UNROLLED if "/unrolled_s" in r]:
        m, t = load(Path(run), device)
        state = {"scale": 1.0}

        def hook(_mod, _inp, outp):
            o = outp.clone()
            o[:, :m.stages] *= state["scale"]
            return o
        h = m.hypa.register_forward_hook(hook)
        with torch.no_grad():
            for scale in scales:
                state["scale"] = float(scale)
                for kset, kernels in ksets.items():
                    for kid, kern in enumerate(kernels):
                        for sigma in EVAL_SIGMAS:
                            for i in range(0, len(images), 10):
                                x = images[i:i + 10].to(device)
                                n = len(x)
                                k = torch.from_numpy(kern).float().to(device).expand(n, -1, -1)
                                s = torch.full((n,), float(sigma), device=device)
                                gen = torch.Generator(device=device).manual_seed(
                                    zlib.crc32(f"{kset}|{kid}|{sigma}|{i}".encode()))
                                y = degrade_pair(x, k, s, t["mode"], t["floor"], gen)[0]  # bn order
                                est = predict(m, t, y, k, s)
                                for v in psnr(est[SL].clamp(0, 1), x[SL]).tolist():
                                    rows.append((Path(run).name, scale, kset, kid, sigma, v))
                print(f"alpha {run} x{scale} done", flush=True)
        h.remove()
    with open(out / "alpha_scaling.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "scale", "kset", "kid", "sigma", "psnr"])
        w.writerows(rows)


def runtime(device, out, reps=50):
    rows = []
    k = torch.from_numpy(random_gaussian_kernel(np.random.default_rng(0))).float().to(device)[None]
    for size in (256, 512):
        y = torch.rand(1, 3, size, size, device=device)
        s = torch.tensor([5.0], device=device)
        sl = (..., slice(10, -10), slice(10, -10))
        for run in ("runs/mix_none_s0", "runs_cond/unrolled_s0", "runs_cond/unrolled_both_s0"):
            m, t = load(Path(run), device)
            fns = {"network": lambda: predict(m, t, y, k, s)}
            if run.endswith("unrolled_s0"):
                xw = wiener_deconv(y, k, s, 0.29)
                fns["wiener"] = lambda: wiener_deconv(y, k, s, 0.29)
                fns["gate scores (rho+nu)"] = lambda: (rho_fn(y, y, k, 5.0, 0.29, sl), nu_fn(y, xw, k, 5.0, 0.29, sl))
            for name, fn in fns.items():
                with torch.no_grad():
                    for _ in range(5):
                        fn()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        fn()
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / reps * 1000
                params = sum(p.numel() for p in m.parameters()) if name == "network" else 0
                rows.append((size, Path(run).name if name == "network" else name, params, f"{ms:.2f}"))
    with open(out / "runtime.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["size", "what", "params", "ms"])
        w.writerows(rows)
    print("runtime done", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", default=["hypa", "runtime", "calib", "alpha"])
    p.add_argument("--train-dir", type=Path, default=(DIV2K_DIR / "DIV2K_train_HR"))
    p.add_argument("--test-dir", type=Path, default=(DIV2K_DIR / "DIV2K_valid_HR"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    args = p.parse_args()
    device = torch.device("cuda")
    args.out.mkdir(exist_ok=True)
    if "hypa" in args.only:
        hypa(device, args.out)
    if "runtime" in args.only:
        runtime(device, args.out)
    if "calib" in args.only:
        dev = center_crops(build_cache(args.train_dir, Path("cache/train"))[-50:], 256)
        calib(UNROLLED + NAF, dev, device, (GATE_OUT / "calib"))
    if "alpha" in args.only:
        images = center_crops(build_cache(args.test_dir, Path("cache/valid")), 256)
        alpha_scaling(images, device, args.out)


if __name__ == "__main__":
    main()
