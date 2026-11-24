"""Locked paired evaluation of one trained run on DIV2K_valid_HR (0801-0900).

For every image, kernel and sigma, the two orders share the same noise draws.
Blind runs are evaluated with the UNKNOWN token. Oracle runs are evaluated with
the true order token and, as a sensitivity check, with the wrong token.
Writes one row per (image, kernel set, kernel, sigma, order, token).
"""

import argparse
import csv
import zlib
from pathlib import Path

import numpy as np
import torch

from degrade_torch import DESCRIPTORS, IN_CH, degrade_pair, eval_kernel_sets, model_input
from nafnet import UNKNOWN, NAFNet
from unrolled import build_model
from train_2x2 import build_cache, center_crops, psnr

EVAL_SIGMAS = (2, 5, 15, 25)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="training output directory with last.pt")
    p.add_argument("--test-dir", type=Path, required=True, help="DIV2K_valid_HR")
    p.add_argument("--cache", type=Path, default=Path("cache/valid"))
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--border", type=int, default=10, help="pixels excluded from PSNR at each edge")
    p.add_argument("--batch", type=int, default=10,
                   help="keep identical for all runs: noise seeds are tied to batch position")
    p.add_argument("--out", type=Path, default=None, help="default: <run>/eval.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.run / "last.pt", map_location=device)
    targs = ckpt["args"]
    cond = targs["cond"]
    model = build_model(cond, targs["width"], DESCRIPTORS[cond][0] if cond in DESCRIPTORS else 0,
                        IN_CH.get(cond, 3), targs["floor"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    files = build_cache(args.test_dir, args.cache)
    images = center_crops(files, args.crop)
    ksets = eval_kernel_sets()
    np.savez(args.run / "eval_kernels.npz", **ksets)
    dropout = targs.get("token_dropout", 0.0)
    # "wrong": order flipped (oracle) or sigma doubled (params); "unknown": signal hidden
    tokens = {"none": ("unknown",), "oracle": ("true", "wrong"), "params": ("true", "wrong"),
              "kernel": ("true", "wrong"), "wiener": ("true", "wrong"), "unrolled": ("true", "wrong"),
              "unrolled_emb": ("true", "wrong"), "unrolled_both": ("true", "wrong"),
              "unrolled_blind": ("true",)}[cond]
    if dropout > 0:
        tokens += ("unknown",)
    b = args.border
    out = args.out or args.run / "eval.csv"
    tmp = out.with_suffix(".partial")  # renamed only after a complete pass
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "train_order", "cond", "token_dropout", "seed", "mode", "floor", "token", "image",
                    "kset", "kid", "sigma", "order", "psnr"])
        with torch.no_grad():
            for kset, kernels in ksets.items():
                for kid, kern in enumerate(kernels):
                    for sigma in EVAL_SIGMAS:
                        for i in range(0, len(images), args.batch):
                            x = images[i:i + args.batch].to(device)
                            n = len(x)
                            k = torch.from_numpy(kern).float().to(device).expand(n, -1, -1)
                            seed = zlib.crc32(f"{kset}|{kid}|{sigma}|{i}".encode())
                            gen = torch.Generator(device=device).manual_seed(seed)
                            pair = degrade_pair(x, k, torch.full((n,), float(sigma), device=device),
                                                targs["mode"], targs["floor"], gen)
                            for order, y in enumerate(pair):
                                for tok_name in tokens:
                                    params, inp = None, y
                                    if cond == "wiener":
                                        tok = UNKNOWN
                                        s = torch.full((n,), float(sigma), device=device)
                                        inp = model_input(cond, y, k, 2 * s if tok_name == "wrong" else s,
                                                          targs["floor"])
                                    elif cond in DESCRIPTORS:
                                        tok = UNKNOWN
                                        s = torch.full((n,), float(sigma), device=device)
                                        params = DESCRIPTORS[cond][1](
                                            k, 2 * s if tok_name == "wrong" else s,
                                            torch.full((n,), tok_name != "unknown", device=device))
                                    else:
                                        tok = {"true": order, "wrong": 1 - order, "unknown": UNKNOWN}[tok_name]
                                    pred = model(inp, torch.full((n,), tok, device=device, dtype=torch.long), params)
                                    sl = (..., slice(b, -b or None), slice(b, -b or None))
                                    scores = psnr(pred[sl].float(), x[sl]).tolist()
                                    for j, s in enumerate(scores):
                                        w.writerow([args.run.name, targs["train_order"], targs["cond"], dropout,
                                                    targs["seed"], targs["mode"], targs["floor"], tok_name,
                                                    files[i + j].stem, kset, kid, sigma,
                                                    "bn" if order == 0 else "nb", f"{s:.4f}"])
                print(f"{args.run.name}: {kset} done", flush=True)
    tmp.replace(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
