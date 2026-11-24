"""Summarise gate4 (two-sided gate) outputs into reports/gate4_*.

Per model x kernel set: mean PSNR of the network, of the rho-only gate (gate3 rule) and the
two-sided gate (rho AND nu) with Wiener or blind-NAFNet fallback, accept rates, and the paired
difference of the two-sided gate (Wiener fallback) vs blind NAFNet with a 95 % image bootstrap CI.
"""

from paths import GATE_OUT  # noqa: E402
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_gate3 import ci, md


def add_gates(d):
    d["acc_r"] = d.rho <= d.tau
    d["acc"] = d.acc_r & (d.nu <= d.tau_nu)
    d["g_rho_w"] = np.where(d.acc_r, d.psnr_net, d.psnr_wiener)
    d["g2_w"] = np.where(d.acc, d.psnr_net, d.psnr_wiener)
    d["g2_b"] = np.where(d.acc, d.psnr_net, d.psnr_blind)
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(GATE_OUT / "gate4"))
    p.add_argument("--order", default="both", choices=("bn", "nb", "both"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    args = p.parse_args()
    rng = np.random.default_rng(0)
    d = add_gates(pd.concat([pd.read_csv(f, dtype={"image": str}) for f in glob.glob(f"{args.root}/*/gate4.csv")]))
    if args.order != "both":
        d = d[d.order == args.order]
    cols = ["psnr_net", "g_rho_w", "g2_w", "g2_b", "psnr_blind", "psnr_wiener"]
    rows = []
    for (run, kset), g in d.groupby(["run", "kset"]):
        im = g.groupby("image")[cols].mean()
        diff = (im.g2_w - im.psnr_blind).to_numpy()
        lo, hi = ci(diff, rng)
        rows.append(dict(run=run, kset=kset, acc_rho=g.acc_r.mean(), acc_both=g.acc.mean(),
                         **{c: im[c].mean() for c in cols},
                         d_net=(im.psnr_net - im.psnr_blind).mean(),
                         d_gate2_w=f"{diff.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]"))
    summ = pd.DataFrame(rows)
    sig = d.groupby(["run", "kset", "sigma"])[["acc_r", "acc"] + cols].mean().reset_index()
    args.out.mkdir(exist_ok=True)
    summ.to_csv(args.out / f"gate4_summary_{args.order}.csv", index=False)
    sig.to_csv(args.out / f"gate4_sigma_{args.order}.csv", index=False)
    lines = [f"# Two-sided gate summary (order: {args.order})", "",
             "g_rho_w = rho-only gate, Wiener fallback; g2_w / g2_b = rho AND nu gate, Wiener / blind fallback.",
             "d_gate2_w = paired diff of g2_w vs blind NAFNet [95 % CI].", "", md(summ), ""]
    r = add_gates(pd.concat([pd.read_csv(f) for f in glob.glob(f"{args.root}/*/gate4_real.csv")]))
    rs = r.groupby("run").agg(acc_rho=("acc_r", "mean"), acc_both=("acc", "mean"), net=("psnr_net", "mean"),
                              g_rho_w=("g_rho_w", "mean"), g2_w=("g2_w", "mean"), g2_b=("g2_b", "mean"),
                              blind=("psnr_blind", "mean"), wiener=("psnr_wiener", "mean")).reset_index()
    rs.to_csv(args.out / "gate4_real.csv", index=False)
    lines += ["## Real Levin captures (28 pairs)", "", md(rs), ""]
    (args.out / f"gate4_{args.order}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
