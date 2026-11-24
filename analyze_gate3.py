"""Summarise gate3 outputs into reports/gate3_*.csv and reports/gate3.md.

Per model x kernel set (and per sigma): mean PSNR of the network, of the gate with Wiener fallback
and with blind-NAFNet fallback, the accept rate, and paired differences vs the blind NAFNet (same
images, kernels, noise) with 95 % image-bootstrap CIs. --order bn|nb|both selects the degradation
order (bn = standard y = k*x + n). Real Levin captures summarised separately.
"""

from paths import GATE_OUT  # noqa: E402
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def ci(d, rng, n=2000):
    b = rng.choice(d, size=(n, len(d))).mean(1)
    return np.percentile(b, [2.5, 97.5])


def md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--order", default="both", choices=("bn", "nb", "both"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--extra", default=str(GATE_OUT / "gate3"), help="additional root with <run>/gate3*.csv")
    args = p.parse_args()
    rng = np.random.default_rng(0)
    d = pd.concat([pd.read_csv(f, dtype={"image": str}) for f in sorted(set(glob.glob("runs*/*/gate3.csv")) | set(glob.glob(f"{args.extra}/*/gate3.csv")))])
    if args.order != "both":
        d = d[d.order == args.order]
    d["accept"] = d.rho <= d.tau
    d["gate_wiener"] = np.where(d.accept, d.psnr_net, d.psnr_wiener)
    d["gate_blind"] = np.where(d.accept, d.psnr_net, d.psnr_blind)
    rows = []
    for (run, kset), g in d.groupby(["run", "kset"]):
        im = g.groupby("image")[["psnr_net", "gate_wiener", "gate_blind", "psnr_blind", "psnr_wiener"]].mean()
        row = dict(run=run, kset=kset, accept=g.accept.mean(), net=im.psnr_net.mean(),
                   gate_w=im.gate_wiener.mean(), gate_b=im.gate_blind.mean(), blind=im.psnr_blind.mean(),
                   wiener=im.psnr_wiener.mean())
        for c in ("psnr_net", "gate_wiener"):
            diff = (im[c] - im.psnr_blind).to_numpy()
            lo, hi = ci(diff, rng)
            row[f"d_{c}"] = f"{diff.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]"
        rows.append(row)
    summ = pd.DataFrame(rows)
    sig = d.groupby(["run", "kset", "sigma"]).agg(accept=("accept", "mean"), net=("psnr_net", "mean"),
                                                   gate_w=("gate_wiener", "mean"), blind=("psnr_blind", "mean"),
                                                   wiener=("psnr_wiener", "mean")).reset_index()
    args.out.mkdir(exist_ok=True)
    summ.to_csv(args.out / f"gate3_summary_{args.order}.csv", index=False)
    sig.to_csv(args.out / f"gate3_sigma_{args.order}.csv", index=False)
    lines = [f"# Gate3 summary (order: {args.order})", "",
             "net/gate_w/gate_b/blind/wiener = mean PSNR; d_* = paired diff vs blind NAFNet [95 % CI]", "",
             md(summ), ""]
    rf = glob.glob("runs*/*/gate3_real.csv") + glob.glob(f"{args.extra}/*/gate3_real.csv")
    if rf:
        r = pd.concat([pd.read_csv(f) for f in rf])
        r["accept"] = r.rho <= r.tau
        r["gate_wiener"] = np.where(r.accept, r.psnr_net, r.psnr_wiener)
        rs = r.groupby("run").agg(accept=("accept", "mean"), net=("psnr_net", "mean"),
                                  gate_w=("gate_wiener", "mean"), blind=("psnr_blind", "mean"),
                                  wiener=("psnr_wiener", "mean")).reset_index()
        rs.to_csv(args.out / "gate3_real.csv", index=False)
        lines += ["## Real Levin captures (28 pairs)", "", md(rs), ""]
    (args.out / f"gate3_{args.order}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
