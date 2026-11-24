"""Paired analysis of the conditioning-shortcut pilot.

Baselines from runs/ (mix_none = blind, mix_oracle = order token without dropout),
new runs from runs_cond/. Every model is compared with the blind model on the same
images, kernels, sigmas, orders and noise draws; seeds are matched by number and
averaged per image before the image-level bootstrap (see analyze_2x2.compare).

Pre-registered go/no-go (seed 0 pilot, then 3 seeds):
  HARM   : params(true) - blind on motion <= -0.20 dB with CI high < 0
  REMEDY : params_do50(true) - blind on motion >= 0 and on gauss >= +0.05 dB
  CONTROL (full kernel, added after the seed-0 pilot): kernel(true) - blind on motion
    > -0.20 dB  -> harm was descriptor misspecification;
    <= -0.20 dB with CI high < 0 -> harm persists with lossless conditioning (extrapolation)
GO needs HARM; the method story additionally needs REMEDY.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_2x2 import KEY, compare


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, default=Path("runs"))
    p.add_argument("--cond-runs", type=Path, default=Path("runs_cond"))
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--out", type=Path, default=Path("runs_cond/analysis_cond.csv"))
    args = p.parse_args()
    frames = [pd.read_csv(f) for f in sorted(args.base.glob("mix_*/eval.csv"))]
    frames += [pd.read_csv(f) for f in sorted(args.cond_runs.glob("*/eval.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["token_dropout"] = df.get("token_dropout", 0.0)
    df["token_dropout"] = df["token_dropout"].fillna(0.0)
    df["family"] = df["run"].str.replace(r"_s\d+$", "", regex=True)
    df["model"] = np.where(df["family"] == "mix_none", "blind", df["family"] + ":" + df["token"])
    tables = {name: sub.pivot_table(index=KEY, columns="seed", values="psnr")
              for name, sub in df.groupby("model")}
    print("models:", {k: list(v.columns) for k, v in tables.items()})
    rng = np.random.default_rng(0)
    rows = []
    for name in sorted(tables):
        if name == "blind":
            continue
        for kset in ("gauss", "motion"):
            for order in ("both", "bn", "nb"):
                r = compare(tables, name, "blind", kset, order, args.boot, rng)
                if r:
                    rows.append(r)
    res = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    both = res[res.order == "both"]
    with pd.option_context("display.width", 200, "display.float_format", "{:+.3f}".format):
        print(both.pivot_table(index="A", columns="kset", values=["diff_db", "ci_low", "ci_high"]).to_string())

    def get(a, kset):
        r = both[(both.A == a) & (both.kset == kset)]
        return r.iloc[0] if len(r) else None

    harm = get("params:true", "motion")
    if harm is not None:
        ok = harm.diff_db <= -0.20 and harm.ci_high < 0
        print(f"\nHARM   params(true) - blind, motion: {harm.diff_db:+.3f} dB "
              f"[{harm.ci_low:+.3f}, {harm.ci_high:+.3f}] -> {'MET' if ok else 'NOT MET'}")
    rem_m, rem_g = get("params_do50:true", "motion"), get("params_do50:true", "gauss")
    if rem_m is not None and rem_g is not None:
        ok = rem_m.diff_db >= 0 and rem_g.diff_db >= 0.05
        print(f"REMEDY params_do50(true) - blind: motion {rem_m.diff_db:+.3f}, gauss {rem_g.diff_db:+.3f} dB "
              f"-> {'MET' if ok else 'NOT MET'}")
    ctl = get("kernel:true", "motion")
    if ctl is not None:
        if ctl.diff_db > -0.20:
            verdict = "harm gone -> params harm was descriptor misspecification"
        elif ctl.ci_high < 0:
            verdict = "harm persists with lossless descriptor -> conditioning fails to extrapolate"
        else:
            verdict = "inconclusive"
        print(f"CONTROL kernel(true) - blind, motion: {ctl.diff_db:+.3f} dB "
              f"[{ctl.ci_low:+.3f}, {ctl.ci_high:+.3f}] -> {verdict}")
    phy_m, phy_g = get("wiener:true", "motion"), get("wiener:true", "gauss")
    if phy_m is not None and phy_g is not None:
        # physics-based conditioning extrapolates if it keeps the ID gain without the OOD loss
        ok = phy_m.diff_db > -0.20 and phy_g.diff_db >= 0.05
        print(f"PHYSICS wiener(true) - blind: motion {phy_m.diff_db:+.3f} [{phy_m.ci_low:+.3f}, "
              f"{phy_m.ci_high:+.3f}], gauss {phy_g.diff_db:+.3f} dB -> "
              f"{'EXTRAPOLATES' if ok else 'does not extrapolate'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
