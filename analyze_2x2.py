"""Paired analysis of the 2x2 order-mismatch study.

Reads every <runs>/*/eval.csv and reports, for each comparison A - B:
  mean paired PSNR difference, image-level 95% bootstrap CI (all kernels, sigmas
  and orders of one image form one block; seeds are averaged per image first),
  and the mean difference of each seed.

Primary pre-registered test (printed last):
  oracle (mix + true order) - blind (mix), motion kernels, both orders:
  diff >= 0.10 dB, CI lower bound > 0, every seed > 0.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["image", "kset", "kid", "sigma", "order"]


def model_table(df):
    """One PSNR column per (model, seed); rows indexed by KEY."""
    pick = {
        "blind": (df.train_order == "mix") & (df.cond == "none"),
        "oracle": (df.train_order == "mix") & (df.cond == "oracle") & (df.token == "true"),
        "oracle_wrong_token": (df.train_order == "mix") & (df.cond == "oracle") & (df.token == "wrong"),
        "expert_bn": df.train_order == "bn",
        "expert_nb": df.train_order == "nb",
    }
    tables = {}
    for name, mask in pick.items():
        sub = df[mask]
        if len(sub):
            tables[name] = sub.pivot_table(index=KEY, columns="seed", values="psnr")
    if "expert_bn" in tables and "expert_nb" in tables:
        bn, nb = tables["expert_bn"], tables["expert_nb"]
        is_bn = bn.index.get_level_values("order") == "bn"
        tables["expert_routed"] = pd.concat([bn[is_bn], nb[~is_bn]]).sort_index()
        tables["expert_mismatched"] = pd.concat([nb[is_bn], bn[~is_bn]]).sort_index()
    return tables


def compare(tables, a, b, kset, order, n_boot, rng):
    if a not in tables or b not in tables:
        return None
    ta, tb = tables[a], tables[b]
    seeds = sorted(set(ta.columns) & set(tb.columns))
    idx = ta.index.intersection(tb.index)
    sel = idx[(idx.get_level_values("kset") == kset)
              & ((idx.get_level_values("order") == order) if order != "both" else True)]
    diff = ta.loc[sel, seeds] - tb.loc[sel, seeds]
    per_image_seed = diff.groupby(level="image").mean()          # images x seeds
    per_image = per_image_seed.mean(axis=1).to_numpy()
    boots = rng.choice(per_image, size=(n_boot, len(per_image))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(A=a, B=b, kset=kset, order=order, n_images=len(per_image), n_seeds=len(seeds),
                diff_db=per_image.mean(), ci_low=lo, ci_high=hi,
                per_seed=" ".join(f"{v:+.3f}" for v in per_image_seed.mean().to_numpy()),
                min_seed=per_image_seed.mean().min())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--out", type=Path, default=Path("runs/analysis_2x2.csv"))
    args = p.parse_args()
    files = sorted(args.runs.glob("*/eval.csv"))
    if not files:
        raise SystemExit(f"no eval.csv under {args.runs}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for col in ("mode", "floor"):
        if df[col].nunique() > 1:
            raise SystemExit(f"runs mix different {col} settings: {df[col].unique()}")
    counts = df.groupby(["run", "token"]).size()
    if counts.nunique() > 1:
        raise SystemExit("incomplete or inconsistent eval.csv files (rows per run/token):\n"
                         + counts.to_string())
    tables = model_table(df)
    for name, t in tables.items():
        if t.isna().any().any():
            raise SystemExit(f"missing evaluations for {name}: re-run eval_2x2.py for those runs")
    print("models found:", {k: list(v.columns) for k, v in tables.items()})

    comparisons = [
        ("oracle", "blind", "Does knowing the true order help a model trained on both orders?"),
        ("expert_routed", "blind", "Order-specific experts routed by the true order vs blind"),
        ("expert_routed", "expert_mismatched", "Order mismatch gap: right-order vs wrong-order expert"),
        ("blind", "expert_mismatched", "Training on both orders vs a single wrong-order model"),
        ("oracle_wrong_token", "oracle", "Penalty when the oracle model receives the wrong order"),
    ]
    rng = np.random.default_rng(0)
    rows = []
    for a, b, question in comparisons:
        for kset in ("gauss", "motion"):
            for order in ("both", "bn", "nb"):
                r = compare(tables, a, b, kset, order, args.boot, rng)
                if r:
                    r["question"] = question
                    rows.append(r)
    res = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(res.drop(columns="question").to_string(index=False))

    prim = res[(res.A == "oracle") & (res.B == "blind") & (res.kset == "motion") & (res.order == "both")]
    if len(prim):
        r = prim.iloc[0]
        ok = r.diff_db >= args.threshold and r.ci_low > 0 and r.min_seed > 0 and r.n_seeds >= 3
        print(f"\nPRIMARY (motion, both orders, oracle - blind): {r.diff_db:+.3f} dB "
              f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}], seeds {r.per_seed} -> "
              f"{'MET' if ok else 'NOT MET'} (threshold {args.threshold} dB, {r.n_seeds} seeds)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
