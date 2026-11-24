"""LaTeX tables for the paper, generated from the evaluation outputs (never edit the .tex by hand).

Writes paper/tables/{factorial,gate,accept,other_nets,ablation,alpha,runtime}.tex and
reports/paper_numbers.md (every number quoted in the text, for cross-checking).
"""

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from paths import GATE_OUT  # noqa: E402
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"
GATE = (GATE_OUT / "gate4")
CALIB = (GATE_OUT / "calib")
FAMS = ["gauss", "motion", "levin", "defocus"]
FAM_TEX = {"gauss": "Gauss (ID)", "motion": "Motion", "levin": "Levin", "defocus": "Defocus", "real": "Real"}
ARMS = [("unrolled_blind", "No kernel"), ("unrolled_emb", "Embedding"), ("unrolled", "Data step"),
        ("unrolled_both", "Data step + emb.")]
SEEDS = (0, 1, 2)
CALIB_SIGMAS = (1, 2, 3, 5, 10, 15, 25)
NOTES = []


def note(s):
    NOTES.append(s)


def ms(v, d=2):
    v = np.asarray(v, float)
    if len(v) == 1:
        return f"{v[0]:.{d}f}"
    return f"{v.mean():.{d}f}{{\\scriptsize$\\pm${v.std(ddof=0):.{d}f}}}"


def sd(v, d=2):
    v = np.asarray(v, float)
    return f"{v.mean():+.{d}f}"


def run_dir(run):
    for r in ("runs_cond", "runs"):
        if (ROOT / r / run).exists():
            return ROOT / r / run
    raise FileNotFoundError(run)


def synth(run):
    d = run_dir(run)
    df = pd.concat([pd.read_csv(d / f, dtype={"image": str, "token": str})
                    for f in ("eval.csv", "eval_shift.csv")])
    df = df[df.token.isin(["true", "unknown"])]
    return df.groupby("kset").psnr.mean()


def real(run):
    return pd.read_csv(run_dir(run) / "eval_levin_real.csv").psnr.mean()


def load_gate():
    g = pd.concat([pd.read_csv(f, dtype={"image": str}) for f in glob.glob(str(GATE / "*" / "gate4.csv"))])
    r = pd.concat([pd.read_csv(f) for f in glob.glob(str(GATE / "*" / "gate4_real.csv"))])
    for d in (g, r):
        d["acc_r"] = d.rho <= d.tau
        d["acc"] = d.acc_r & (d.nu <= d.tau_nu)
        d["g_rho"] = np.where(d.acc_r, d.psnr_net, d.psnr_wiener)
        d["g2_w"] = np.where(d.acc, d.psnr_net, d.psnr_wiener)
        d["g2_b"] = np.where(d.acc, d.psnr_net, d.psnr_blind)
    return g, r


def table(name, body, cols, caption, label, wide=False, size="\\footnotesize"):
    env = "table*" if wide else "table"
    tex = [f"\\begin{{{env}}}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
           "\\setlength{\\tabcolsep}{3.5pt}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule", *body,
           "\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    (OUT / f"{name}.tex").write_text("\n".join(tex), encoding="utf-8")


def factorial(g):
    rows = ["Model & " + " & ".join(FAM_TEX[f] for f in FAMS + ["real"]) + " \\\\", "\\midrule"]
    w = g[g.run == "unrolled_s0"].groupby("kset").psnr_wiener.mean()
    wr = pd.read_csv(ROOT / "runs_cond/wiener_only/eval_levin_real.csv").psnr.mean()
    rows.append("Wiener (no learning) & " + " & ".join(f"{w[f]:.2f}" for f in FAMS) + f" & {wr:.2f} \\\\")
    blind = {s: synth(f"mix_none_s{s}") for s in SEEDS}
    blind_r = {s: real(f"mix_none_s{s}") for s in SEEDS}
    rows.append("NAFNet, blind (3.0\\,M) & " + " & ".join(ms([blind[s][f] for s in SEEDS]) for f in FAMS)
                + " & " + ms([blind_r[s] for s in SEEDS]) + " \\\\")
    note(f"Wiener synthetic: {dict(w.round(2))}, real {wr:.2f}")
    note("NAFNet blind: " + str({f: round(np.mean([blind[s][f] for s in SEEDS]), 2) for f in FAMS})
         + f", real {np.mean(list(blind_r.values())):.2f}")
    rows += ["\\midrule", "\\multicolumn{6}{l}{\\emph{Unrolled HQS (0.46\\,M), kernel route:}} \\\\"]
    for arm, label in ARMS:
        p = {s: synth(f"{arm}_s{s}") for s in SEEDS}
        pr = [real(f"{arm}_s{s}") for s in SEEDS]
        cells = []
        for f in FAMS:
            dlt = [p[s][f] - blind[s][f] for s in SEEDS]
            cells.append(f"{ms([p[s][f] for s in SEEDS])} ({sd(dlt)})")
            note(f"{arm} {f}: psnr {np.mean([p[s][f] for s in SEEDS]):.2f}, delta vs blind {np.mean(dlt):+.2f}"
                 f" +- {np.std(dlt):.2f} (per seed {', '.join(f'{x:+.2f}' for x in dlt)})")
        dr = [pr[i] - blind_r[s] for i, s in enumerate(SEEDS)]
        cells.append(f"{ms(pr)} ({sd(dr)})")
        note(f"{arm} real: {np.mean(pr):.2f} (delta {np.mean(dr):+.2f})")
        rows.append(f"\\quad {label} & " + " & ".join(cells) + " \\\\")
    table("factorial", rows, "l" + "c" * 5,
          "Controlled factorial study: PSNR (dB) of the networks alone, mean$\\pm$std over three seeds; in "
          "parentheses the mean paired difference to the blind NAFNet of the same seed. Gauss is the training "
          "distribution; motion, Levin and defocus are unseen kernel families; the Real column contains the 28 real blurred "
          "images of Levin~\\emph{et al.}~\\cite{levin2009understanding}.", "tab:factorial", wide=True)


def gate_table(g, r):
    blind = g[g.run == "mix_none_s0"].groupby("kset").psnr_blind.mean()
    wien = g[g.run == "unrolled_s0"].groupby("kset").psnr_wiener.mean()
    rb, rw = r.psnr_blind.mean() / 1, r[r.run == "unrolled_s0"].psnr_wiener.mean()
    rows = ["Model & Decision & " + " & ".join(FAM_TEX[f] for f in FAMS + ["real"]) + " \\\\", "\\midrule",
            "Wiener & -- & " + " & ".join(f"{wien[f]:.2f}" for f in FAMS) + f" & {rw:.2f} \\\\",
            "NAFNet blind (s0) & -- & " + " & ".join(f"{blind[f]:.2f}" for f in FAMS)
            + f" & {r[r.run == 'mix_none_s0'].psnr_net.mean():.2f} \\\\"]
    for arm, label in ARMS:
        rows.append("\\midrule")
        for key, dec in (("psnr_net", "network"), ("g_rho", "$\\rho$-gate"), ("g2_w", "two-sided gate")):
            cells = []
            for f in FAMS:
                v = [g[(g.run == f"{arm}_s{s}") & (g.kset == f)][key].mean() for s in SEEDS]
                cells.append(ms(v))
                if key == "g2_w":
                    note(f"GATE {arm} {f}: two-sided {np.mean(v):.2f} (vs blind {np.mean(v) - blind[f]:+.2f}, "
                         f"vs Wiener {np.mean(v) - wien[f]:+.2f})")
            v = [r[r.run == f"{arm}_s{s}"][key if key != "psnr_net" else "psnr_net"].mean() for s in SEEDS]
            cells.append(ms(v))
            if key == "g2_w":
                note(f"GATE {arm} real: two-sided {np.mean(v):.2f}")
            name = label if key == "psnr_net" else ""
            rows.append(f"{name} & {dec} & " + " & ".join(cells) + " \\\\")
    table("gate", rows, "llccccc",
          "Effect of the gate (PSNR in dB, mean$\\pm$std over three seeds, both formation orders). "
          "``$\\rho$-gate'' rejects only on the re-blurring residual; the two-sided gate rejects if either "
          "$\\rho$ or $\\nu$ exceeds its in-distribution 99\\,\\% quantile. Rejected outputs are replaced by the "
          "Wiener estimate.", "tab:gate", wide=True)


def accept(g, r):
    rows = ["Model & " + " & ".join(FAM_TEX[f] for f in FAMS + ["real"]) + " \\\\", "\\midrule"]
    for arm, label in ARMS:
        cells = []
        for f in FAMS:
            s = g[g.run.str.match(rf"{arm}_s\d$") & (g.kset == f)]
            cells.append(f"{100 * s.acc_r.mean():.0f} / {100 * s.acc.mean():.0f}")
            note(f"ACCEPT {arm} {f}: rho {100 * s.acc_r.mean():.1f} %, both {100 * s.acc.mean():.1f} %")
        s = r[r.run.str.match(rf"{arm}_s\d$")]
        cells.append(f"{100 * s.acc_r.mean():.0f} / {100 * s.acc.mean():.0f}")
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    table("accept", rows, "lccccc",
          "Acceptance rate (\\%) of the $\\rho$-gate / two-sided gate, averaged over seeds, noise levels and "
          "both orders.", "tab:accept", size="\\scriptsize")


def other_nets(g, r):
    nets = [("mix_none_s0", "NAFNet blind"), ("params_s0", "NAFNet + moment FiLM"),
            ("kernel_s0", "NAFNet + kernel FiLM"), ("wiener_s0", "NAFNet + Wiener input")]
    rows = ["Model & " + " & ".join(FAM_TEX[f] for f in FAMS + ["real"]) + " \\\\", "\\midrule"]
    for run, label in nets:
        cells = []
        for f in FAMS:
            s = g[(g.run == run) & (g.kset == f)]
            cells.append(f"{s.psnr_net.mean():.2f}$\\to${s.g2_w.mean():.2f}")
        s = r[r.run == run]
        cells.append(f"{s.psnr_net.mean():.2f}$\\to${s.g2_w.mean():.2f}")
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        note(f"OTHER {run}: " + " ".join(c for c in cells))
    table("other_nets", rows, "lccccc",
          "The gate applied unchanged to NAFNet-based models with other conditioning routes (seed 0): "
          "network $\\to$ two-sided gate, PSNR in dB.", "tab:other", wide=True)


def ablation(g, r):
    rows = ["Variant & " + " & ".join(FAM_TEX[f] for f in FAMS + ["real"]) + " \\\\", "\\midrule"]
    d = g[g.run.str.match(r"unrolled_s\d$")]
    dr = r[r.run.str.match(r"unrolled_s\d$")]

    def line(label, key, sub=None, subr=None):
        sub = d if sub is None else sub
        subr = dr if subr is None else subr
        cells = [ms([sub[(sub.run == f"unrolled_s{s}") & (sub.kset == f)][key].mean() for s in SEEDS]) for f in FAMS]
        cells.append("--" if subr is None or len(subr) == 0 else
                     ms([subr[subr.run == f"unrolled_s{s}"][key].mean() for s in SEEDS]))
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        note(f"ABL {label}: " + " | ".join(cells))

    rows.append("\\multicolumn{6}{l}{\\emph{(A2) fallback}} \\\\")
    line("blind NAFNet", "g2_b")
    line("Wiener (ours)", "g2_w")
    # cascade: blind NAFNet if it passes its own two-sided gate, else Wiener
    key = ["image", "kset", "kid", "sigma", "order"]
    bl = g[g.run == "mix_none_s0"][key + ["acc"]].rename(columns={"acc": "acc_b"})
    dc = d.merge(bl, on=key, how="left")
    dc["casc"] = np.where(dc.acc, dc.psnr_net, np.where(dc.acc_b, dc.psnr_blind, dc.psnr_wiener))
    blr = r[r.run == "mix_none_s0"][["pair", "acc"]].rename(columns={"acc": "acc_b"})
    drc = dr.merge(blr, on="pair", how="left")
    drc["casc"] = np.where(drc.acc, drc.psnr_net, np.where(drc.acc_b, drc.psnr_blind, drc.psnr_wiener))
    line("gated blind, then Wiener", "casc", dc, drc)
    rows.append("\\multicolumn{6}{l}{\\emph{(A3) formation order}} \\\\")
    line("blur$\\to$noise", "g2_w", d[d.order == "bn"], dr.iloc[:0])
    line("noise$\\to$blur", "g2_w", d[d.order == "nb"], dr.iloc[:0])
    # (A4) quantile, post hoc from the dumped calibration scores
    if all((CALIB / f"unrolled_s{s}.csv").exists() for s in SEEDS):
        rows.append("\\multicolumn{6}{l}{\\emph{(A4) calibration quantile $q$}} \\\\")
        for q in (0.9, 0.95, 0.99, 0.999):
            parts, partsr = [], []
            for s in SEEDS:
                c = pd.read_csv(CALIB / f"unrolled_s{s}.csv")
                tr = c.groupby("sigma").rho.quantile(q)
                tn = c.groupby("sigma").nu.quantile(q)
                sub = d[d.run == f"unrolled_s{s}"]
                acc = (sub.rho <= sub.sigma.map(tr)) & (sub.nu <= sub.sigma.map(tn))
                parts.append(sub.assign(gq=np.where(acc, sub.psnr_net, sub.psnr_wiener)))
                subr = dr[dr.run == f"unrolled_s{s}"]
                ls = np.log(np.maximum(subr.sigma_est.clip(lower=0.5), CALIB_SIGMAS[0]))
                trr = np.interp(ls, np.log(tr.index), tr.values)
                tnr = np.interp(ls, np.log(tn.index), tn.values)
                accr = (subr.rho <= trr) & (subr.nu <= tnr)
                partsr.append(subr.assign(gq=np.where(accr, subr.psnr_net, subr.psnr_wiener)))
            line(f"$q={q}$" + (" (ours)" if q == 0.99 else ""), "gq", pd.concat(parts), pd.concat(partsr))
        parts, partsr = [], []
        for s in SEEDS:
            tr, tn, n_img = grouped_taus(calib_with_images(CALIB / f"unrolled_s{s}.csv"))
            sub = d[d.run == f"unrolled_s{s}"]
            acc = (sub.rho <= sub.sigma.map(tr)) & (sub.nu <= sub.sigma.map(tn))
            parts.append(sub.assign(gq=np.where(acc, sub.psnr_net, sub.psnr_wiener), accq=acc))
            subr = dr[dr.run == f"unrolled_s{s}"]
            ls = np.log(np.maximum(subr.sigma_est.clip(lower=0.5), CALIB_SIGMAS[0]))
            accr = (subr.rho <= np.interp(ls, np.log(tr.index), tr.values)) & \
                   (subr.nu <= np.interp(ls, np.log(tn.index), tn.values))
            partsr.append(subr.assign(gq=np.where(accr, subr.psnr_net, subr.psnr_wiener)))
        pg = pd.concat(parts)
        note(f"GROUPED calibration: n_images={n_img}, ID false-reject "
             f"{100 * (1 - pg[pg.kset == 'gauss'].accq.mean()):.2f} %, per-family accept "
             + str(pg.groupby("kset").accq.mean().round(3).to_dict()))
        line("image-grouped, $1/(n{+}1)$", "gq", pg, pd.concat(partsr))
    # (A5) Wiener weight of the nu score (gate4.py --nu-alpha; fallback unchanged), thresholds recalibrated
    lam_dirs = {lam: Path(f"{GATE_OUT}/gate4_lam{lam}") for lam in (100, 900)}
    if all((p / f"unrolled_s{s}" / "gate4_real.csv").exists() for p in lam_dirs.values() for s in SEEDS):
        rows.append(r"\multicolumn{6}{l}{\emph{(A5) weight $\lambda$ of the score $\nu$}} \\")
        for lam in (100, 900):  # lambda = 300 s^2 is the 'Wiener (ours)' row of (A2)
            gl = pd.concat([pd.read_csv(lam_dirs[lam] / f"unrolled_s{s}" / "gate4.csv", dtype={"image": str})
                            for s in SEEDS])
            rl = pd.concat([pd.read_csv(lam_dirs[lam] / f"unrolled_s{s}" / "gate4_real.csv") for s in SEEDS])
            for x in (gl, rl):
                x["g2_w"] = np.where((x.rho <= x.tau) & (x.nu <= x.tau_nu), x.psnr_net, x.psnr_wiener)
            line(rf"$\lambda={lam}\,s^2$", "g2_w", gl, rl)
    table("ablation", rows, "lccccc",
          "Ablations for the data-step model (PSNR in dB, mean$\\pm$std over three seeds).", "tab:ablation",
          wide=True)


def rule_table(g, r):
    """(A1) decision rule x conditioning route: which score catches which failure mode."""
    arms = [("unrolled_emb", "Embedding"), ("unrolled", "Data step")]
    fams = ["motion", "levin", "defocus"]
    head = " & ".join(f"\\multicolumn{{4}}{{c}}{{{lab}}}" for _, lab in arms)
    sub = " & ".join(" & ".join(FAM_TEX[f] for f in fams + ["real"]) for _ in arms)
    rows = [f"Rule & {head} \\\\", "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}", f" & {sub} \\\\", "\\midrule"]
    rules = [("network only", None), ("$\\rho$ only", "r"), ("$\\nu$ only", "n"), ("$\\rho$ and $\\nu$", "rn")]
    for label, rule in rules:
        cells = []
        for arm, _ in arms:
            for f in fams + ["real"]:
                d = (r if f == "real" else g[g.kset == f])
                d = d[d.run.str.match(rf"{arm}_s\d$")]
                acc = {None: np.ones(len(d), bool), "r": d.rho <= d.tau, "n": d.nu <= d.tau_nu,
                       "rn": (d.rho <= d.tau) & (d.nu <= d.tau_nu)}[rule]
                v = np.where(acc, d.psnr_net, d.psnr_wiener)
                cells.append(f"{pd.Series(v).groupby(d.run.values).mean().mean():.2f}")
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        note(f"RULE {label}: " + " | ".join(cells))
    table("rule", rows, "l" + "c" * 8,
          "Which consistency score catches which failure (PSNR in dB, mean over three seeds; rejected outputs "
          "are replaced by Wiener). The residual score $\\rho$ mitigates the embedding failure, the null-space score "
          "$\\nu$ mitigates the data-step failure; only their conjunction mitigates both.", "tab:rule", wide=True)


def calib_with_images(path):
    """Calibration dump rows are written in the order sigma > kernel > batch of 10 > order (bn, nb) > image."""
    c = pd.read_csv(path)
    w = c.groupby("sigma").cumcount()
    c["image"] = (w % 100) // 20 * 10 + w % 10
    return c


def grouped_taus(c):
    """Image-grouped split-conformal thresholds: per image the max score over its kernels and orders, threshold =
    max over the n held-out images, so P(any output of a new ID image fails one test) <= 1/(n+1)."""
    per = c.groupby(["sigma", "image"])[["rho", "nu"]].max()
    return per.groupby("sigma").rho.max(), per.groupby("sigma").nu.max(), per.groupby("sigma").size().iloc[0]


def decision_table(g, r):
    """Quality of the accept/reject decision against the oracle choice between network and Wiener."""
    arms = [("unrolled", "Data step"), ("unrolled_emb", "Embedding")]
    rows = ["Model & Metric & " + " & ".join(FAM_TEX[f] for f in FAMS + ["real"]) + " \\\\", "\\midrule"]
    for arm, label in arms:
        stats = {}
        for f in FAMS + ["real"]:
            d = (r if f == "real" else g[g.kset == f])
            d = d[d.run.str.match(rf"{arm}_s\d$")]
            gap = d.psnr_net - d.psnr_wiener
            stats[f] = dict(fa=100 * (d.acc & (gap < -0.5)).mean(), fr=100 * (~d.acc & (gap > 0.5)).mean(),
                            gate=d.g2_w.mean(), oracle=np.maximum(d.psnr_net, d.psnr_wiener).mean())
        for key, name, fmt in (("fa", "false accept (\\%)", "{:.1f}"), ("fr", "false reject (\\%)", "{:.1f}"),
                               ("gate", "gate PSNR", "{:.2f}"), ("oracle", "oracle PSNR", "{:.2f}")):
            cells = [fmt.format(stats[f][key]) for f in FAMS + ["real"]]
            rows.append(f"{label if key == 'fa' else ''} & {name} & " + " & ".join(cells) + " \\\\")
            note(f"DECISION {arm} {key}: " + " | ".join(cells))
        rows.append("\\midrule" if arm != arms[-1][0] else "")
    table("decision", rows, "llccccc",
          "Quality of the gate decision (three seeds, both orders). A false accept keeps a network output that is more "
          "than 0.5\\,dB worse than the Wiener fallback; a false reject discards one that is more than 0.5\\,dB better. "
          "The oracle picks the better of the two per image.", "tab:decision", wide=True)


def ssim_table(g, r):
    """SSIM of the networks and of the gated outputs (gate decisions from gate4, thresholds unchanged)."""
    root = (GATE_OUT / "ssim")
    if not (root / "_ref" / "ssim.csv").exists():
        return
    key = ["image", "kset", "kid", "sigma", "order"]
    ref = pd.read_csv(root / "_ref" / "ssim.csv", dtype={"image": str})
    refr = pd.read_csv(root / "_ref" / "ssim_real.csv")

    def run_vals(run):
        f = root / run / "ssim.csv"
        if not f.exists():
            return None
        s = pd.read_csv(f, dtype={"image": str}).merge(ref, on=key)
        s = s.merge(g[g.run == run][key + ["acc"]], on=key, how="left")
        s["ssim_gate"] = np.where(s.acc, s.ssim_net, s.ssim_wiener)
        sr = pd.read_csv(root / run / "ssim_real.csv").merge(refr, on="pair")
        sr = sr.merge(r[r.run == run][["pair", "acc"]], on="pair", how="left")
        sr["ssim_gate"] = np.where(sr.acc, sr.ssim_net, sr.ssim_wiener)
        out = {f: {c: s[s.kset == f][c].mean() for c in ("ssim_net", "ssim_gate")} for f in FAMS}
        out["real"] = {c: sr[c].mean() for c in ("ssim_net", "ssim_gate")}
        return out

    first = ref.groupby("kset")[["ssim_blind", "ssim_wiener"]].mean()
    rows = ["Model & " + " & ".join(FAM_TEX[f] for f in FAMS + ["real"]) + " \\\\", "\\midrule",
            "Wiener & " + " & ".join(f"{first.loc[f, 'ssim_wiener']:.3f}" for f in FAMS)
            + f" & {refr.ssim_wiener.mean():.3f} \\\\",
            "NAFNet blind (s0) & " + " & ".join(f"{first.loc[f, 'ssim_blind']:.3f}" for f in FAMS)
            + f" & {refr.ssim_blind.mean():.3f} \\\\", "\\midrule"]
    note("SSIM Wiener: " + str(first.ssim_wiener.round(3).to_dict()) + f" real {refr.ssim_wiener.mean():.3f}")
    note("SSIM blind: " + str(first.ssim_blind.round(3).to_dict()) + f" real {refr.ssim_blind.mean():.3f}")

    def add(label, runs, col):
        vals = [run_vals(x) for x in runs]
        vals = [v for v in vals if v is not None]
        if not vals:
            return
        cells = [ms([v[f][col] for v in vals], 3) for f in FAMS + ["real"]]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        note(f"SSIM {label}: " + " | ".join(cells))

    for arm, label in ARMS:
        add(label, [f"{arm}_s{s}" for s in SEEDS], "ssim_net")
    rows.append("\\midrule")
    add("Data step + gate", [f"unrolled_s{s}" for s in SEEDS], "ssim_gate")
    add("Embedding + gate", [f"unrolled_emb_s{s}" for s in SEEDS], "ssim_gate")
    if (root / "usrnet" / "ssim.csv").exists():
        rows.append("\\midrule")
        for run, label in (("usrnet", "USRNet"), ("dpir", "DPIR")):
            add(label, [run], "ssim_net")
            add(f"{label} + gate", [run], "ssim_gate")
    table("ssim", rows, "lccccc",
          "SSIM of the networks alone and of the gated outputs (same images, kernels, noise and gate decisions as "
          "the PSNR results; mean$\\pm$std over three seeds for our networks).", "tab:ssim", wide=True)


def pretrained_table(g, r):
    """Pretrained USRNet / DPIR under the same protocol and gate (official weights, no retraining)."""
    models = [("usrnet", "USRNet~\\cite{zhang2020deep}"), ("dpir", "DPIR~\\cite{zhang2022plug}")]
    if not all((g.run == m).any() for m, _ in models):
        return
    wien = g[g.run == "usrnet"].groupby("kset").psnr_wiener.mean()
    blind = g[g.run == "usrnet"].groupby("kset").psnr_blind.mean()
    rr = r[r.run == "usrnet"]
    rows = ["Model & Decision & " + " & ".join(FAM_TEX[f] for f in FAMS + ["real"]) + " \\\\", "\\midrule",
            "Wiener & -- & " + " & ".join(f"{wien[f]:.2f}" for f in FAMS) + f" & {rr.psnr_wiener.mean():.2f} \\\\",
            "NAFNet blind (s0) & -- & " + " & ".join(f"{blind[f]:.2f}" for f in FAMS)
            + f" & {rr.psnr_blind.mean():.2f} \\\\"]
    for run, label in models:
        rows.append("\\midrule")
        for key, dec in (("psnr_net", "network"), ("g_rho", "$\\rho$-gate"), ("g2_w", "two-sided gate")):
            cells = [f"{g[(g.run == run) & (g.kset == f)][key].mean():.2f}" for f in FAMS]
            cells.append(f"{r[r.run == run][key].mean():.2f}")
            rows.append(f"{label if key == 'psnr_net' else ''} & {dec} & " + " & ".join(cells) + " \\\\")
            note(f"PRETRAINED {run} {dec}: " + " | ".join(cells))
        acc = [f"{100 * g[(g.run == run) & (g.kset == f)].acc.mean():.0f}" for f in FAMS]
        acc.append(f"{100 * r[r.run == run].acc.mean():.0f}")
        rows.append(f" & accepted (\\%) & " + " & ".join(acc) + " \\\\")
        note(f"PRETRAINED {run} accept: " + " | ".join(acc))
        d = g[g.run == run]
        piv = d.groupby(["kset", "sigma"])[["psnr_net", "g2_w", "psnr_wiener", "psnr_blind", "acc"]].mean()
        note(f"PRETRAINED {run} per sigma:\n" + piv.round(2).to_string())
    table("pretrained", rows, "llccccc",
          "Pretrained unfolding and plug-and-play deblurrers with their official weights under our protocol, "
          "with the unchanged gate (thresholds calibrated on our in-distribution data). PSNR in dB, both orders.",
          "tab:pretrained", wide=True)


def alpha_tab():
    f = ROOT / "reports" / "alpha_scaling.csv"
    if not f.exists():
        return
    a = pd.read_csv(f)
    rows = ["Scale of $\\alpha_k$ & " + " & ".join(FAM_TEX[f] for f in FAMS) + " \\\\", "\\midrule"]
    for sc, g in a.groupby("scale"):
        cells = [ms([g[(g.run == f"unrolled_s{s}") & (g.kset == fam)].psnr.mean() for s in SEEDS]) for fam in FAMS]
        rows.append(f"$\\times{sc}$ & " + " & ".join(cells) + " \\\\")
        note(f"ALPHA x{sc}: " + " | ".join(cells))
    piv = a[a.kset == "defocus"].groupby(["scale", "sigma"]).psnr.mean().unstack()
    note("ALPHA defocus per sigma:\n" + piv.round(2).to_string())
    table("alpha", rows, "lcccc",
          "Test-time scaling of the learned data-step weights $\\alpha_k$ of the data-step model "
          "(blur$\\to$noise order, PSNR in dB, mean$\\pm$std over seeds).", "tab:alpha")


def runtime_tab():
    t = pd.read_csv(ROOT / "reports" / "runtime.csv")
    names = {"mix_none_s0": "NAFNet blind", "unrolled_s0": "Unrolled, data step",
             "unrolled_both_s0": "Unrolled, data step + emb.", "wiener": "Wiener filter",
             "gate scores (rho+nu)": "Gate scores $\\rho,\\nu$"}
    rows = ["Component & Params & $256^2$ (ms) & $512^2$ (ms) \\\\", "\\midrule"]
    for what, g in t.groupby("what", sort=False):
        p = g.params.iloc[0]
        ps = f"{p / 1e6:.2f}\\,M" if p else "--"
        rows.append(f"{names[what]} & {ps} & {g[g['size'] == 256].ms.iloc[0]:.2f} & "
                    f"{g[g['size'] == 512].ms.iloc[0]:.2f} \\\\")
    table("runtime", rows, "lccc", "Parameters and GPU time per RGB image (RTX 3090, PyTorch, batch 1).",
          "tab:runtime")


def hypa_note():
    h = pd.read_csv(ROOT / "reports" / "hypa_alpha.csv")
    note("HYPA (mean over unrolled_s*):\n" + h[h.run.str.match(r"unrolled_s\d$")].groupby("sigma")
         [["alpha1", "alpha2", "alpha3", "alpha4", "wiener_lam"]].mean().to_string())


def sigma_notes(g):
    d = g[g.run.str.match(r"unrolled_s\d$")]
    piv = d.groupby(["kset", "sigma"])[["psnr_net", "g2_w", "psnr_blind", "psnr_wiener", "acc"]].mean()
    note("PER SIGMA data step:\n" + piv.round(2).to_string())
    idc = d[d.kset == "gauss"]
    note(f"ID cost data step: net {idc.psnr_net.mean():.3f} gate {idc.g2_w.mean():.3f}; "
         f"false reject {100 * (1 - idc.acc.mean()):.2f} %")


MIXK_GATE = (GATE_OUT / "gate4_mixk")


def mixk_table():
    """Second training distribution (Gaussian + motion kernels, seed 0) next to the Gaussian-trained arms."""
    g = pd.concat([pd.read_csv(f, dtype={"image": str}) for f in glob.glob(str(MIXK_GATE / "*" / "gate4.csv"))])
    r = pd.concat([pd.read_csv(f) for f in glob.glob(str(MIXK_GATE / "*" / "gate4_real.csv"))])
    for d in (g, r):
        d["acc"] = (d.rho <= d.tau) & (d.nu <= d.tau_nu)
        d["g2_w"] = np.where(d.acc, d.psnr_net, d.psnr_wiener)
    fams = FAMS + ["real"]

    def row(label, vals):
        return f"{label} & " + " & ".join(f"{vals[f]:.2f}" for f in fams) + " \\\\"

    body = ["Arm & Gauss & Motion & Levin & Defocus & Real \\\\", "\\midrule",
            "\\multicolumn{6}{l}{\\emph{Gaussian training (three seeds)}} \\\\"]
    for arm, lab in ARMS:
        v = {f: np.mean([synth(f"{arm}_s{s}")[f] for s in SEEDS]) for f in FAMS}
        v["real"] = np.mean([real(f"{arm}_s{s}") for s in SEEDS])
        body.append(row(lab, v))
    body += ["\\midrule", "\\multicolumn{6}{l}{\\emph{Gaussian and motion training (seed 0)}} \\\\"]
    for arm, lab in ARMS:
        run = f"{arm}_mixk_s0"
        v = dict(g[g.run == run].groupby("kset").psnr_net.mean())
        v["real"] = r[r.run == run].psnr_net.mean()
        body.append(row(lab, v))
        note(f"MIXK {run}: " + str({f: round(v[f], 2) for f in fams}))
    for arm, lab in [("unrolled_emb", "Embedding + gate"), ("unrolled", "Data step + gate")]:
        run = f"{arm}_mixk_s0"
        v = dict(g[g.run == run].groupby("kset").g2_w.mean())
        v["real"] = r[r.run == run].g2_w.mean()
        body.append(row(lab, v))
        acc = g[g.run == run].groupby("kset").acc.mean()
        note(f"MIXK gate {run}: " + str({f: round(v[f], 2) for f in fams}) + " accept "
             + str(acc.round(3).to_dict()) + f" real {r[r.run == run].acc.mean():.2f}")
    w = dict(g[g.run == "unrolled_mixk_s0"].groupby("kset").psnr_wiener.mean())
    w["real"] = r[r.run == "unrolled_mixk_s0"].psnr_wiener.mean()
    body += ["\\midrule", row("Wiener", w)]
    table("mixk", body, "lccccc",
          "Second training distribution (PSNR in dB, both formation orders). Motion is out of distribution for the "
          "Gaussian-trained arms and in distribution for the mixture-trained arms; Levin, defocus and the real "
          "captures are out of distribution for both. The gate uses thresholds calibrated on the mixture.",
          "tab:mixk")


def realblind_table():
    """Real captures with the oracle noise level versus blind estimates from y alone (MAD, Immerkaer)."""
    roots = {"oracle": ("gate4", "gate4_mixk"), "mad": ("gate4_real_mad", "gate4_real_mad_mixk"),
             "immerkaer": ("gate4_real_immerkaer", "gate4_real_immerkaer_mixk")}
    res = {}
    for est, (rg, rm) in roots.items():
        fs = glob.glob(f"{GATE_OUT}/{rg}/unrolled*_s[012]/gate4_real.csv") + \
            glob.glob(f"{GATE_OUT}/{rm}/unrolled*_mixk_s0/gate4_real.csv")
        r = pd.concat([pd.read_csv(f) for f in fs])
        r["acc"] = (r.rho <= r.tau) & (r.nu <= r.tau_nu)
        r["g2_w"] = np.where(r.acc, r.psnr_net, r.psnr_wiener)
        r["arm"] = r.run.str.replace(r"_s\d$", "", regex=True)
        res[est] = r
        note(f"REALBLIND {est}: sigma {r.groupby('pair').sigma_est.first().mean():.2f}, "
             + str(r.groupby("arm")[["psnr_net", "g2_w", "acc"]].mean().round(2).to_dict()))
    ests = list(roots)

    def line(label, f):
        return f"{label} & " + " & ".join(f"{f(res[e]):.2f}" for e in ests) + " \\\\"

    sig = {e: res[e].groupby("pair").sigma_est.first().mean() for e in ests}
    gauss = lambda r: r[~r.arm.str.contains("mixk")]
    body = ["Method & Oracle & MAD & Immerk{\\ae}r \\\\", "\\midrule",
            "mean $\\sigma$ (gray levels) & " + " & ".join(f"{sig[e]:.2f}" for e in ests) + " \\\\", "\\midrule",
            line("Wiener", lambda r: r.groupby("pair").psnr_wiener.first().mean()),
            line("Blind NAFNet", lambda r: gauss(r).groupby("pair").psnr_blind.first().mean())]
    for arm, lab in [("unrolled", "Data step"), ("unrolled_emb", "Embedding"), ("unrolled_mixk", "Data step (mixture)")]:
        body.append(line(lab, lambda r, a=arm: r[r.arm == a].psnr_net.mean()))
        body.append(line(f"{lab} + gate", lambda r, a=arm: r[r.arm == a].g2_w.mean()))
    table("realblind", body, "lccc",
          "Real captures with oracle and blind noise levels (PSNR in dB; Gaussian training: three seeds, "
          "mixture: seed 0). Estimates below $0.5$ gray levels are raised to $0.5$.", "tab:realblind")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g, r = load_gate()
    factorial(g)
    gate_table(g, r)
    accept(g, r)
    other_nets(g, r)
    rule_table(g, r)
    pretrained_table(g, r)
    decision_table(g, r)
    ssim_table(g, r)
    ablation(g, r)
    alpha_tab()
    runtime_tab()
    hypa_note()
    sigma_notes(g)
    mixk_table()
    realblind_table()
    (ROOT / "reports" / "paper_numbers.md").write_text("\n".join(NOTES), encoding="utf-8")
    print("\n".join(NOTES))


if __name__ == "__main__":
    main()
