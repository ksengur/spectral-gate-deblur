"""Paper figures (vector PDF + 300 dpi PNG) into paper/figs/. Run from the project root:
    python paper/make_figures.py [--only factorial pipeline scores spectrum sigma visual real]

fig_factorial : PSNR difference vs blind NAFNet, route x kernel family (diverging map, teaser)
fig_pipeline  : method diagram with real thumbnails of one defocus example (routes A/B, scores, gate)
fig_scores    : rho vs nu of the data-step net at sigma=2 with marginal distributions and thresholds
fig_spectrum  : (top) radial error power of embedding / data-step / Wiener, (bottom) |K|^2 with the learned
                alpha_t and the region the null-space score watches (W > 1/2); motion and defocus
fig_sigma     : PSNR vs noise level per kernel family (data-step net, gate, blind NAFNet, Wiener)
fig_visual    : synthetic examples (motion, Levin, defocus), sigma=2, blur->noise
fig_real      : real Levin capture example
Data: runs*/*/eval*.csv, outputs/gate4/*/gate4*.csv, reports/hypa_alpha.csv.
"""

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from paths import GATE_OUT, DIV2K_DIR  # noqa: E402
import argparse
import glob
import sys
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figs"
GATE = (GATE_OUT / "gate4")

# reference palette (dataviz skill): categorical slots, ink tokens, diverging red <-> grey <-> blue
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3, GRID = "#1a1a19", "#52514e", "#8a8985", "#e4e3df"
TINT = {"blue": "#e8f1fc", "orange": "#fdeee6", "aqua": "#e4f6ef", "grey": "#f3f2ef"}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#b8302f", "#e34948", "#f3f2ef", "#3987e5", "#1c5cab"])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.2, "legend.handlelength": 1.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL1, COL2 = 3.5, 7.16  # IEEE single / double column width (in)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def clean(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def panel(ax, text, y=-0.30):
    """IEEE-style sub-caption '(a) ...' centered under an axes."""
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top", fontsize=8)


def load_gate(pattern="*"):
    g = pd.concat([pd.read_csv(f, dtype={"image": str}) for f in glob.glob(str(GATE / pattern / "gate4.csv"))])
    g["acc"] = (g.rho <= g.tau) & (g.nu <= g.tau_nu)
    g["gate"] = np.where(g.acc, g.psnr_net, g.psnr_wiener)
    return g


# ================================================================ factorial heatmap (teaser)
def fig_factorial():
    def load_csvs(pattern):
        fs = sorted(glob.glob(str(ROOT / pattern)))
        return pd.concat([pd.read_csv(f, dtype={"image": str, "token": str}) for f in fs]) if fs else pd.DataFrame()

    allp = pd.concat([load_csvs("runs*/*/eval.csv"), load_csvs("runs*/*/eval_shift.csv")])
    allp = allp[allp.token.isin(["true", "unknown"])]
    blind = allp[allp.run.str.startswith("mix_none")]
    g4 = load_gate()
    rows = [("No kernel", "unrolled_blind"), ("Embedding (B)", "unrolled_emb"), ("Data step (A)", "unrolled"),
            ("A + B", "unrolled_both"), ("A + gate", "unrolled+gate"), ("B + gate", "unrolled_emb+gate")]
    cols = [("Gauss", "gauss"), ("Motion", "motion"), ("Levin", "levin"), ("Defocus", "defocus")]
    M = np.full((len(rows), len(cols)), np.nan)
    S = np.full_like(M, np.nan)
    for r, (_, cond) in enumerate(rows):
        for c, (_, kset) in enumerate(cols):
            diffs = []
            for seed in (0, 1, 2):
                ref = blind[(blind.run == f"mix_none_s{seed}") & (blind.kset == kset)].groupby("image").psnr.mean()
                if cond.endswith("+gate"):
                    a = g4[(g4.run == f"{cond[:-5]}_s{seed}") & (g4.kset == kset)].groupby("image").gate.mean()
                else:
                    a = allp[(allp.run == f"{cond}_s{seed}") & (allp.kset == kset)].groupby("image").psnr.mean()
                if len(a) and len(ref):
                    diffs.append((a - ref).mean())
            if diffs:
                M[r, c], S[r, c] = np.mean(diffs), np.std(diffs)
    fig, ax = plt.subplots(figsize=(COL1, 2.45))
    im = ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(0, -7, 3), aspect="auto")
    for r in range(len(rows)):
        for c in range(len(cols)):
            v = M[r, c]
            dark = v < -3.5 or v > 2.2
            ax.text(c, r - 0.13, f"{v:+.2f}", ha="center", va="center", fontsize=7.2,
                    color="white" if dark else INK, fontweight="bold" if r >= 4 else "normal")
            ax.text(c, r + 0.24, f"$\\pm${S[r, c]:.2f}", ha="center", va="center", fontsize=5.6,
                    color="white" if dark else INK2)
    ax.set_xticks(range(len(cols)), [c[0] for c in cols])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.axhline(3.5, color="white", lw=3)
    ax.axvline(0.5, color="white", lw=3)
    ax.tick_params(length=0, pad=2)
    clean(ax, keep=())
    ax.text(0, -1.05, "train", ha="center", va="bottom", fontsize=6.5, color=INK3, style="italic")
    ax.text(2, -1.05, "unseen kernel families", ha="center", va="bottom", fontsize=6.5, color=INK3, style="italic")
    ax.annotate("", xy=(0.62, -0.98), xytext=(3.45, -0.98), xycoords="data", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.6))
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025, ticks=[-6, -4, -2, 0, 2])
    cb.set_label("$\\Delta$PSNR vs. blind NAFNet (dB)", fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2)
    save(fig, "fig_factorial")


# ================================================================ model helpers
def models_and_data():
    import torch
    from degrade_torch import degrade_pair, eval_kernel_sets, wiener_deconv
    from eval_shift import load, predict
    from ext_kernels import shift_kernel_sets
    from train_2x2 import build_cache, center_crops
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = {k: load(ROOT / v, dev) for k, v in
            {"blind": "runs/mix_none_s0", "emb": "runs_cond/unrolled_emb_s0", "data": "runs_cond/unrolled_s0"}.items()}
    files = build_cache((DIV2K_DIR / "DIV2K_valid_HR"), ROOT / "cache/valid")
    images = center_crops(files, 256)
    ksets = dict(eval_kernel_sets(), **shift_kernel_sets())
    return torch, dev, nets, files, images, ksets, degrade_pair, wiener_deconv, predict


def restore_all(ctx, x, kern, kset, kid, sigma, i_batch):
    torch, dev, nets, files, images, ksets, degrade_pair, wiener_deconv, predict = ctx
    k = torch.from_numpy(kern).float().to(dev)[None]
    s = torch.tensor([float(sigma)], device=dev)
    gen = torch.Generator(device=dev).manual_seed(zlib.crc32(f"fig|{kset}|{kid}|{sigma}|{i_batch}".encode()))
    t = nets["blind"][1]
    y = degrade_pair(x, k, s, t["mode"], t["floor"], gen)[0]
    with torch.no_grad():
        out = {"Input": y, "Blind NAFNet": predict(*nets["blind"], y, k, s),
               "Embedding (B)": predict(*nets["emb"], y, k, s), "Data step (A)": predict(*nets["data"], y, k, s),
               "Wiener": wiener_deconv(y, k, s, t["floor"])}
    return {n: v.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() for n, v in out.items()}, k


def psnr_np(a, b, border=10):
    a, b = a[border:-border, border:-border], b[border:-border, border:-border]
    return 10 * np.log10(1 / max(((a - b) ** 2).mean(), 1e-12))


def gate_scores(ctx, outs, k, sigma):
    """rho, nu of the data-step output and the data-step net's calibrated thresholds at sigma."""
    torch, dev = ctx[0], ctx[1]
    from gate3 import rho_fn
    from gate4 import nu_fn
    g = pd.read_csv(GATE / "unrolled_s0" / "gate4.csv")
    row = g[g.sigma == sigma].iloc[0]
    t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().to(dev)
    sl = (..., slice(10, -10), slice(10, -10))
    floor = ctx[2]["blind"][1]["floor"]
    est, xw, yy = t(outs["Data step (A)"]), t(outs["Wiener"]), t(outs["Input"])
    rho = float(rho_fn(est, yy, k, sigma, floor, sl))
    nu = float(nu_fn(est, xw, k, sigma, floor, sl))
    return rho, nu, float(row.tau), float(row.tau_nu)


def textured_index(images, n=30):
    grad = [float((im[:, :, 1:] - im[:, :, :-1]).abs().mean()) for im in images[:n]]
    return int(np.argmax(grad))


# ================================================================ method diagram with thumbnails
def fig_pipeline(kset="defocus", kid=3, sigma=2):
    ctx = models_and_data()
    torch, dev, nets, files, images, ksets, *_ = ctx
    idx = textured_index(images)
    x = images[idx:idx + 1].to(dev)
    outs, k = restore_all(ctx, x, ksets[kset][kid], kset, kid, sigma, idx)
    rho, nu, tr, tn = gate_scores(ctx, outs, k, sigma)
    ok = rho <= tr and nu <= tn
    final = outs["Data step (A)"] if ok else outs["Wiener"]
    crop = lambda a: a[64:192, 64:192]

    W, H = 716, 262  # canvas in 1/100 inch
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x0, y0, w, h, title, sub=None, tint="grey", edge=INK2, lw=0.8, ls="-"):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=6", fc=TINT[tint],
                                    ec=edge, lw=lw, ls=ls, zorder=2))
        if sub is None:
            ax.text(x0 + w / 2, y0 + h / 2, title, ha="center", va="center", fontsize=7.2, color=INK, zorder=3)
        else:
            ax.text(x0 + w / 2, y0 + h * 0.66, title, ha="center", va="center", fontsize=7.2, color=INK, zorder=3)
            ax.text(x0 + w / 2, y0 + h * (0.30 if "\n" not in sub else 0.32), sub, ha="center", va="center",
                    fontsize=6.6, color=INK2, zorder=3, linespacing=1.25)

    def arrow(p, q, col=INK2, lw=0.9, rad=0.0, ls="-", z=1):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=3.2,head_width=1.8", color=col, lw=lw,
                                     ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))

    def thumb(x0, y0, size, img, label, edge=INK3, cmap=None):
        a = fig.add_axes([x0 / W, y0 / H, size / W, size / H])
        a.imshow(img, cmap=cmap, interpolation="nearest" if cmap else "lanczos")
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(edge)
            s.set_linewidth(0.9)
        ax.text(x0 + size / 2, y0 - 6, label, ha="center", va="top", fontsize=6.8, color=INK)
        return a

    T = 74
    # --- inputs
    thumb(8, 150, T, crop(outs["Input"]), "observation $\\mathbf{y}$")
    kimg = ksets[kset][kid]
    thumb(24, 50, 42, kimg / kimg.max(), "kernel $k$, noise $\\sigma$", cmap="gray")
    # --- unrolled network frame
    ax.add_patch(FancyBboxPatch((112, 88), 232, 170, boxstyle="round,pad=0,rounding_size=8", fc="white",
                                ec=INK3, lw=0.8, ls=(0, (3, 2)), zorder=0))
    ax.text(228, 248, "unrolled HQS network, $T=4$ stages", ha="center", va="center", fontsize=7.2, color=INK)
    box(126, 168, 92, 46, "data step", "closed form, FFT", tint="blue", edge=BLUE)
    box(240, 168, 90, 46, "prior $\\mathcal{D}_\\theta$", "NAFNet denoiser")
    box(240, 100, 90, 42, "kernel code", "$\\phi(k,\\sigma)\\rightarrow$ FiLM", tint="orange", edge=ORANGE)
    arrow((218, 191), (240, 191))
    arrow((285, 142), (285, 168), col=ORANGE)
    arrow((285, 214), (172, 214), rad=0.3)
    ax.text(228, 222, "$\\times T$", ha="center", va="center", fontsize=6.8, color=INK2)
    # routes
    arrow((82, 187), (126, 191))
    arrow((66, 80), (126, 176), col=BLUE, lw=1.1)
    ax.text(48, 122, "A: kernel in\ndata step", ha="center", va="center", fontsize=6.6, color=BLUE, linespacing=1.1)
    arrow((66, 66), (240, 116), col=ORANGE, lw=1.1)
    ax.text(170, 72, "B: kernel as code", ha="center", va="center", fontsize=6.6, color=ORANGE)
    # --- outputs of the two estimators
    thumb(362, 168, T, crop(outs["Data step (A)"]), "network output $\\hat{\\mathbf{x}}$")
    arrow((344, 191), (362, 205))
    box(126, 14, 92, 40, "Wiener filter", "$k$, $\\sigma$, gradient prior", tint="aqua", edge=AQUA)
    arrow((66, 58), (126, 36))
    thumb(362, 22, T, crop(outs["Wiener"]), "Wiener estimate $\\mathbf{x}_W$")
    arrow((218, 34), (362, 52))
    # --- scores
    box(462, 186, 104, 46, "re-blurring residual $\\rho$", "$\\|k\\ast\\hat{\\mathbf{x}}-\\mathbf{y}\\|^2/(Ps^2)$", tint="blue", edge=BLUE)
    box(462, 112, 104, 46, "null-space deviation $\\nu$", "$\\|W\\odot\\mathcal{F}(\\hat{\\mathbf{x}}-\\mathbf{x}_W)\\|^2/P$",
        tint="aqua", edge=AQUA)
    arrow((436, 210), (462, 209))
    arrow((436, 196), (462, 140))
    arrow((436, 66), (462, 128))
    pass_r, pass_n = rho <= tr, nu <= tn
    ax.text(514, 178, f"here {rho:.2f} vs $\\tau_\\rho$ {tr:.2f}: " + ("pass" if pass_r else "fail"),
            ha="center", va="center", fontsize=6.0, color=AQUA if pass_r else "#c0392b")
    ax.text(514, 104, f"here {nu:.3f} vs $\\tau_\\nu$ {tn:.4f}: " + ("pass" if pass_n else "fail"),
            ha="center", va="center", fontsize=6.0, color=AQUA if pass_n else "#c0392b")
    # --- decision
    box(584, 134, 46, 76, "both\npass?")
    arrow((566, 209), (584, 186))
    arrow((566, 135), (584, 158))
    thumb(640, 150, T - 6, crop(final), "gated output", edge=AQUA if not ok else BLUE)
    arrow((630, 180), (640, 184))
    ax.text(674, 126, "yes $\\rightarrow\\hat{\\mathbf{x}}$,  no $\\rightarrow\\mathbf{x}_W$", ha="center", va="top", fontsize=6.2,
            color=INK2)
    # --- calibration
    box(462, 14, 246, 62, "calibration, once per network (in-distribution data only)",
        "held-out images $\\times$ Gaussian training-type kernels\n$\\rightarrow$ 99 % quantiles $\\tau_\\rho(\\sigma)$, $\\tau_\\nu(\\sigma)$",
        tint="grey", edge=INK3, ls=(0, (3, 2)))
    arrow((607, 76), (607, 134), ls=(0, (3, 2)))
    save(fig, "fig_pipeline")


# ================================================================ score scatter with marginals
def fig_scores(run="unrolled_s0", sigma=2):
    g = pd.read_csv(GATE / run / "gate4.csv", dtype={"image": str})
    g = g[(g.sigma == sigma) & (g.order == "bn")]
    tr, tn = g.tau.iloc[0], g.tau_nu.iloc[0]
    fams = (("gauss", BLUE, "Gaussian (ID)"), ("motion", ORANGE, "Motion"), ("defocus", AQUA, "Defocus"))
    fig = plt.figure(figsize=(COL1, 2.75))
    gs = fig.add_gridspec(2, 2, width_ratios=(4.2, 1), height_ratios=(1, 4.2), wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    rb = np.logspace(np.log10(g.rho.min() * 0.9), np.log10(g.rho.max() * 1.1), 40)
    nb = np.logspace(np.log10(g.nu.min() * 0.9), np.log10(g.nu.max() * 1.1), 40)
    ax.add_patch(Rectangle((rb[0], nb[0]), tr - rb[0], tn - nb[0], fc=TINT["blue"], ec="none", zorder=0))
    for kset, col, lab in fams:
        s = g[g.kset == kset]
        ax.scatter(s.rho, s.nu, s=6, c=col, alpha=0.55, edgecolors="none", label=lab, zorder=2)
        axt.hist(s.rho, bins=rb, histtype="stepfilled", color=col, alpha=0.25, lw=0)
        axt.hist(s.rho, bins=rb, histtype="step", color=col, lw=0.9)
        axr.hist(s.nu, bins=nb, histtype="stepfilled", color=col, alpha=0.25, lw=0, orientation="horizontal")
        axr.hist(s.nu, bins=nb, histtype="step", color=col, lw=0.9, orientation="horizontal")
    for a in (ax, axt):
        a.axvline(tr, color=INK, lw=0.8, ls=(0, (4, 2)))
    for a in (ax, axr):
        a.axhline(tn, color=INK, lw=0.8, ls=(0, (4, 2)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(rb[0], rb[-1])
    ax.set_ylim(nb[0], nb[-1])
    ax.text(tr * 1.08, nb[-1] * 0.6, "$\\tau_\\rho$", fontsize=8, va="top", color=INK)
    ax.text(rb[-1] * 0.92, tn * 1.25, "$\\tau_\\nu$", fontsize=8, ha="right", va="bottom", color=INK)
    ax.text(tr * 0.92, nb[0] * 1.3, "accept", fontsize=7, ha="right", va="bottom", color=BLUE, style="italic",
            zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel("re-blurring residual $\\rho$")
    ax.set_ylabel("null-space deviation $\\nu$")
    ax.grid(True, which="major", color=GRID, lw=0.4)
    clean(ax)
    for a in (axt, axr):
        a.axis("off")
    ax.legend(frameon=False, loc="lower right", markerscale=2.2, handletextpad=0.2, borderaxespad=0.2)
    save(fig, "fig_scores")


# ================================================================ spectra: error power and |K|^2 with alpha_t
def radial(a):
    h, w = a.shape
    yy, xx = np.indices((h, w))
    r = np.hypot(yy - h // 2, xx - w // 2).astype(int)
    return np.bincount(r.ravel(), a.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def fig_spectrum(sigma=2, n_img=10, size=256):
    ctx = models_and_data()
    torch, dev, nets, files, images, ksets, *_ = ctx
    hyp = pd.read_csv(ROOT / "reports" / "hypa_alpha.csv")
    al = hyp[(hyp.run == "unrolled_s0") & (hyp.sigma == sigma)].iloc[0]
    floor = nets["blind"][1]["floor"]
    lam = 300 * (sigma ** 2 + floor ** 2) / 255 ** 2
    gx = np.zeros((size, size))
    gx[0, 0], gx[0, 1] = -1, 1
    G2 = np.abs(np.fft.fft2(gx)) ** 2 + np.abs(np.fft.fft2(gx.T)) ** 2
    fig = plt.figure(figsize=(COL2, 2.75))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.55, 1), hspace=0.08, wspace=0.22)
    series = (("Embedding (B)", ORANGE, "-"), ("Data step (A)", BLUE, "-"), ("Wiener", AQUA, (0, (3, 1.5))))
    for c, (kset, kid, title) in enumerate((("motion", 3, "(a) Motion kernel"), ("defocus", 3, "(b) Defocus kernel"))):
        kern = ksets[kset][kid]
        acc = {n: 0 for n, *_ in series}
        for i in range(n_img):
            x = images[i:i + 1].to(dev)
            outs, _ = restore_all(ctx, x, kern, kset, kid, sigma, i)
            gt = x[0].permute(1, 2, 0).cpu().numpy()
            for n in acc:
                e = (outs[n] - gt).mean(2)
                acc[n] = acc[n] + np.abs(np.fft.fftshift(np.fft.fft2(e))) ** 2 / e.size
        kp = np.zeros((size, size))
        kp[:kern.shape[0], :kern.shape[1]] = kern
        K2 = np.abs(np.fft.fft2(kp)) ** 2
        Wf = lam * G2 / (K2 + lam * G2 + 1e-20)
        f = np.arange(size // 2 + 1) / size
        k2r = radial(np.fft.fftshift(K2))[:len(f)]
        wr = radial(np.fft.fftshift(Wf))[:len(f)]
        watch = wr > 0.5
        ax = fig.add_subplot(gs[0, c])
        axk = fig.add_subplot(gs[1, c], sharex=ax)
        for a in (ax, axk):
            for lo, hi in _runs(f, watch):
                a.axvspan(lo, hi, color=TINT["aqua"], lw=0, zorder=0)
        for n, col, ls in series:
            ax.semilogy(f, radial(acc[n] / n_img)[:len(f)], color=col, ls=ls, lw=1.2, label=n)
        ax.set_ylabel("error power")
        ax.set_title(title, loc="left", fontsize=8)
        ax.grid(True, color=GRID, lw=0.4)
        plt.setp(ax.get_xticklabels(), visible=False)
        clean(ax)
        axk.semilogy(f, k2r, color=INK, lw=1.1, label="$|K(f)|^2$")
        for j, (name, lsty) in enumerate((("alpha1", (0, (1, 1.5))), ("alpha4", (0, (4, 2))))):
            axk.axhline(al[name], color=BLUE, lw=0.8, ls=lsty)
            axk.text(0.497, al[name] * 1.6, f"$\\alpha_{name[-1]}$", ha="right", va="bottom", fontsize=7, color=BLUE)
        axk.set_ylim(1e-8, 2)
        axk.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
        axk.set_ylabel("$|K(f)|^2$")
        axk.set_xlabel("radial frequency (cycles/pixel)")
        axk.set_xlim(0, 0.5)
        axk.grid(True, color=GRID, lw=0.4)
        clean(axk)
        if c == 0:
            h, l = ax.get_legend_handles_labels()
            h.append(Rectangle((0, 0), 1, 1, fc=TINT["aqua"], ec="none"))
            l.append("region watched by $\\nu$ ($W>1/2$)")
            fig.legend(h, l, frameon=False, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.04))
    save(fig, "fig_spectrum")


def _runs(f, mask):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = f[i]
        if not m and start is not None:
            out.append((start, f[i]))
            start = None
    if start is not None:
        out.append((start, f[-1]))
    return out


# ================================================================ PSNR vs noise level
def fig_sigma():
    g4 = load_gate("unrolled_s?")
    fams = [("gauss", "(a) Gaussian (ID)"), ("motion", "(b) Motion"), ("levin", "(c) Levin"), ("defocus", "(d) Defocus")]
    lines = [("gate", "Data step + gate (ours)", INK, "-", "s", 1.5), ("psnr_net", "Data-step network", BLUE, "-", "o", 1.1),
             ("psnr_blind", "Blind NAFNet", ORANGE, (0, (4, 2)), "^", 1.1),
             ("psnr_wiener", "Wiener", AQUA, (0, (1.5, 1.2)), "D", 1.1)]
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 1.95))
    for ax, (kset, title) in zip(axes, fams):
        d = g4[g4.kset == kset]
        per_seed = d.groupby(["run", "sigma"])[[c for c, *_ in lines]].mean()
        m, s = per_seed.groupby("sigma").mean(), per_seed.groupby("sigma").std()
        for col, lab, c, ls, mk, lw in lines[::-1]:
            ax.plot(m.index, m[col], color=c, ls=ls, marker=mk, ms=3.0, lw=lw, label=lab,
                    mec="white", mew=0.4, zorder=3 if col == "gate" else 2)
            if col in ("gate", "psnr_net"):
                ax.fill_between(m.index, m[col] - s[col], m[col] + s[col], color=c, alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 15, 25], ["2", "5", "15", "25"])
        ax.minorticks_off()
        ax.set_xlabel("noise level $\\sigma$")
        ax.grid(True, color=GRID, lw=0.4)
        clean(ax)
        panel(ax, title, y=-0.36)
    axes[0].set_ylabel("PSNR (dB)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], lab[::-1], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_sigma")


# ================================================================ qualitative figures
def _frame(ax, color, lw=1.6):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(color)
        s.set_linewidth(lw)


def fig_visual(sigma=2):
    ctx = models_and_data()
    torch, dev, nets, files, images, ksets, *_ = ctx
    idx = textured_index(images)
    x = images[idx:idx + 1].to(dev)
    gt = x[0].permute(1, 2, 0).cpu().numpy()
    cases = [("motion", 3, "Motion"), ("levin", 2, "Levin"), ("defocus", 3, "Defocus")]
    names = ["Input", "Blind NAFNet", "Embedding (B)", "Data step (A)", "A + gate (ours)", "Wiener", "Ground truth"]
    fig, axes = plt.subplots(len(cases), len(names), figsize=(COL2, COL2 / len(names) * len(cases) * 1.14))
    zy, zx, zs = 64, 64, 128
    for r, (kset, kid, rname) in enumerate(cases):
        outs, k = restore_all(ctx, x, ksets[kset][kid], kset, kid, sigma, idx)
        rho, nu, tr, tn = gate_scores(ctx, outs, k, sigma)
        ok = rho <= tr and nu <= tn
        outs["A + gate (ours)"] = outs["Data step (A)"] if ok else outs["Wiener"]
        outs["Ground truth"] = gt
        best = max(psnr_np(outs[n], gt) for n in names[:-1])
        for c, n in enumerate(names):
            ax = axes[r, c]
            ax.imshow(outs[n][zy:zy + zs, zx:zx + zs], interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            clean(ax, keep=())
            if n == "A + gate (ours)":
                _frame(ax, BLUE if ok else AQUA)
            if n != "Ground truth":
                p = psnr_np(outs[n], gt)
                lab = f"{p:.2f} dB"
                if n == "A + gate (ours)":
                    lab += "  (net)" if ok else "  (Wiener)"
                ax.set_xlabel(lab, fontsize=6.8, labelpad=1.5, fontweight="bold" if abs(p - best) < 1e-6 else "normal")
            if r == 0:
                ax.set_title(n, fontsize=7.2, pad=3)
            if c == 0:
                ax.set_ylabel(rname, fontsize=7.5)
        kin = axes[r, 0].inset_axes([0.0, 0.64, 0.36, 0.36])
        kk = ksets[kset][kid]
        kin.imshow(kk / kk.max(), cmap="gray", interpolation="nearest")
        kin.set_xticks([])
        kin.set_yticks([])
        _frame(kin, "white", lw=0.8)
    fig.subplots_adjust(wspace=0.04, hspace=0.22)
    save(fig, "fig_visual")


def fig_real(pair="im05_k6"):
    ctx = models_and_data()
    torch, dev, nets, files, images, ksets, degrade_pair, wiener_deconv, predict = ctx
    from ext_kernels import levin_real_pairs
    from eval_shift import aligned_psnr
    name, x, y, kern = [p for p in levin_real_pairs() if p[0] == pair][0]
    real = pd.read_csv(GATE / "unrolled_s0" / "gate4_real.csv")
    rr = real[real.pair == pair].iloc[0]
    ok = rr.rho <= rr.tau and rr.nu <= rr.tau_nu
    k = torch.from_numpy(kern).float().to(dev)[None]
    s = torch.tensor([max(float(rr.sigma_est), 0.5)], device=dev)
    yt = torch.from_numpy(y).to(dev)[None, None].expand(1, 3, -1, -1).contiguous()
    with torch.no_grad():
        outs = {"Real blurred input": yt, "Blind NAFNet": predict(*nets["blind"], yt, k, s),
                "Embedding (B)": predict(*nets["emb"], yt, k, s), "Data step (A)": predict(*nets["data"], yt, k, s),
                "Wiener": wiener_deconv(yt, k, s, nets["blind"][1]["floor"])}
    outs = {n: v.clamp(0, 1).mean(1)[0].cpu().numpy() for n, v in outs.items()}
    outs["A + gate (ours)"] = outs["Data step (A)"] if ok else outs["Wiener"]
    outs["Sharp capture"] = x
    names = ["Real blurred input", "Blind NAFNet", "Embedding (B)", "Data step (A)", "A + gate (ours)", "Wiener",
             "Sharp capture"]
    fig, axes = plt.subplots(1, len(names), figsize=(COL2, COL2 / len(names) * 1.3))
    ps = {n: aligned_psnr(outs[n], x) for n in names[:-1]}
    best = max(ps.values())
    for ax, n in zip(axes, names):
        ax.imshow(outs[n][40:216, 40:216], cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        clean(ax, keep=())
        ax.set_title(n, fontsize=7.2, pad=3)
        if n == "A + gate (ours)":
            _frame(ax, BLUE if ok else AQUA)
        if n != "Sharp capture":
            lab = f"{ps[n]:.2f} dB"
            if n == "A + gate (ours)":
                lab += "  (net)" if ok else "  (Wiener)"
            ax.set_xlabel(lab, fontsize=6.8, labelpad=1.5, fontweight="bold" if abs(ps[n] - best) < 1e-6 else "normal")
    kin = axes[0].inset_axes([0.0, 0.66, 0.34, 0.34])
    kin.imshow(kern / kern.max(), cmap="gray", interpolation="nearest")
    kin.set_xticks([])
    kin.set_yticks([])
    _frame(kin, "white", lw=0.8)
    fig.subplots_adjust(wspace=0.04)
    save(fig, "fig_real")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*",
                   default=["factorial", "pipeline", "scores", "spectrum", "sigma", "visual", "real"])
    a = p.parse_args()
    for name in a.only:
        globals()[f"fig_{name}"]()
