#!/usr/bin/env python3
"""Render the four figures for nueronce_paper.tex as vector PDFs.

Print-friendly: one accent pair (blue #1D4ED8 / orange #C2410C, CVD-validated),
neutral grays for structure, dash patterns as secondary encoding so every
distinction survives grayscale printing. Figure 4 plots the REAL held-out
bits-per-byte log from the 2026-07-02 A100 session (docs/reports/
COLAB_A100_SESSION_2026-07-02.md); the interval log below was transcribed
from the live training output.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE = "#1D4ED8", "#C2410C"
INK, MUTED, FAINT = "#1F2937", "#6B7280", "#D1D5DB"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "figure.dpi": 200,
})


def box(ax, x, y, w, h, text, fc="#F3F4F6", ec=MUTED, fs=8, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, color=INK, ls="-", lw=1.2, alpha=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=9, color=color, ls=ls, lw=lw,
                                 alpha=alpha, shrinkA=1, shrinkB=1))


# ---------------------------------------------------------------- Figure 1
def fig1():
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
    stages = [
        (1.0, 11.0, "Raw byte\nstream", "#FFFFFF"),
        (14.0, 12.5, "Perception\nCNN (causal\nconvs +\nRMSNorm)", "#F3F4F6"),
        (28.5, 13.0, "Dynamic\npatcher\n(thresholded\nboundary prob)", "#DBEAFE"),
        (43.5, 13.0, "Unit embedder\n+ typed\nmemory\n(7 channels)", "#F3F4F6"),
        (58.5, 14.0, "Hybrid routed\ncore (4 paths,\nrouter, logical-\ndepth reuse)", "#DBEAFE"),
        (74.5, 13.0, "Byte decoder\n(completed-\nunit cross-\nattention)", "#F3F4F6"),
        (89.5, 10.0, "Next-byte\nlogits", "#FFFFFF"),
    ]
    h, y0 = 15, 11
    xs = []
    for x, ww, label, fc in stages:
        box(ax, x, y0, ww, h, label, fc=fc, fs=6.3)
        xs.append((x, x + ww))
    for (_l1, r1), (l2, _r2) in zip(xs[:-1], xs[1:]):
        arrow(ax, r1 + 0.1, y0 + h / 2, l2 - 0.1, y0 + h / 2)
    # boundary head + two gradient routes
    box(ax, 16.0, 1.0, 11.5, 6.5, "boundary head\n(per-byte logit)", fc="#FFF7ED", fs=6.6)
    arrow(ax, 21.0, 11, 21.5, 7.7, color=MUTED, lw=1.0)
    arrow(ax, 36.0, 11.0, 28.0, 4.5, color=ORANGE, ls=(0, (4, 2)), lw=1.2)
    ax.text(38.5, 3.2, "LM-loss gradient via per-unit\nmean-boundary feature",
            fontsize=6.2, color=ORANGE, ha="left")
    arrow(ax, 9.5, 4.2, 15.8, 4.2, color=BLUE, ls=(0, (4, 2)), lw=1.2)
    ax.text(0.5, 1.2, "aux word-onset\nboundary loss", fontsize=6.2, color=BLUE)
    ax.text(35.0, 29.5, "discrete cuts detached from gradient", fontsize=6.6,
            color=MUTED, style="italic")
    arrow(ax, 47.0, 29.0, 35.5, 26.5, color=MUTED, ls=":", lw=0.9)
    fig.savefig(OUT / "fig1_pipeline.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_pipeline.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def fig2():
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    box(ax, 34, 88, 32, 9, "normalized input", fc="#FFFFFF")
    paths = [
        (2, "selective\nSSM scan"), (27, "local windowed\nattention"),
        (52, "sparse-global\ntop-k attention"), (77, "retrieval\ncross-attention"),
    ]
    for x, label in paths:
        box(ax, x, 62, 21, 14, label, fc="#DBEAFE", fs=7.2)
        arrow(ax, 50, 88, x + 10.5, 76.5, color=MUTED, lw=1.0)
        arrow(ax, x + 10.5, 62, 50, 47.5, color=MUTED, lw=1.0)
    box(ax, 30, 38, 40, 9.5, "router MLP -> per-position\n4-way softmax blend", fc="#FFF7ED", fs=7.2)
    ax.text(74, 43, "causal outputs only", fontsize=6.4, color=MUTED, style="italic")
    arrow(ax, 50, 38, 50, 29.5)
    box(ax, 30, 19, 40, 9.5, "gated feed-forward + residual", fc="#F3F4F6", fs=7.2)
    arrow(ax, 50, 19, 50, 10.5)
    box(ax, 34, 1, 32, 9, "block output", fc="#FFFFFF")
    ax.text(2, 92, "x logical-depth reuse", fontsize=7.0, color=BLUE, fontweight="bold")
    fig.savefig(OUT / "fig2_coreblock.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2_coreblock.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def fig3():
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    cols = [(2, "Framework backend\n(primitive tensor ops)", BLUE),
            (66, "NumPy autograd engine\n(from scratch, float64)", ORANGE)]
    items = ["architecture", "optimizer", "eval harness", "authority classifier"]
    for x, title, c in cols:
        box(ax, x, 82, 32, 13, title, fc="#FFFFFF", ec=c, fs=6.8, bold=True)
        for i, it in enumerate(items):
            box(ax, x + 2.5, 62 - i * 13, 25, 9.5, it, fc="#F3F4F6", fs=7.2)
        arrow(ax, x + 15, 20.5, 50, 12.5, color=c, lw=1.3)
    box(ax, 33, 4, 34, 11, "bit-identical\nbenchmark output?", fc="#FFF7ED", fs=7.6, bold=True)
    for i, chip in enumerate(["OS: Windows / Linux", "Python: 3.13 / 3.11", "backend swap"]):
        box(ax, 36.5, 90 - i * 12.5, 27, 8.5, chip, fc="#FFFFFF", ec=FAINT, fs=6.8)
    ax.text(50, 0.2, "gradient checks: analytic = finite-difference (float64)",
            ha="center", fontsize=6.4, color=MUTED, style="italic")
    fig.savefig(OUT / "fig3_dualbackend.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_dualbackend.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4
HELD = [(150,3.609),(175,3.568),(200,3.540),(225,3.499),(250,3.502),(275,3.455),
 (300,3.448),(575,3.109),(600,3.061),(625,3.043),(650,3.001),(675,2.967),
 (700,2.932),(1000,2.571),(1025,2.535),(1050,2.535),(1075,2.498),(1100,2.485),
 (1125,2.461),(1150,2.465),(1575,2.252),(1600,2.244),(1625,2.242),(1650,2.225),
 (1675,2.236),(1700,2.205),(1725,2.210),(1750,2.205),(2150,2.116),(2175,2.106),
 (2200,2.109),(2225,2.102),(2250,2.093),(2275,2.085),(2300,2.084),(2500,2.060),
 (2525,2.050),(2550,2.052),(2575,2.056),(2600,2.050),(2625,2.042),(2650,2.044),
 (2675,2.031),(2775,2.017),(2800,2.023),(2825,2.020),(2850,2.006),(2875,2.008),
 (2900,2.009),(2925,2.005),(2950,2.000),(2975,2.002),(3000,1.993),(3025,1.997),
 (3050,1.991),(3075,1.994),(3100,1.991),(3125,1.984),(3150,1.985),(3175,1.982),
 (3200,1.985),(3225,1.988),(3250,1.988),(3275,1.978),(3300,1.974),(3325,1.970),
 (3350,1.967),(3375,1.970),(3825,1.930),(3850,1.930),(3875,1.929),(3900,1.926),
 (3925,1.921),(3950,1.930),(3975,1.919),(4025,1.918),(4050,1.923),(4075,1.922),
 (4100,1.917),(4125,1.916),(4150,1.914),(4175,1.908),(4200,1.908),(4225,1.904),
 (4250,1.900),(4275,1.907),(4300,1.903),(4325,1.908),(4350,1.898),(4375,1.893),
 (4400,1.896),(5000,1.878),(5025,1.861),(5050,1.874),(5075,1.868),(5100,1.873),
 (5125,1.861),(5150,1.868),(5175,1.862),(5200,1.862),(5225,1.861),(6300,1.822),
 (6325,1.824),(6350,1.819),(6375,1.824),(6400,1.821),(6425,1.823),(6450,1.821),
 (6475,1.814),(6500,1.817),(6700,1.811),(6725,1.808),(6750,1.803),(6775,1.805),
 (6800,1.805),(6825,1.803),(6850,1.804),(6875,1.800),(6900,1.800),(6925,1.806),
 (6950,1.801),(6975,1.804),(7000,1.797),(7075,1.799),(7100,1.802),(7125,1.800),
 (7150,1.801),(7175,1.802),(7200,1.797),(7225,1.797),(7250,1.801),(7275,1.795),
 (7300,1.791),(7325,1.795),(7350,1.795),(7375,1.800),(7400,1.791),(7425,1.785),
 (7450,1.788),(7475,1.788),(7500,1.787)]
TRAIN = [(150,3.185),(300,2.446),(700,2.340),(1025,1.938),(1700,1.805),
 (2300,1.560),(2950,1.508),(3375,1.541),(3975,1.289),(4400,1.520),
 (5225,1.264),(6500,1.226),(7000,1.121),(7500,1.254)]


def fig4():
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    hx, hy = zip(*HELD); tx, ty = zip(*TRAIN)
    ax.plot([25, 150], [4.515, 3.609], color=BLUE, lw=1.8)
    ax.plot(hx, hy, color=BLUE, lw=1.8, label="held-out")
    ax.plot(tx, ty, color=ORANGE, lw=1.4, ls=(0, (4, 2)), label="train (sampled)")
    ax.axhline(1.8, color=MUTED, lw=1.0, ls=":")
    ax.text(150, 1.815, "pre-registered SFT gate (1.8)", fontsize=6.6,
            color=MUTED, ha="left", va="bottom")
    for step, bpb, note, off in [(700, 2.932, "2.9: word-shape\nstatistics", (10, 16)),
                                 (2150, 2.116, "2.1: legal\ninvented words", (-30, 26)),
                                 (2675, 2.031, "2.0: indented code,\nregister split", (26, 10))]:
        ax.plot([step], [bpb], "o", ms=4.5, color=BLUE, mfc="white", mew=1.2)
        ax.annotate(note, (step, bpb), textcoords="offset points",
                    xytext=off, fontsize=6.2, color=INK)
    ax.plot([7425], [1.785], "o", ms=4.5, color=BLUE)
    ax.annotate("1.785 @ 55 min\n(gate cleared min. 51)", (7425, 1.785),
                textcoords="offset points", xytext=(-30, 22), fontsize=6.4)
    ax.annotate("later session: best 1.313\n(checkpoint lost; Sec. VII)",
                (7500, 1.313), textcoords="offset points", xytext=(-160, -12),
                fontsize=6.4, color=MUTED, style="italic")
    ax.plot([7150, 7500], [1.34, 1.313], color=MUTED, lw=1.0, ls=(0, (2, 3)))
    ax.set_xlabel("training step (25-step evaluation intervals)")
    ax.set_ylabel("bits per byte")
    ax.set_ylim(1.15, 4.7); ax.set_xlim(0, 7700)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=FAINT, lw=0.5, alpha=0.6)
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    fig.savefig(OUT / "fig4_curve.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig4_curve.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    for f in sorted(OUT.glob("*.pdf")):
        print(f.name, f.stat().st_size, "bytes")
