#!/usr/bin/env python3
"""Generate README/model-card charts from the real 0c campaign data.

Three charts x light/dark, static SVG (no hover layer in READMEs).
Palette: validated reference instance (dataviz skill) — single-series
charts use categorical slot 1; text wears text tokens, never series color.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ANALYSIS = json.loads(Path(
    "/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/"
    "runs/0c-campaign/eps-analysis.json").read_text())

MODES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  grid="#e7e6e2", series="#2a78d6"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7",
                 grid="#33322f", series="#3987e5"),
}


def style(ax, m):
    ax.set_facecolor(m["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(m["grid"])
    ax.tick_params(colors=m["ink2"], labelsize=9)
    ax.yaxis.grid(True, color=m["grid"], linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def fig_ax(m, w=7.2, h=4.0):
    f, ax = plt.subplots(figsize=(w, h), dpi=100)
    f.patch.set_facecolor(m["surface"])
    style(ax, m)
    return f, ax


def save(f, name, mode):
    f.tight_layout(pad=1.2)
    f.savefig(HERE / f"{name}-{mode}.svg", facecolor=f.get_facecolor(),
              bbox_inches="tight")
    f2 = None
    plt.close(f)


def eps_ladder(mode, m):
    ks = [2, 3, 4, 5]
    eps = [ANALYSIS["mean_eps_per_k"][str(k)] for k in ks]
    f, ax = fig_ax(m)
    ax.set_yscale("log")
    ax.plot(ks, eps, color=m["series"], linewidth=2, zorder=3)
    ax.plot(ks, eps, "o", color=m["series"], markersize=9, zorder=4,
            markeredgecolor=m["surface"], markeredgewidth=2)
    for k, e in zip(ks, eps):
        ax.annotate(f"{e:.4f}", (k, e), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10,
                    color=m["ink"], fontweight="bold")
    for i in range(3):
        ratio = eps[i] / eps[i + 1]
        ax.annotate(f"÷{ratio:.1f}", (ks[i] + 0.5,
                    (eps[i] * eps[i + 1]) ** 0.5),
                    ha="center", fontsize=9, color=m["ink2"])
    ax.set_xticks(ks)
    ax.set_xticklabels([f"K{k}" for k in ks], fontsize=11, color=m["ink"])
    ax.set_ylim(8e-4, 0.2)
    ax.set_title("Per-expert encode error vs bit-width — GLM-5.2 proxy, "
                 "one shared calibration",
                 fontsize=12, color=m["ink"], loc="left", pad=14)
    ax.text(0, 1.015, "mean relative round-trip MSE (log scale) — each +1 bit ≈ 3.8× lower error",
            transform=ax.transAxes, fontsize=9.5, color=m["ink2"])
    save(f, "eps-ladder", mode)


def benefit_concentration(mode, m):
    sys.path.insert(0, "/home/mbelleau/protensors-work/vllm-voipmonitor/"
                       "research/fungible-quant/tools")
    import fq_eps
    eps, phi, layers = fq_eps.load_eps(
        Path("/home/mbelleau/fq-0c"), [3, 4])
    delta = eps[3] - eps[4]
    phi_n = phi / np.maximum(phi.sum(axis=1, keepdims=True), 1)
    benefit = delta * phi_n
    f, ax = fig_ax(m)
    curves = []
    for i in range(benefit.shape[0]):
        b = np.sort(benefit[i])[::-1]
        c = np.cumsum(b) / b.sum()
        curves.append(c)
    curves = np.array(curves)
    x = np.arange(1, curves.shape[1] + 1)
    ax.fill_between(x, curves.min(0), curves.max(0),
                    color=m["series"], alpha=0.18, linewidth=0)
    med = np.median(curves, axis=0)
    ax.plot(x, med, color=m["series"], linewidth=2)
    k16 = float(med[15])
    ax.plot([16], [k16], "o", color=m["series"], markersize=9,
            markeredgecolor=m["surface"], markeredgewidth=2)
    ax.annotate(f"top 16 experts →\n{k16*100:.0f}% of benefit",
                (16, k16), textcoords="offset points", xytext=(16, -34),
                fontsize=10, color=m["ink"], fontweight="bold")
    ax.set_xlim(1, 256)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 64, 128, 192, 256])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("experts, ranked by benefit (of 256)", fontsize=10,
                  color=m["ink2"])
    ax.set_title("Why mixed-K wins: upgrade benefit concentrates in few experts",
                 fontsize=12, color=m["ink"], loc="left", pad=14)
    ax.text(0, 1.015, "cumulative share of K3→K4 error-reduction × routing mass "
                      "(median across layers, band = min–max)",
            transform=ax.transAxes, fontsize=9.5, color=m["ink2"])
    save(f, "benefit-concentration", mode)


def allocation(mode, m):
    solve = next(s for s in ANALYSIS["solves"]
                 if abs(s["budget_frac"] - 0.42) < 1e-9)
    items = sorted(solve["n_k4_per_layer"].items(), key=lambda kv: int(kv[0]))
    layers = [int(k) for k, _ in items]
    counts = [v for _, v in items]
    uniform = round(0.42 * 256)
    f, ax = fig_ax(m)
    ax.bar([str(l) for l in layers], counts, color=m["series"],
           width=0.62, zorder=3)
    ax.axhline(uniform, color=m["ink2"], linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.55, zorder=2)
    ax.annotate(f"uniform budget ({uniform}/layer)", (0.02, uniform),
                xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 6), ha="left",
                fontsize=9, color=m["ink2"])
    for i, v in enumerate(counts):
        ax.annotate(str(v), (i, v), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9.5, color=m["ink"], zorder=5)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_xlabel("MoE layer", fontsize=10, color=m["ink2"])
    ax.set_title("The solve allocates the K4 budget unevenly — "
                 "layers earn their bits",
                 fontsize=12, color=m["ink"], loc="left", pad=14)
    ax.text(0, 1.015, "K4 experts per layer at a fixed global budget "
                      "(42% of experts), measured benefit-ranked",
            transform=ax.transAxes, fontsize=9.5, color=m["ink2"])
    save(f, "k4-allocation", mode)


for mode, m in MODES.items():
    eps_ladder(mode, m)
    benefit_concentration(mode, m)
    allocation(mode, m)
print("charts written:", sorted(p.name for p in HERE.glob("*.svg")))
