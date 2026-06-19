"""
Meningioma Paradox — per-class F1 grouped bar chart.
Run on LAPTOP — reads hardcoded values from evaluation results.
Output: meningioma_paradox_chart.pdf / .png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Per-class F1 data (from full 703-sample test set) ─────────────────────
# Source: thesis Table 9.3 (4-Class CNN) + eval_results JSON (MedViT V2)
CLASSES = ["Glioma", "Meningioma", "No Tumour", "Pituitary"]

models = {
    "4-Class CNN\n(1.28 M)":    [98.75, 97.58, 99.26, 99.15],
    "MedViT V2 Base\n(49.3 M)": [92.41, 88.02, 97.00, 93.77],
    "MedViT V2 Dual\n(49.9 M)": [98.76, 97.21, 98.75, 97.78],
}

N_MODELS  = len(models)
N_CLASSES = len(CLASSES)
x         = np.arange(N_CLASSES)
bar_w     = 0.24
offsets   = np.linspace(-(N_MODELS - 1) / 2, (N_MODELS - 1) / 2, N_MODELS) * bar_w

# Colours: Meningioma column gets a distinct hatch to highlight the paradox
BASE_COLORS  = ["#4c72b0", "#55a868", "#c44e52"]   # blue, green, red
MENING_IDX   = CLASSES.index("Meningioma")          # column 1

fig, ax = plt.subplots(figsize=(9, 5))

bars_for_legend = []
for m_i, (model_name, f1s) in enumerate(models.items()):
    for c_i, (cls, f1) in enumerate(zip(CLASSES, f1s)):
        is_mening = (c_i == MENING_IDX)
        bar = ax.bar(
            x[c_i] + offsets[m_i], f1,
            width=bar_w,
            color=BASE_COLORS[m_i],
            alpha=0.55 if is_mening else 0.82,
            hatch="////" if is_mening else "",
            edgecolor="black" if is_mening else BASE_COLORS[m_i],
            linewidth=1.2 if is_mening else 0.4,
            zorder=3
        )
        # value label on top of each bar
        ax.text(
            x[c_i] + offsets[m_i], f1 + 0.08,
            f"{f1:.1f}",
            ha="center", va="bottom",
            fontsize=7.2, color="#222",
            rotation=90 if f1 < 95 else 0
        )
    # legend proxy
    bars_for_legend.append(mpatches.Patch(
        color=BASE_COLORS[m_i], alpha=0.85,
        label=model_name.replace("\n", "  ")
    ))

# ── Highlight Meningioma column background ─────────────────────────────────
ax.axvspan(MENING_IDX - 0.45, MENING_IDX + 0.45,
           color="#ffe0b2", alpha=0.35, zorder=1,
           label="_nolegend_")

# ── Annotate the paradox ───────────────────────────────────────────────────
ax.annotate(
    "Meningioma Paradox:\ncorrect localisation,\nlowest F1 across all models",
    xy=(MENING_IDX, 87.5), xytext=(MENING_IDX + 0.72, 89.5),
    fontsize=8, color="#8b0000",
    arrowprops=dict(arrowstyle="->", color="#8b0000", lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", fc="#fff3f3", ec="#8b0000", lw=1)
)

# ── Axes formatting ────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, fontsize=10)
ax.set_ylabel("Per-class F1 Score (%)", fontsize=10)
ax.set_ylim(84, 101.5)
ax.set_xlim(-0.55, N_CLASSES - 0.45)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# hatched legend entry for Meningioma column
mening_patch = mpatches.Patch(
    facecolor="white", edgecolor="black",
    hatch="////", label="Meningioma column\n(paradox highlight)"
)
ax.legend(
    handles=bars_for_legend + [mening_patch],
    loc="lower right", fontsize=8, framealpha=0.9,
    ncol=1
)

ax.set_title(
    "Per-class F1 Scores — Meningioma Paradox\n"
    "Correct spatial localisation (XAI evidence) yet lowest classification F1",
    fontsize=10, pad=8
)

plt.tight_layout()
fig.savefig("meningioma_paradox_chart.pdf", dpi=300, bbox_inches="tight")
fig.savefig("meningioma_paradox_chart.png", dpi=300, bbox_inches="tight")
print("Saved: meningioma_paradox_chart.pdf / .png")
