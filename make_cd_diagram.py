"""
Publication-quality Critical Difference Diagram (Demšar 2006 style).
Best models labeled on the LEFT, worst on the RIGHT — no overlaps.
Run on LAPTOP.   Output: statistical_analysis_results/cd_diagram_pub.pdf/.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Data ───────────────────────────────────────────────────────────────────
with open("statistical_analysis_results/extended_statistical_results.json") as f:
    data = json.load(f)

avg_ranks = data["average_ranks"]
cd        = data["critical_difference"]   # 0.536
chi2      = data["friedman"]["statistic"] # 164.45
n_samples = 503

models = sorted(avg_ranks.items(), key=lambda x: x[1])   # best first
names  = [m[0] for m in models]
ranks  = np.array([m[1] for m in models])

# ── Nemenyi cliques ────────────────────────────────────────────────────────
def find_cliques(ranks, cd):
    raw = []
    for i in range(len(ranks)):
        j = i + 1
        while j < len(ranks) and (ranks[j] - ranks[i]) < cd:
            j += 1
        if j - i > 1:
            raw.append((i, j - 1))
    def contained(c, others):
        return any(o != c and o[0] <= c[0] and o[1] >= c[1] for o in others)
    return [c for c in raw if not contained(c, raw)]

cliques = find_cliques(ranks, cd)
print("Cliques:")
for ci, cj in cliques:
    print(f"  {names[ci]} → {names[cj]}  Δ={ranks[cj]-ranks[ci]:.3f}")

# ── Layout parameters ──────────────────────────────────────────────────────
N        = len(models)                      # 9
N_LEFT   = (N + 1) // 2                    # 5  (best models)
N_RIGHT  = N - N_LEFT                      # 4  (worst models)
left_idx = list(range(N_LEFT))             # [0,1,2,3,4]
right_idx= list(range(N_LEFT, N))          # [5,6,7,8]

fig_w, fig_h = 11, 5.2
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")

# Coordinate space: x in data-rank units, y in [0, 1]
RANK_LO  = 4.60    # left edge of axis
RANK_HI  = 5.45    # right edge of axis
ax.set_xlim(RANK_LO - 0.45, RANK_HI + 0.45)   # extra space for labels
ax.set_ylim(0.0, 1.0)

AXIS_Y      = 0.42   # rank axis line
LABEL_Y_TOP = 0.88   # y of topmost label
LABEL_Y_GAP = 0.095  # vertical gap between labels
BAR_Y_BASE  = 0.12   # lowest clique bar
BAR_STEP    = 0.09
TICK_H      = 0.022

# ── Rank axis ──────────────────────────────────────────────────────────────
ax.plot([RANK_LO, RANK_HI], [AXIS_Y, AXIS_Y],
        "-", color="#222", lw=1.6, solid_capstyle="round")

for tv in np.arange(4.75, 5.45, 0.25):
    ax.plot([tv, tv], [AXIS_Y - TICK_H, AXIS_Y + TICK_H],
            "-", color="#555", lw=0.9)
    ax.text(tv, AXIS_Y - TICK_H - 0.018, f"{tv:.2f}",
            ha="center", va="top", fontsize=7.5, color="#555")

ax.text((RANK_LO + RANK_HI) / 2, AXIS_Y - 0.12,
        "Average Rank",
        ha="center", va="top", fontsize=8, color="#666", style="italic")

# ── LEFT labels (best models) ──────────────────────────────────────────────
# Stacked top-to-bottom on the left; connected by an L-shaped line
LEFT_X = RANK_LO - 0.06   # x-position of right edge of left labels

for i, idx in enumerate(left_idx):
    label_y = LABEL_Y_TOP - i * LABEL_Y_GAP
    rank    = float(ranks[idx])

    # dot on axis
    ax.plot(rank, AXIS_Y, "o", color="#222", ms=5.5, zorder=5)

    # vertical line: axis → label height
    ax.plot([rank, rank], [AXIS_Y + TICK_H, label_y],
            "-", color="#bbb", lw=0.8, zorder=1)

    # horizontal line: rank → left edge
    ax.plot([rank, LEFT_X], [label_y, label_y],
            "-", color="#bbb", lw=0.8, zorder=1)

    # label text (right-aligned to LEFT_X)
    ax.text(LEFT_X - 0.015, label_y, names[idx],
            ha="right", va="center", fontsize=8.5, color="#111")
    ax.text(LEFT_X - 0.015, label_y - 0.040, f"rank {rank:.3f}",
            ha="right", va="center", fontsize=6.5, color="#777")

# ── RIGHT labels (worst models) ────────────────────────────────────────────
RIGHT_X = RANK_HI + 0.06

for i, idx in enumerate(right_idx):
    label_y = LABEL_Y_TOP - i * LABEL_Y_GAP
    rank    = float(ranks[idx])

    ax.plot(rank, AXIS_Y, "o", color="#222", ms=5.5, zorder=5)
    ax.plot([rank, rank], [AXIS_Y + TICK_H, label_y],
            "-", color="#bbb", lw=0.8, zorder=1)
    ax.plot([rank, RIGHT_X], [label_y, label_y],
            "-", color="#bbb", lw=0.8, zorder=1)
    ax.text(RIGHT_X + 0.015, label_y, names[idx],
            ha="left", va="center", fontsize=8.5, color="#111")
    ax.text(RIGHT_X + 0.015, label_y - 0.040, f"rank {rank:.3f}",
            ha="left", va="center", fontsize=6.5, color="#777")

# ── Clique bars (below axis) ───────────────────────────────────────────────
COLORS = ["#1a6faf", "#2e8b57"]

n_cliques = len(cliques)
for bar_i, (ci, cj) in enumerate(cliques):
    y   = BAR_Y_BASE + (n_cliques - 1 - bar_i) * BAR_STEP
    x0  = float(ranks[ci])
    x1  = float(ranks[cj])
    clr = COLORS[bar_i % len(COLORS)]
    ax.plot([x0, x1], [y, y], "-", color=clr, lw=5.5,
            alpha=0.78, solid_capstyle="butt", zorder=3)
    for xv in [x0, x1]:
        ax.plot([xv, xv], [y - 0.016, y + 0.016],
                "-", color=clr, lw=1.5, zorder=4)

ax.text(RANK_HI + 0.04, BAR_Y_BASE - 0.05,
        "Bars: groups not significantly\ndifferent by Nemenyi (α = 0.05)",
        ha="left", va="top", fontsize=7, color="#666", style="italic",
        linespacing=1.4)

# ── CD reference bar ───────────────────────────────────────────────────────
CD_Y  = 0.96
CD_X0 = float(ranks[0])   # anchored to best model (Binary CNN)
CD_X1 = CD_X0 + cd

ax.annotate("", xy=(CD_X1, CD_Y), xytext=(CD_X0, CD_Y),
            arrowprops=dict(arrowstyle="<->", color="#cc0000",
                            lw=1.7, mutation_scale=10))
for xv in [CD_X0, CD_X1]:
    ax.plot([xv, xv], [CD_Y - 0.018, CD_Y + 0.018],
            "-", color="#cc0000", lw=1.3)
ax.text((CD_X0 + CD_X1) / 2, CD_Y + 0.040,
        f"CD = {cd:.3f}", ha="center", va="bottom",
        fontsize=9, color="#cc0000", fontweight="bold")

# ── Title ──────────────────────────────────────────────────────────────────
ax.text((RANK_LO + RANK_HI) / 2, 1.005,
        rf"Critical Difference Diagram  —  9 Models  |  "
        rf"Friedman $\chi^2$ = {chi2:.2f},  $p < 0.001$,  $n = {n_samples}$",
        ha="center", va="top", fontsize=9.5, color="#111")

plt.tight_layout(pad=0.2)
out = Path("statistical_analysis_results")
fig.savefig(out / "cd_diagram_pub.pdf", dpi=300, bbox_inches="tight")
fig.savefig(out / "cd_diagram_pub.png", dpi=300, bbox_inches="tight")
print("Done →", out / "cd_diagram_pub.png")
