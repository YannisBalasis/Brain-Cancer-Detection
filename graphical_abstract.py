"""
Graphical Abstract v6 — CMPB
Layout: 2 columns
  Left  (55%): Horizontal bar chart — 9 architectures
  Right (45%): top = Grad-CAM grid (CNN / ViT)
               bottom = Meningioma Paradox F1 bars
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = "#1b3a6b"
ORANGE = "#c8440a"
BLUE   = "#3a72b0"
LGRAY  = "#9ab4d0"
WHITE  = "#ffffff"
BG     = "#f4f7fb"
DARK   = "#1a1a2e"

MODELS = [
    ("Binary CNN",           98.60, NAVY),
    ("MedViT V2 Dual ★",    98.15, ORANGE),
    ("4-Class CNN",          98.01, NAVY),
    ("3-Class CNN",          97.80, NAVY),
    ("1-vs-1 Ensemble",      97.60, NAVY),
    ("Multi-Dual System",    93.41, BLUE),
    ("MedViT V2 Base",       93.03, BLUE),
    ("EfficientNet-B3 Dual", 86.03, LGRAY),
    ("ResNet-50 Dual",       76.85, LGRAY),
]

F1_DATA = {
    "4-Class CNN":    [98.75, 97.58, 99.26, 99.15],
    "MedViT V2 Base": [92.41, 88.02, 97.00, 93.77],
    "MedViT V2 Dual": [98.76, 97.21, 98.75, 97.78],
}
F1_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ── XAI image: Meningioma column only (CNN + ViT) ────────────────────────────
# Original: 3 rows (Original MRI / CNN Grad-CAM / MedViT Grad-CAM)
#           4 cols (Glioma / Meningioma / No Tumor / Pituitary)
xai_full = plt.imread("xai_comparison_cnn_vs_medvit_clean.png")
H_x, W_x = xai_full.shape[:2]
rh = H_x // 3
cw = W_x // 4
# Meningioma column (col 1): CNN row (1) and ViT row (2)
cnn_meni_raw = xai_full[rh + rh // 3 : 2 * rh, cw : 2 * cw]   # 4-Class CNN Grad-CAM
vit_meni_raw  = xai_full[2 * rh      : 3 * rh, cw : 2 * cw]  # MedViT V2 Grad-CAM

def autocrop(img, thr=0.93):
    """Strip white/near-white border rows and columns."""
    gray = np.mean(img[:, :, :3], axis=2)
    mask = gray < thr
    r = np.any(mask, axis=1); c = np.any(mask, axis=0)
    r0, r1 = np.argmax(r), len(r) - np.argmax(r[::-1])
    c0, c1 = np.argmax(c), len(c) - np.argmax(c[::-1])
    return img[r0:r1, c0:c1]

cnn_meni = autocrop(cnn_meni_raw)
vit_meni  = autocrop(vit_meni_raw)

# ── Figure: 9 × 4 inch @ 300 DPI → 2700 × 1200 px ───────────────────────────
fig = plt.figure(figsize=(9, 4), facecolor=WHITE)

# Outer: 2 columns
gs_outer = gridspec.GridSpec(
    1, 2,
    figure=fig,
    width_ratios=[1.2, 1.0],
    wspace=0.28,
    left=0.195, right=0.985, top=0.885, bottom=0.13,
)

ax_bar = fig.add_subplot(gs_outer[0])

# Right column: 2 rows (XAI top, Meningioma bottom)
gs_right = gridspec.GridSpecFromSubplotSpec(
    2, 1,
    subplot_spec=gs_outer[1],
    hspace=0.52,
    height_ratios=[1.0, 1.0],
)
ax_xai  = fig.add_subplot(gs_right[0])
ax_meni = fig.add_subplot(gs_right[1])

# ── Figure-level title ────────────────────────────────────────────────────────
fig.text(
    0.59, 0.975,
    "Grad-CAM: CNN vs. Transformer  |  Meningioma Paradox",
    ha="center", va="top", fontsize=9, color="#555", style="italic",
)


# ═══════════════════════════════════════════════════════════════════════════════
# LEFT — Accuracy Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
names  = [m[0] for m in MODELS]
accs   = [m[1] for m in MODELS]
colors = [m[2] for m in MODELS]
y_pos  = np.arange(len(MODELS))

ax_bar.set_facecolor(BG)
ax_bar.barh(y_pos, accs, color=colors, height=0.70,
            zorder=3, edgecolor=WHITE, linewidth=0.5)
ax_bar.invert_yaxis()

for i, (acc, col) in enumerate(zip(accs, colors)):
    tc = WHITE if col in (NAVY, ORANGE, BLUE) else DARK
    ax_bar.text(acc - 0.25, i, f"{acc:.2f}%",
                va="center", ha="right", fontsize=8,
                fontweight="bold", color=tc, zorder=4)

ax_bar.axvline(97.0, color="#bbb", ls="--", lw=0.9, zorder=2)
ax_bar.set_xlim(70, 100.4)
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(names, fontsize=9.5)
ax_bar.set_xlabel("Accuracy — 503 tumour-positive test samples (%)", fontsize=9, labelpad=4)
ax_bar.set_title("Classification Performance (9 Architectures)",
                 fontsize=11, fontweight="bold", color=DARK, pad=5)
ax_bar.grid(axis="x", color=WHITE, lw=1.3, zorder=1)
ax_bar.tick_params(left=False)
for sp in ax_bar.spines.values():
    sp.set_visible(False)

# Wilcoxon note
ax_bar.annotate(
    "$p_{Bonf}=0.240$",
    xy=(98.15, 1), xytext=(83, 3.2),
    fontsize=8, color=ORANGE, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9,
                    connectionstyle="arc3,rad=-0.25"),
    bbox=dict(boxstyle="round,pad=0.2", fc=WHITE, ec=ORANGE, lw=0.7, alpha=0.95),
)

legend_items = [
    mpatches.Patch(color=NAVY,   label="Top-tier"),
    mpatches.Patch(color=ORANGE, label="MedViT V2 Dual ★ (novel)"),
    mpatches.Patch(color=BLUE,   label="Mid-tier"),
    mpatches.Patch(color=LGRAY,  label="Low-tier"),
]
ax_bar.legend(handles=legend_items, fontsize=8, loc="lower right",
              framealpha=0.93, edgecolor="#ccc", ncol=2,
              handlelength=1.1, handletextpad=0.3, columnspacing=0.7, borderpad=0.4)


# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT TOP — 2 Grad-CAM images side by side (Meningioma: CNN vs ViT)
# ═══════════════════════════════════════════════════════════════════════════════
ax_xai.set_facecolor(BG)
ax_xai.set_xticks([])
ax_xai.set_yticks([])
for sp in ax_xai.spines.values():
    sp.set_visible(False)
ax_xai.set_title("Meningioma — Grad-CAM",
                 fontsize=10, fontweight="bold", color=DARK, pad=5)

# CNN image (left inset) — label OUTSIDE below
ax_c = ax_xai.inset_axes([0.01, 0.24, 0.47, 0.70])
ax_c.imshow(cnn_meni, aspect="auto")
ax_c.set_xticks([])
ax_c.set_yticks([])
for sp in ax_c.spines.values():
    sp.set_color(NAVY); sp.set_linewidth(2.5)

ax_xai.text(0.25, 0.12, "4-Class CNN\n(Focused)", ha="center", va="center",
            fontsize=9, color=NAVY, fontweight="bold",
            transform=ax_xai.transAxes,
            bbox=dict(boxstyle="round,pad=0.22", fc=WHITE, ec=NAVY, lw=1.5))

# ViT image (right inset) — label OUTSIDE below
ax_v = ax_xai.inset_axes([0.52, 0.24, 0.47, 0.70])
ax_v.imshow(vit_meni, aspect="auto")
ax_v.set_xticks([])
ax_v.set_yticks([])
for sp in ax_v.spines.values():
    sp.set_color(ORANGE); sp.set_linewidth(2.5)

ax_xai.text(0.76, 0.12, "MedViT V2\n(Distributed)", ha="center", va="center",
            fontsize=9, color=ORANGE, fontweight="bold",
            transform=ax_xai.transAxes,
            bbox=dict(boxstyle="round,pad=0.22", fc=WHITE, ec=ORANGE, lw=1.5))


# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT BOTTOM — Meningioma Paradox
# ═══════════════════════════════════════════════════════════════════════════════
x = np.arange(len(F1_CLASSES))
w = 0.25
bar_cols = [NAVY, ORANGE, BLUE]

ax_meni.set_facecolor(BG)
for sp in ax_meni.spines.values():
    sp.set_visible(False)

for i, (mname, vals) in enumerate(F1_DATA.items()):
    ax_meni.bar(x + i * w, vals, w, label=mname,
                color=bar_cols[i], alpha=0.90, zorder=3)

# Highlight Meningioma column
ax_meni.add_patch(mpatches.FancyBboxPatch(
    (x[1]-0.07, 82.2), 3*w+0.14, 19.3, boxstyle="square,pad=0",
    facecolor="#fde8d0", lw=0, zorder=1.5, alpha=0.60))
ax_meni.add_patch(mpatches.FancyBboxPatch(
    (x[1]-0.07, 82.2), 3*w+0.14, 19.3, boxstyle="square,pad=0",
    edgecolor=ORANGE, facecolor="none", lw=1.5, zorder=4))

ax_meni.set_ylim(82, 101)
ax_meni.set_xticks(x + w)
ax_meni.set_xticklabels(F1_CLASSES, fontsize=8.5)
ax_meni.set_ylabel("F1 (%)", fontsize=9)
ax_meni.set_title("Meningioma Paradox", fontsize=10.5,
                  fontweight="bold", color=DARK, pad=4)
ax_meni.grid(axis="y", color=WHITE, lw=1.2, zorder=1)
ax_meni.tick_params(bottom=False)

# Compact annotation inside the Meningioma column
ax_meni.text(x[1] + w, 99.5,
             "Lowest F1\nacross all models",
             ha="center", va="top", fontsize=8, color="#a02010",
             fontweight="bold", zorder=5,
             bbox=dict(boxstyle="round,pad=0.2", fc="#fff0ed",
                       ec="#c0392b", lw=0.8, alpha=0.95))

ax_meni.legend(fontsize=7.5, loc="upper right",
               framealpha=0.93, edgecolor="#ccc",
               handlelength=1.0, labelspacing=0.25, borderpad=0.4)


# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig("graphical_abstract.tiff", dpi=300, facecolor=WHITE)
fig.savefig("graphical_abstract.pdf",            facecolor=WHITE)
fig.savefig("graphical_abstract.png",  dpi=150,  facecolor=WHITE)
print("Saved: graphical_abstract.tiff (300 DPI, 2700×1200 px) | .pdf | .png")
