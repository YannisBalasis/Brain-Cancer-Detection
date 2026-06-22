"""
Confusion Matrices — 503-sample tumour-positive subset
Supplementary Figure S2: 4-Class CNN | MedViT V2 Base | MedViT V2 Dual

Matrix: 3 true classes × 3 predicted classes (Glioma / Meningioma / Pituitary)
No Tumor predictions (if any) are tracked separately and reported in caption text.

Run on DGX:
  export CUDA_VISIBLE_DEVICES=0
  python confusion_matrices_503.py
"""

import numpy as np
import tensorflow as tf
import cv2
import inspect
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Config ───────────────────────────────────────────────────────────────────
BASE         = "/storage/data4/up1084631"
DATA_DIR     = f"{BASE}/data_multiclass"
OUT_DIR      = Path("confusion_matrices_503")
OUT_DIR.mkdir(exist_ok=True)
IMG_SIZE     = 224
RANDOM_STATE = 42
BATCH_SIZE   = 32

FOLDER_TO_LABEL  = {"glioma": 0, "meningioma": 1, "no_tumor": 2, "pituitary": 3}
TUMOUR_LABELS    = {0, 1, 3}
CLASS_NAMES      = ["Glioma", "Meningioma", "Pituitary"]
LABEL_TO_IDX_3   = {0: 0, 1: 1, 3: 2}  # 4-class label → 3-class row/col index

# ── Custom objects for MedViT ─────────────────────────────────────────────────
import medvit_v2_architecture
_CUSTOM = {
    name: obj
    for name, obj in inspect.getmembers(medvit_v2_architecture, inspect.isclass)
    if obj.__module__ == "medvit_v2_architecture"
}

MODELS = {
    "4-Class CNN": {
        "path":   f"{BASE}/best_multiclass_4class_model.h5",
        "custom": False,
    },
    "MedViT V2 Base": {
        "path":   f"{BASE}/medvit_v2_experiment_v2/best_medvit_v2_base.keras",
        "custom": True,
    },
    "MedViT V2 Dual": {
        "path":   f"{BASE}/medvit_v2_dual_experiment/best_medvit_v2_dual.keras",
        "custom": True,
    },
}


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading images ...")
X_all, y_all = [], []
for folder, label in sorted(FOLDER_TO_LABEL.items()):
    fp = Path(DATA_DIR) / folder
    imgs = sorted(fp.glob("*.jpg")) + sorted(fp.glob("*.jpeg")) + sorted(fp.glob("*.png"))
    print(f"  {folder}: {len(imgs)}")
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_all.append(img.astype(np.float32) / 255.0)
        y_all.append(label)

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int32)

_, X_test, _, y_test = train_test_split(
    X_all, y_all, test_size=0.10, stratify=y_all, random_state=RANDOM_STATE
)

# Keep only tumour-positive samples
mask   = np.isin(y_test, list(TUMOUR_LABELS))
X_test = X_test[mask]
y_test = y_test[mask]
print(f"\nTest subset: {len(X_test)} tumour-positive samples")
for lbl, name in zip([0, 1, 3], CLASS_NAMES):
    print(f"  {name}: {(y_test == lbl).sum()}")

# True labels in 3-class index
y_true_3 = np.array([LABEL_TO_IDX_3[l] for l in y_test])


# ── Predict + build CM ────────────────────────────────────────────────────────
def predict_model(model, X):
    raw = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    if isinstance(raw, dict):
        raw = max(raw.values(), key=lambda t: t.shape[-1])
    return np.argmax(raw, axis=1)   # 4-class labels: 0,1,2,3


def build_cm_3x3(y_true_3, preds_4class):
    """
    3×3 CM over (Glioma, Meningioma, Pituitary).
    No Tumor predictions (label=2) are counted as misclassifications
    but not assigned to a column — tracked separately.
    Returns (cm_3x3, n_notumor_predictions).
    """
    cm = np.zeros((3, 3), dtype=int)
    n_notumor = 0
    for true_idx, pred_4 in zip(y_true_3, preds_4class):
        if pred_4 == 2:          # No Tumor prediction
            n_notumor += 1
        else:
            pred_idx = LABEL_TO_IDX_3[pred_4]
            cm[true_idx, pred_idx] += 1
    return cm, n_notumor


results = {}
for model_name, cfg in MODELS.items():
    print(f"\n{'─'*50}")
    print(f"Model: {model_name}")
    custom_obj = _CUSTOM if cfg["custom"] else {}
    model = tf.keras.models.load_model(
        cfg["path"], compile=False, custom_objects=custom_obj
    )
    preds = predict_model(model, X_test)
    cm, n_notumor = build_cm_3x3(y_true_3, preds)

    acc = np.diag(cm).sum() / len(y_true_3) * 100
    print(f"  Accuracy (503 subset): {acc:.2f}%")
    print(f"  No Tumor predictions:  {n_notumor}")
    print(f"  CM:\n{cm}")

    results[model_name] = {"cm": cm, "n_notumor": n_notumor, "acc": acc}

    del model
    tf.keras.backend.clear_session()


# ── Plot — one PDF per model ──────────────────────────────────────────────────
CMAP         = sns.color_palette("Blues", as_cmap=True)
class_counts = np.array([(y_true_3 == i).sum() for i in range(3)])

FILE_SLUG = {
    "4-Class CNN":    "cm_4class_cnn_503",
    "MedViT V2 Base": "cm_medvit_base_503",
    "MedViT V2 Dual": "cm_medvit_dual_503",
}

for model_name, res in results.items():
    cm        = res["cm"]
    n_notumor = res["n_notumor"]
    acc       = res["acc"]
    cm_pct    = cm.astype(float) / class_counts[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        cm_pct,
        annot=False,
        cmap=CMAP,
        vmin=0, vmax=100,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        linecolor="#BDC3C7",
        ax=ax,
        cbar_kws={"label": "Row-normalised (%)"},
    )

    for i in range(3):
        for j in range(3):
            count = cm[i, j]
            pct   = cm_pct[i, j]
            color = "white" if pct > 55 else "#2C3E50"
            ax.text(j + 0.5, i + 0.38, f"{count}",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)
            ax.text(j + 0.5, i + 0.65, f"({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=10, color=color)

    for i in range(3):
        ax.add_patch(plt.Rectangle(
            (i, i), 1, 1, fill=False,
            edgecolor="#27AE60", lw=2.5, zorder=3
        ))

    notumor_str = f" [†{n_notumor} pred. as No Tumor]" if n_notumor > 0 else ""
    ax.set_title(
        f"{model_name} — Accuracy: {acc:.2f}%{notumor_str}\n"
        "503-sample tumour-positive subset",
        fontsize=11, fontweight="bold", color="#2C3E50", pad=12
    )
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
    ax.set_ylabel("True Class",      fontsize=11, labelpad=8)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, rotation=0,  fontsize=11)

    if n_notumor > 0:
        fig.text(
            0.5, -0.02,
            "† Samples predicted as No Tumor are misclassifications not shown in any column.",
            ha="center", fontsize=8, color="#7F8C8D", style="italic"
        )

    plt.tight_layout()
    out = OUT_DIR / f"{FILE_SLUG[model_name]}.pdf"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")
