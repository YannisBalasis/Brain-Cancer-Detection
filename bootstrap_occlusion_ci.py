"""
Bootstrap Confidence Intervals — Occlusion Sensitivity Analysis
Quantifies the Meningioma Paradox across all multiclass models.

Design:
  - Reproduces the 70/20/10 stratified split (random_state=42) used in training
  - Runs sliding-window occlusion (28x28 px, stride 7) on ALL correctly-classified
    tumour-positive test images (503-sample common subset)
  - Max confidence drop per image = max over all patch positions of
    (baseline_conf - occluded_conf) / baseline_conf * 100
  - Bootstrap resamples B=10,000 -> 95% CI on mean max-drop per class per model

Output: occlusion_bootstrap_results.json  +  printed table
Run on DGX  (~45-90 min total)
"""

import numpy as np
import tensorflow as tf
import cv2
import json
import inspect
import time
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────────────
BASE          = "/storage/data4/up1084631"
DATA_DIR      = f"{BASE}/data_multiclass"
OUT_FILE      = "occlusion_bootstrap_results.json"

IMG_SIZE      = 224
PATCH_SIZE    = 28
STRIDE        = 7
N_BOOTSTRAP   = 10_000
CI_LEVEL      = 95
RANDOM_STATE  = 42
PATCH_BATCH   = 128    # occluded patches per GPU forward pass

# ── Model registry ──────────────────────────────────────────────────────────
# patch_batch: reduce for memory-heavy models (MedViT DiNA layer)
MODELS_CFG = {
    "MedViT V2 Base": {
        "path":        f"{BASE}/medvit_v2_experiment_v2/best_medvit_v2_base.keras",
        "classes":     {"Glioma": 0, "Meningioma": 1, "Pituitary": 3},
        "custom":      True,
        "patch_batch": 32,
    },
    "MedViT V2 Dual": {
        "path":        f"{BASE}/medvit_v2_dual_experiment/best_medvit_v2_dual.keras",
        "classes":     {"Glioma": 0, "Meningioma": 1, "Pituitary": 3},
        "custom":      True,
        "patch_batch": 32,
    },
}

# ── Results from previous runs (hardcoded) ───────────────────────────────────
all_results = {
    "4-Class CNN": {
        "Glioma":     {"mean": 5.8,  "std": 15.3, "ci_lower": 3.6,  "ci_upper": 8.3,  "n": 160, "n_excluded_misclassified": 2,  "ci_level": 95},
        "Meningioma": {"mean": 64.1, "std": 37.2, "ci_lower": 58.1, "ci_upper": 69.9, "n": 156, "n_excluded_misclassified": 9,  "ci_level": 95},
        "Pituitary":  {"mean": 1.3,  "std": 8.7,  "ci_lower": 0.3,  "ci_upper": 2.8,  "n": 176, "n_excluded_misclassified": 0,  "ci_level": 95},
    },
    "3-Class CNN": {
        "Glioma":     {"mean": 11.0, "std": 23.2, "ci_lower": 7.6,  "ci_upper": 14.7, "n": 161, "n_excluded_misclassified": 1,  "ci_level": 95},
        "Meningioma": {"mean": 45.8, "std": 40.4, "ci_lower": 39.5, "ci_upper": 52.0, "n": 162, "n_excluded_misclassified": 3,  "ci_level": 95},
        "Pituitary":  {"mean": 6.0,  "std": 11.3, "ci_lower": 4.4,  "ci_upper": 7.8,  "n": 175, "n_excluded_misclassified": 1,  "ci_level": 95},
    },
    "Multi-Dual": {
        "Glioma":     {"mean": 8.8,  "std": 25.0, "ci_lower": 5.2,  "ci_upper": 13.0, "n": 156, "n_excluded_misclassified": 6,  "ci_level": 95},
        "Meningioma": {"mean": 51.7, "std": 41.2, "ci_lower": 45.0, "ci_upper": 58.3, "n": 147, "n_excluded_misclassified": 18, "ci_level": 95},
        "Pituitary":  {"mean": 1.8,  "std": 10.9, "ci_lower": 0.4,  "ci_upper": 3.6,  "n": 176, "n_excluded_misclassified": 0,  "ci_level": 95},
    },
}

FOLDER_TO_LABEL = {"glioma": 0, "meningioma": 1, "no_tumor": 2, "pituitary": 3}
LABEL_TO_DISPLAY = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
TUMOUR_LABELS = {0, 1, 3}   # exclude No Tumor from 503-sample subset


# ── Load custom objects for MedViT ──────────────────────────────────────────
import medvit_v2_architecture
_CUSTOM_OBJECTS = {
    name: obj
    for name, obj in inspect.getmembers(medvit_v2_architecture, inspect.isclass)
    if obj.__module__ == "medvit_v2_architecture"
}
print(f"Registered {len(_CUSTOM_OBJECTS)} MedViT custom classes.")


# ── Load all images & reproduce frozen test split ────────────────────────────
print("\nLoading all images from data_multiclass ...")
t0 = time.time()
X_all, y_all = [], []
for folder, label in sorted(FOLDER_TO_LABEL.items()):   # sorted -> reproducible order
    folder_path = Path(DATA_DIR) / folder
    img_files = sorted(
        list(folder_path.glob("*.jpg")) +
        list(folder_path.glob("*.jpeg")) +
        list(folder_path.glob("*.png"))
    )
    print(f"  {folder}: {len(img_files)} images")
    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_all.append(img.astype(np.float32) / 255.0)
        y_all.append(label)

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int32)
print(f"Loaded {len(X_all)} images in {time.time()-t0:.1f}s")

# Reproduce split: 90% trainval / 10% frozen test (same random_state as training)
_, X_test, _, y_test = train_test_split(
    X_all, y_all,
    test_size=0.10,
    stratify=y_all,
    random_state=RANDOM_STATE,
)
# Keep only tumour-positive samples (503-sample common subset)
tumour_mask = np.isin(y_test, list(TUMOUR_LABELS))
X_test = X_test[tumour_mask]
y_test = y_test[tumour_mask]
print(f"Frozen test set (tumour-positive): {len(X_test)} images")
for lbl in sorted(TUMOUR_LABELS):
    print(f"  {LABEL_TO_DISPLAY[lbl]}: {(y_test == lbl).sum()}")


# ── Occlusion helpers ────────────────────────────────────────────────────────
def _build_occluded_batch(img: np.ndarray) -> tuple:
    """
    Returns (positions, batch) where:
      positions: list of (y, x) top-left corners
      batch: (N, H, W, 3) float32 array of occluded images
    """
    H, W = img.shape[:2]
    positions, batch = [], []
    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            occluded = img.copy()
            occluded[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = 0.0
            positions.append((y, x))
            batch.append(occluded)
    return positions, np.array(batch, dtype=np.float32)


def _extract_probs(raw_output) -> np.ndarray:
    """Handle tensor, dict, or list model outputs -> 2-D numpy (batch, classes)."""
    if isinstance(raw_output, dict):
        # Multi-output model: take the head with the most output neurons
        raw_output = max(raw_output.values(), key=lambda t: t.shape[-1])
    elif isinstance(raw_output, (list, tuple)):
        raw_output = max(raw_output, key=lambda t: t.shape[-1])
    return raw_output.numpy()


def max_confidence_drop(model, img: np.ndarray, class_idx: int, patch_batch: int = PATCH_BATCH) -> tuple[float, float]:
    """
    Returns (baseline_conf, max_drop_pct) for one image.
    max_drop_pct = max over all positions of (baseline - occluded) / baseline * 100
    Returns (baseline_conf, -1) if model predicts wrong class (excluded from bootstrap).
    """
    baseline_pred = _extract_probs(model(img[np.newaxis], training=False))[0]
    baseline_conf = float(baseline_pred[class_idx])

    # Exclude incorrectly classified images
    if np.argmax(baseline_pred) != class_idx:
        return baseline_conf, -1.0

    _, occ_batch = _build_occluded_batch(img)
    n_patches = len(occ_batch)

    # Run in sub-batches to control GPU memory
    all_confs = []
    for start in range(0, n_patches, PATCH_BATCH):
        chunk = occ_batch[start:start + PATCH_BATCH]
        preds = _extract_probs(model(chunk, training=False))
        all_confs.extend(preds[:, class_idx])

    min_occ_conf = min(all_confs)
    drop_pct = (baseline_conf - min_occ_conf) / baseline_conf * 100.0
    return baseline_conf, drop_pct


# ── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_mean_ci(data: np.ndarray, B: int = N_BOOTSTRAP, ci: float = CI_LEVEL):
    rng = np.random.default_rng(seed=0)
    boot_means = np.array([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(B)
    ])
    alpha = (100 - ci) / 2
    lower = float(np.percentile(boot_means, alpha))
    upper = float(np.percentile(boot_means, 100 - alpha))
    return {
        "mean":  float(data.mean()),
        "std":   float(data.std()),
        "ci_lower": lower,
        "ci_upper": upper,
        "n": int(len(data)),
        "ci_level": ci,
    }


# ── Main loop (continues from hardcoded results above) ────────────────────────
for model_name, cfg in MODELS_CFG.items():
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Load model
    custom_obj = _CUSTOM_OBJECTS if cfg["custom"] else {}
    model = tf.keras.models.load_model(
        cfg["path"], compile=False, custom_objects=custom_obj
    )
    model_results = {}

    for class_name, class_idx in cfg["classes"].items():
        # Convert display name to numeric label
        label = {"Glioma": 0, "Meningioma": 1, "Pituitary": 3}[class_name]
        class_imgs = X_test[y_test == label]
        print(f"\n  Class: {class_name}  ({len(class_imgs)} images)")

        drops = []
        n_excluded = 0
        t_class = time.time()

        for i, img in enumerate(class_imgs):
            baseline_conf, drop = max_confidence_drop(model, img, class_idx,
                                                   patch_batch=cfg.get("patch_batch", PATCH_BATCH))
            if drop < 0:        # misclassified — exclude
                n_excluded += 1
            else:
                drops.append(drop)
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t_class
                print(f"    [{i+1}/{len(class_imgs)}]  "
                      f"running mean drop: {np.mean(drops):.1f}%  "
                      f"({elapsed:.0f}s elapsed)")

        drops = np.array(drops)
        stats = bootstrap_mean_ci(drops)
        stats["n_excluded_misclassified"] = n_excluded
        model_results[class_name] = stats

        print(f"  -> n={stats['n']}, excluded={n_excluded}")
        print(f"  -> mean drop = {stats['mean']:.1f}% ± {stats['std']:.1f}%  "
              f"95% CI [{stats['ci_lower']:.1f}, {stats['ci_upper']:.1f}]")

    all_results[model_name] = model_results

    # Free GPU memory before loading next model
    del model
    tf.keras.backend.clear_session()


# ── Save results ─────────────────────────────────────────────────────────────
with open(OUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {OUT_FILE}")


# ── Print summary table ───────────────────────────────────────────────────────
print("\n" + "="*80)
print("OCCLUSION CONFIDENCE DROP — BOOTSTRAP 95% CI SUMMARY")
print("="*80)
header = f"{'Model':<20} {'Class':<12} {'n':>5} {'Mean%':>7} {'Std':>6} "
header += f"{'CI_lo':>7} {'CI_hi':>7}"
print(header)
print("-"*80)
for model_name, model_results in all_results.items():
    for class_name, s in model_results.items():
        row = (f"{model_name:<20} {class_name:<12} {s['n']:>5} "
               f"{s['mean']:>7.1f} {s['std']:>6.1f} "
               f"{s['ci_lower']:>7.1f} {s['ci_upper']:>7.1f}")
        print(row)
    print()

print(f"\nBootstrap B={N_BOOTSTRAP}, CI={CI_LEVEL}%")
print("Misclassified images excluded from each class/model combination.")
