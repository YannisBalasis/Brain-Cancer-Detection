"""
XAI Comparison Figure: 4-Class CNN vs MedViT V2 Grad-CAM side-by-side.
Run on DGX — needs model files and test data.

Layout (3 rows × 4 columns):
  Row 0: Original MRI
  Row 1: 4-Class CNN Grad-CAM  (target layer: conv2d_7)
  Row 2: MedViT V2 Grad-CAM   (target layer: s4_l1_gfp0)
  Columns: Glioma | Meningioma | No Tumor | Pituitary

Output: xai_comparison_cnn_vs_medvit.pdf  (+ .png)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

# ── Paths ── adjust if your experiment folder has a different timestamp ─────
BASE         = "/storage/data4/up1084631"
CNN_MODEL    = f"{BASE}/best_multiclass_4class_model.h5"
MEDVIT_MODEL = f"{BASE}/medvit_v2_experiment_v2/best_medvit_v2_base.keras"

# Image data directory (class subfolders)
DATA_DIR     = f"{BASE}/data_multiclass"

# ── Class definitions ──────────────────────────────────────────────────────
# 4-Class CNN class order (matches training)
CNN_CLASSES    = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CNN_CLASS_IDX  = {c: i for i, c in enumerate(CNN_CLASSES)}

# MedViT V2 class order (alphabetical, as trained)
MEDVIT_CLASSES   = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
MEDVIT_CLASS_IDX = {c: i for i, c in enumerate(["glioma","meningioma","no_tumor","pituitary"])}
# mapping: display label → MedViT index
MEDVIT_IDX = {"Glioma": 0, "Meningioma": 1, "No Tumor": 2, "Pituitary": 3}

DISPLAY_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ── Grad-CAM implementation ────────────────────────────────────────────────
def get_gradcam(model, img_array, target_layer_name, class_idx):
    """
    img_array: (1, H, W, 3), float32 in [0,1].
    Returns: heatmap (H, W) normalised to [0,1].
    Supports both Sequential and Functional models (Keras 3 compatible).
    """
    layer_names = [l.name for l in model.layers]
    assert target_layer_name in layer_names, (
        f"Layer '{target_layer_name}' not found.\n"
        f"Available: {[n for n in layer_names if 'conv' in n.lower()][:15]}"
    )

    img_tensor = tf.constant(img_array, dtype=tf.float32)
    conv_out   = None

    try:
        # Functional model path: encoder → watch → decoder
        target_kt = model.get_layer(target_layer_name).output
        encoder   = tf.keras.Model(inputs=model.inputs,  outputs=target_kt)
        decoder   = tf.keras.Model(inputs=target_kt,     outputs=model.outputs[0])
        with tf.GradientTape() as tape:
            conv_out = encoder(img_tensor, training=False)
            tape.watch(conv_out)
            preds = decoder(conv_out, training=False)
            loss  = preds[:, class_idx]
    except Exception:
        # Sequential / fallback: run layer-by-layer so we can watch mid-stream
        with tf.GradientTape() as tape:
            x = img_tensor
            for layer in model.layers:
                x = layer(x, training=False)
                if layer.name == target_layer_name:
                    conv_out = x
                    tape.watch(conv_out)
            preds = x
            loss  = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        print(f"  Warning: gradients are None for layer '{target_layer_name}' — returning blank heatmap")
        return np.zeros(img_array.shape[1:3])

    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam     = tf.reduce_sum(weights * conv_out, axis=-1)[0]
    cam     = tf.nn.relu(cam).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def overlay_cam(img, cam, alpha=0.45):
    """
    img: (H, W, 3) uint8 or float [0,1].
    cam: (h, w) float [0,1].
    Returns: (H, W, 3) uint8.
    """
    import cv2
    H, W = img.shape[:2]
    if img.max() <= 1.0:
        img_u8 = (img * 255).astype(np.uint8)
    else:
        img_u8 = img.astype(np.uint8)

    cam_resized = cv2.resize(cam, (W, H))
    heatmap     = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended     = cv2.addWeighted(img_u8, 1 - alpha, heatmap_rgb, alpha, 0)
    return blended


# ── Load models ────────────────────────────────────────────────────────────
print("Loading 4-Class CNN …")
cnn_model = tf.keras.models.load_model(CNN_MODEL)
print(f"  CNN loaded: {cnn_model.input_shape}")

print("Loading MedViT V2 …")
import medvit_v2_architecture
import inspect
_custom_objects = {
    name: obj
    for name, obj in inspect.getmembers(medvit_v2_architecture, inspect.isclass)
    if obj.__module__ == "medvit_v2_architecture"
}
print(f"  Registering {len(_custom_objects)} custom classes: {list(_custom_objects.keys())}")
medvit_model = tf.keras.models.load_model(MEDVIT_MODEL, custom_objects=_custom_objects)
print(f"  MedViT loaded: {medvit_model.input_shape}")

CNN_LAYER    = "conv4_2"
MEDVIT_LAYER = "s4_l1_gfp0"

# ── Load test data from image folders ──────────────────────────────────────
import cv2
from pathlib import Path

FOLDER_TO_IDX = {"glioma": 0, "meningioma": 1, "no_tumor": 2, "pituitary": 3}
IMG_SIZE      = 224
MAX_PER_CLASS = 300   # enough to find median-confidence sample

print("Loading images from data_multiclass …")
X_list, y_list = [], []
for folder, class_idx in FOLDER_TO_IDX.items():
    folder_path = Path(DATA_DIR) / folder
    img_files   = sorted(
        list(folder_path.glob("*.jpg")) +
        list(folder_path.glob("*.jpeg")) +
        list(folder_path.glob("*.png"))
    )[:MAX_PER_CLASS]
    print(f"  {folder}: {len(img_files)} images")
    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_list.append(img.astype(np.float32) / 255.0)
        y_list.append(class_idx)

X_test = np.array(X_list, dtype=np.float32)
y_test = np.array(y_list, dtype=np.int32)
print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

# ── MedViT preprocessing (may differ from CNN) ─────────────────────────────
# MedViT V2 was trained with ImageNet mean/std if using EfficientNet backbone
# adjust ONLY if MedViT was trained with a different normalisation
MEDVIT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MEDVIT_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_medvit(img_01):
    """img_01: float32 [0,1] → normalised for MedViT."""
    # If MedViT was trained with simple [0,1] scaling (no ImageNet stats):
    #   return img_01
    # If MedViT was trained with ImageNet normalisation:
    #   return (img_01 - MEDVIT_MEAN) / MEDVIT_STD
    # Check your medvit_v2_dual_train.py to confirm. Default: same [0,1] as CNN.
    return img_01   # <- change to ImageNet norm if needed


# ── Select representative samples per class ────────────────────────────────
def find_median_confidence_sample(model, X, y, class_label_idx, preprocess_fn=None):
    """
    Among correctly classified samples of class_label_idx,
    return the one closest to median confidence.
    """
    mask   = (y == class_label_idx)
    X_cls  = X[mask]
    idxs   = np.where(mask)[0]

    preds_all = []
    batch_size = 64
    for start in range(0, len(X_cls), batch_size):
        batch = X_cls[start:start+batch_size]
        if preprocess_fn:
            batch = preprocess_fn(batch)
        preds_all.append(model.predict(batch, verbose=0))
    preds = np.concatenate(preds_all, axis=0)

    correct_mask   = (np.argmax(preds, axis=1) == class_label_idx)
    correct_confs  = preds[correct_mask, class_label_idx]
    correct_X      = X_cls[correct_mask]
    correct_global = idxs[correct_mask]

    if len(correct_confs) == 0:
        print(f"  Warning: no correctly classified samples for class {class_label_idx}")
        return X_cls[0], 0.0

    median_conf = np.median(correct_confs)
    best_i      = np.argmin(np.abs(correct_confs - median_conf))
    return correct_X[best_i], float(correct_confs[best_i])


# ── Collect one image per class ────────────────────────────────────────────
selected_imgs   = {}     # class_label → (H, W, 3) float [0,1]
cnn_confs       = {}
medvit_confs    = {}

for display_cls in DISPLAY_CLASSES:
    cnn_idx    = CNN_CLASS_IDX[display_cls]
    medvit_idx = MEDVIT_IDX[display_cls]

    print(f"\nClass: {display_cls} (CNN idx={cnn_idx}, MedViT idx={medvit_idx})")

    # Use CNN's class index for selecting from y_test (CNN labelling)
    img, conf = find_median_confidence_sample(cnn_model, X_test, y_test, cnn_idx)
    selected_imgs[display_cls] = img
    cnn_confs[display_cls]     = conf
    print(f"  CNN median-conf sample: {conf:.3f}")

    # MedViT confidence on same image
    mv_in   = preprocess_medvit(img[np.newaxis])
    mv_pred = medvit_model.predict(mv_in, verbose=0)[0]
    medvit_confs[display_cls] = float(mv_pred[medvit_idx])
    print(f"  MedViT conf (same img): {medvit_confs[display_cls]:.3f}")


# ── Compute Grad-CAMs ──────────────────────────────────────────────────────
cnn_cams    = {}
medvit_cams = {}

for display_cls in DISPLAY_CLASSES:
    img        = selected_imgs[display_cls]
    cnn_idx    = CNN_CLASS_IDX[display_cls]
    medvit_idx = MEDVIT_IDX[display_cls]

    cnn_input    = img[np.newaxis]
    medvit_input = preprocess_medvit(img[np.newaxis])

    print(f"Grad-CAM {display_cls} …")
    cnn_cams[display_cls]    = get_gradcam(cnn_model,    cnn_input,    CNN_LAYER,    cnn_idx)
    medvit_cams[display_cls] = get_gradcam(medvit_model, medvit_input, MEDVIT_LAYER, medvit_idx)


# ── Build figure ───────────────────────────────────────────────────────────
ROW_LABELS = ["Original MRI", "4-Class CNN\n(Grad-CAM)", "MedViT V2\n(Grad-CAM)"]
CMAP       = "jet"
ALPHA      = 0.45

fig = plt.figure(figsize=(15, 7.5))
outer = gridspec.GridSpec(3, 4, figure=fig, wspace=0.05, hspace=0.32,
                          left=0.14, right=0.98, top=0.90, bottom=0.10)

axes_grid = {}
for row in range(3):
    for col in range(4):
        ax = fig.add_subplot(outer[row, col])
        ax.axis("off")
        axes_grid[(row, col)] = ax

for col, display_cls in enumerate(DISPLAY_CLASSES):
    img = selected_imgs[display_cls]

    # Row 0: Original MRI
    ax = axes_grid[(0, col)]
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.set_title(display_cls, fontsize=11, fontweight="bold", pad=5)

    # Row 1: CNN Grad-CAM
    ax = axes_grid[(1, col)]
    cam  = cnn_cams[display_cls]
    over = overlay_cam(img, cam, alpha=ALPHA)
    ax.imshow(over)
    ax.text(0.5, -0.07, f"conf = {cnn_confs[display_cls]:.3f}",
            transform=ax.transAxes, fontsize=9, ha="center", va="top",
            color="#222", clip_on=False)

    # Row 2: MedViT Grad-CAM
    ax = axes_grid[(2, col)]
    cam  = medvit_cams[display_cls]
    over = overlay_cam(img, cam, alpha=ALPHA)
    ax.imshow(over)
    ax.text(0.5, -0.07, f"conf = {medvit_confs[display_cls]:.3f}",
            transform=ax.transAxes, fontsize=9, ha="center", va="top",
            color="#222", clip_on=False)

# Row labels using figure.text (avoids axes clipping)
row_label_x = 0.01
row_centers_y = []
for row in range(3):
    axs_in_row = [axes_grid[(row, c)] for c in range(4)]
    bboxes = [ax.get_position() for ax in axs_in_row]
    y_mid = np.mean([(b.y0 + b.y1) / 2 for b in bboxes])
    row_centers_y.append(y_mid)

for row, (label, y_mid) in enumerate(zip(ROW_LABELS, row_centers_y)):
    fig.text(row_label_x, y_mid, label,
             ha="left", va="center", fontsize=10, fontweight="bold",
             rotation=90,
             transform=fig.transFigure)

# Master title
fig.suptitle(
    "Grad-CAM Comparison: 4-Class CNN vs MedViT V2",
    fontsize=13, fontweight="bold", y=0.97
)

out_pdf = "xai_comparison_cnn_vs_medvit.pdf"
out_png = "xai_comparison_cnn_vs_medvit.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out_pdf}  /  {out_png}")
plt.show()
