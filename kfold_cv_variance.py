"""
5-Fold Stratified Cross-Validation — Variance Estimation
Clean CV scheme (professor's specification):

  1. Reproduce 90/10 stratified split (random_state=42) -> frozen test set
  2. 5-fold stratified CV on train+val (90%) -> mean +/- std accuracy per model
  3. Train final model on full train+val (epochs = median best epoch across folds)
     -> evaluate on frozen test set

The frozen test set (703 samples) is NEVER touched during CV.

Supported models (run one at a time via MODEL_TO_RUN):
  "4class_cnn"   -> 4-Class CNN  (Glioma/Meningioma/No Tumor/Pituitary, 1.28M)
  "3class_cnn"   -> 3-Class CNN  (Glioma/Meningioma/Pituitary only,     1.28M)
  "binary_cnn"   -> Binary CNN   (Tumor / No Tumor,                       1.44M)

Run on DGX. Estimated time per model: ~40-50 min (5 folds + final).
"""

import numpy as np
import tensorflow as tf
import cv2
import json
import time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ──────────────────────────────────────────────────────────────────
BASE          = "/storage/data4/up1084631"
DATA_DIR      = f"{BASE}/data_multiclass"
RESULTS_DIR   = Path("kfold_results")
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE     = 224
N_FOLDS      = 5
RANDOM_STATE = 42

# ── Edit this to select which model to run ───────────────────────────────────
MODEL_TO_RUN = "binary_cnn"   # "4class_cnn" | "3class_cnn" | "binary_cnn"

# ── Resume from a specific fold (1-indexed). Set to 1 to run all folds.
# Paste previously completed fold results in COMPLETED_FOLDS below.
START_FOLD       = 1
COMPLETED_FOLDS  = {}

FOLDER_TO_LABEL = {"glioma": 0, "meningioma": 1, "no_tumor": 2, "pituitary": 3}


# ── Model builders ───────────────────────────────────────────────────────────

def build_4class_cnn():
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inp
    for filters, drop in [(32, 0.25), (64, 0.25), (128, 0.30), (256, 0.30)]:
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(drop)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(4, activation="softmax")(x)
    m = models.Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def build_3class_cnn():
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inp
    for filters, drop in [(32, 0.25), (64, 0.25), (128, 0.30), (256, 0.30)]:
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(drop)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(3, activation="softmax")(x)
    m = models.Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def build_binary_cnn():
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inp
    for filters, drop in [(32, 0.25), (64, 0.35), (128, 0.40), (256, 0.50)]:
        x = layers.Conv2D(filters, 3, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(drop)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = models.Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return m


# ── Model registry ───────────────────────────────────────────────────────────
REGISTRY = {
    "4class_cnn": {
        "builder":       build_4class_cnn,
        "label_filter":  None,          # use all 4 classes
        "label_remap":   None,
        "binary":        False,
        "epochs":        50,
        "batch_size":    32,
        "patience":      15,
        "prefix":        "4class_cnn",
    },
    "3class_cnn": {
        "builder":       build_3class_cnn,
        "label_filter":  [0, 1, 3],     # exclude No Tumor (label=2)
        "label_remap":   {0: 0, 1: 1, 3: 2},  # Glioma=0, Meningioma=1, Pituitary=2
        "binary":        False,
        "epochs":        50,
        "batch_size":    32,
        "patience":      15,
        "prefix":        "3class_cnn",
    },
    "binary_cnn": {
        "builder":       build_binary_cnn,
        "label_filter":  None,
        "label_remap":   None,
        "binary":        True,           # remap: 0=No Tumor, 1=Tumor (any)
        "epochs":        50,
        "batch_size":    32,
        "patience":      15,
        "prefix":        "binary_cnn",
    },
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_all_images():
    print("Loading all images ...")
    X, y = [], []
    for folder, label in sorted(FOLDER_TO_LABEL.items()):
        folder_path = Path(DATA_DIR) / folder
        img_files = sorted(
            list(folder_path.glob("*.jpg")) +
            list(folder_path.glob("*.jpeg")) +
            list(folder_path.glob("*.png"))
        )
        print(f"  {folder}: {len(img_files)}")
        for img_path in img_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img.astype(np.float32) / 255.0)
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def apply_transforms(X, y, cfg):
    """Filter classes and remap labels according to model config."""
    if cfg["binary"]:
        # 0=No Tumor, 1=Tumor (glioma/meningioma/pituitary)
        y_bin = (y != 2).astype(np.int32)
        return X, y_bin

    if cfg["label_filter"] is not None:
        mask = np.isin(y, cfg["label_filter"])
        X = X[mask]
        y_filtered = np.array([cfg["label_remap"][lbl] for lbl in y[mask]], dtype=np.int32)
        return X, y_filtered

    return X, y


# ── Augmentation ──────────────────────────────────────────────────────────────
def make_augmentor():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        fill_mode="nearest",
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────
def make_callbacks(fold_i, cfg):
    ckpt_path = str(RESULTS_DIR / f"{cfg['prefix']}_fold{fold_i}_best.keras")
    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=cfg["patience"],
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8,
            min_lr=1e-7, verbose=1,
        ),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg = REGISTRY[MODEL_TO_RUN]
    print(f"\n{'='*60}")
    print(f"5-Fold CV: {MODEL_TO_RUN}")
    print(f"{'='*60}")

    # Load data
    X_all, y_all = load_all_images()

    # Reproduce frozen test split (SAME random_state as all training scripts)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y_all,
        test_size=0.10,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )
    print(f"Train+Val: {len(X_trainval)}  |  Frozen Test: {len(X_test)}")

    # Apply class filter / label remap
    X_tv, y_tv   = apply_transforms(X_trainval, y_trainval, cfg)
    X_te, y_te   = apply_transforms(X_test,     y_test,     cfg)
    print(f"After transform -> TrainVal: {len(X_tv)}, Test: {len(X_te)}")

    # ── 5-Fold CV ─────────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    # Restore any already-completed folds
    fold_val_accs = []
    best_epochs   = []
    for fi in range(1, START_FOLD):
        acc, ep = COMPLETED_FOLDS[fi]
        fold_val_accs.append(acc)
        best_epochs.append(ep)
        print(f"  (Fold {fi} restored from cache: val_acc={acc*100:.2f}%  best_epoch={ep})")

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_tv, y_tv), start=1):
        if fold_i < START_FOLD:
            continue   # skip already-completed folds

        print(f"\n--- Fold {fold_i}/{N_FOLDS} ---")
        X_tr, X_val = X_tv[tr_idx], X_tv[val_idx]
        y_tr, y_val = y_tv[tr_idx], y_tv[val_idx]
        print(f"  Train: {len(X_tr)}  Val: {len(X_val)}")

        tf.keras.backend.clear_session()
        model = cfg["builder"]()

        t0 = time.time()
        history = model.fit(
            X_tr, y_tr,
            batch_size=cfg["batch_size"],
            validation_data=(X_val, y_val),
            epochs=cfg["epochs"],
            callbacks=make_callbacks(fold_i, cfg),
            shuffle=True,
            verbose=1,
        )
        elapsed = time.time() - t0

        best_val_acc = float(max(history.history["val_accuracy"]))
        best_ep      = int(np.argmax(history.history["val_accuracy"])) + 1
        fold_val_accs.append(best_val_acc)
        best_epochs.append(best_ep)

        print(f"  Fold {fold_i}: val_acc={best_val_acc*100:.2f}%  "
              f"best_epoch={best_ep}  ({elapsed/60:.1f} min)")
        del model

    # ── CV summary ────────────────────────────────────────────────────────
    accs_pct = np.array(fold_val_accs) * 100
    cv_mean  = float(accs_pct.mean())
    cv_std   = float(accs_pct.std())
    print(f"\n{'='*40}")
    print(f"CV ({N_FOLDS}-fold): {cv_mean:.2f}% ± {cv_std:.2f}%")
    print(f"Per-fold: {[f'{a:.2f}' for a in accs_pct]}")

    # ── Final model on full train+val ─────────────────────────────────────
    final_epochs = int(np.median(best_epochs))
    print(f"\n--- Final model: {len(X_tv)} samples, {final_epochs} epochs ---")

    tf.keras.backend.clear_session()
    final_model = cfg["builder"]()
    t0 = time.time()
    final_model.fit(X_tv, y_tv,
                    batch_size=cfg["batch_size"],
                    epochs=final_epochs,
                    shuffle=True,
                    verbose=1)
    elapsed = time.time() - t0

    test_loss, test_acc = final_model.evaluate(X_te, y_te, verbose=0)
    test_acc_pct = test_acc * 100
    print(f"Final test accuracy: {test_acc_pct:.2f}%  ({elapsed/60:.1f} min)")

    final_model.save(str(RESULTS_DIR / f"{cfg['prefix']}_final.keras"))

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "model": MODEL_TO_RUN,
        "n_folds": N_FOLDS,
        "cv_mean_pct":  cv_mean,
        "cv_std_pct":   cv_std,
        "fold_accuracies_pct": accs_pct.tolist(),
        "best_epochs_per_fold": best_epochs,
        "final_epochs_used": final_epochs,
        "final_test_accuracy_pct": test_acc_pct,
        "final_test_loss": float(test_loss),
    }
    out = RESULTS_DIR / f"{cfg['prefix']}_cv_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out}")
    print(f"\nSUMMARY  [{MODEL_TO_RUN}]")
    print(f"  CV  : {cv_mean:.2f}% ± {cv_std:.2f}%  (5-fold on train+val)")
    print(f"  Test: {test_acc_pct:.2f}%  (frozen test set, final model)")


if __name__ == "__main__":
    main()
