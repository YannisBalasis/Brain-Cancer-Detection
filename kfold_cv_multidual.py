"""
5-Fold Stratified Cross-Validation — Multi-Dual System
=======================================================
Same CV scheme as kfold_cv_variance.py (professor's specification).

Multi-Dual specifics:
  - 3 outputs: branch1_3class (3-class), branch2_4class (4-class), fusion_output (4-class)
  - Branch 1 uses masked loss: No Tumor samples get label=-1 and are excluded
  - Primary evaluation metric: fusion_output accuracy (4-class)
  - EarlyStopping monitors val_loss (total weighted loss, always available)
  - Best val_acc computed from model.predict() after restore_best_weights

Run on DGX:
  export CUDA_VISIBLE_DEVICES=2
  nohup python kfold_cv_multidual.py > kfold_multidual_log.txt 2>&1 &
"""

import numpy as np
import tensorflow as tf
import cv2
import json
import time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras import layers, Model, Input, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# ── Config ───────────────────────────────────────────────────────────────────
BASE         = "/storage/data4/up1084631"
DATA_DIR     = f"{BASE}/data_multiclass"
RESULTS_DIR  = Path("kfold_results")
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE     = 224
N_FOLDS      = 5
RANDOM_STATE = 42
BATCH_SIZE   = 32
EPOCHS       = 50
PATIENCE     = 15
PREFIX       = "multidual"

# ── Resume support ────────────────────────────────────────────────────────────
START_FOLD      = 1
COMPLETED_FOLDS = {}   # fold_i: (val_acc_float, best_epoch_int)

FOLDER_TO_LABEL = {"glioma": 0, "meningioma": 1, "no_tumor": 2, "pituitary": 3}


# ── Label helpers ─────────────────────────────────────────────────────────────
def make_branch1_labels(y):
    """4-class → branch1: No Tumor (2) → -1 (masked), Pituitary (3) → 2."""
    out = y.copy().astype(np.int32)
    out[y == 2] = -1
    out[y == 3] = 2
    return out


def make_y_dict(y):
    return {
        "branch1_3class": make_branch1_labels(y),
        "branch2_4class": y,
        "fusion_output":  y,
    }


# ── Custom loss / metric for Branch 1 ────────────────────────────────────────
def masked_sparse_crossentropy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_clipped = tf.maximum(y_true, 0)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_clipped, y_pred)
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)


def masked_sparse_accuracy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_clipped = tf.cast(tf.maximum(y_true, 0), tf.int64)
    preds = tf.cast(tf.argmax(y_pred, axis=1), tf.int64)
    correct = tf.cast(tf.equal(preds, y_clipped), tf.float32)
    return tf.reduce_sum(correct * mask) / (tf.reduce_sum(mask) + 1e-8)


# ── Architecture (replicates MultiDualSystemArchitecture, dropout_rate=0.3) ──
def build_multi_dual():
    DR = 0.3
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    # Shared backbone (4 conv blocks)
    x = inp
    for filters, drop in [(32, DR*0.5), (64, DR*0.7), (128, DR), (256, DR)]:
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(drop)(x)

    x = layers.GlobalAveragePooling2D()(x)
    backbone = layers.Dense(512, activation="relu")(x)
    backbone = layers.BatchNormalization()(backbone)
    backbone = layers.Dropout(DR)(backbone)

    # Branch 1 — 3-class (masked loss for No Tumor)
    b1 = layers.Dense(256, activation="relu")(backbone)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Dropout(DR)(b1)
    b1 = layers.Dense(128, activation="relu")(b1)
    b1_feat = layers.BatchNormalization()(b1)
    b1 = layers.Dropout(DR * 0.5)(b1_feat)
    branch1_out = layers.Dense(3, activation="softmax", name="branch1_3class")(b1)

    # Branch 2 — 4-class
    b2 = layers.Dense(256, activation="relu")(backbone)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Dropout(DR)(b2)
    b2 = layers.Dense(128, activation="relu")(b2)
    b2_feat = layers.BatchNormalization()(b2)
    b2 = layers.Dropout(DR * 0.5)(b2_feat)
    branch2_out = layers.Dense(4, activation="softmax", name="branch2_4class")(b2)

    # Fusion (concatenate branch features → 4-class)
    fused = layers.Concatenate()([b1_feat, b2_feat])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(DR * 0.5)(fused)
    fusion_out = layers.Dense(4, activation="softmax", name="fusion_output")(fused)

    model = Model(
        inputs=inp,
        outputs={
            "branch1_3class": branch1_out,
            "branch2_4class": branch2_out,
            "fusion_output":  fusion_out,
        },
        name="MultiDualSystem",
    )
    model.compile(
        optimizer=Adam(1e-3),
        loss={
            "branch1_3class": masked_sparse_crossentropy,
            "branch2_4class": SparseCategoricalCrossentropy(),
            "fusion_output":  SparseCategoricalCrossentropy(),
        },
        loss_weights={"branch1_3class": 0.3, "branch2_4class": 0.4, "fusion_output": 0.3},
        metrics={
            "branch1_3class": [masked_sparse_accuracy],
            "branch2_4class": [SparseCategoricalAccuracy(name="branch2_acc")],
            "fusion_output":  [SparseCategoricalAccuracy(name="fusion_acc")],
        },
    )
    return model


# ── Callbacks ─────────────────────────────────────────────────────────────────
def make_callbacks(fold_i):
    ckpt = str(RESULTS_DIR / f"{PREFIX}_fold{fold_i}_best.keras")
    return [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            ckpt, monitor="val_loss", save_best_only=True, verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8,
            min_lr=1e-7, verbose=1,
        ),
    ]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_all_images():
    print("Loading all images ...")
    X, y = [], []
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
            X.append(img.astype(np.float32) / 255.0)
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── Fusion accuracy helper ────────────────────────────────────────────────────
def fusion_accuracy(model, X, y):
    """Compute fusion_output accuracy via predict (avoids metric name ambiguity)."""
    preds = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    fusion_preds = np.argmax(preds["fusion_output"], axis=1)
    return float(np.mean(fusion_preds == y))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print("5-Fold CV: Multi-Dual System")
    print(f"{'='*60}")

    X_all, y_all = load_all_images()

    X_tv, X_te, y_tv, y_te = train_test_split(
        X_all, y_all, test_size=0.10, stratify=y_all, random_state=RANDOM_STATE
    )
    print(f"Train+Val: {len(X_tv)}  |  Frozen Test: {len(X_te)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_val_accs = []
    best_epochs   = []

    # Restore completed folds
    for fi in range(1, START_FOLD):
        acc, ep = COMPLETED_FOLDS[fi]
        fold_val_accs.append(acc)
        best_epochs.append(ep)
        print(f"  (Fold {fi} restored: val_acc={acc*100:.2f}%  best_epoch={ep})")

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_tv, y_tv), start=1):
        if fold_i < START_FOLD:
            continue

        print(f"\n--- Fold {fold_i}/{N_FOLDS} ---")
        X_tr, X_val = X_tv[tr_idx], X_tv[val_idx]
        y_tr, y_val = y_tv[tr_idx], y_tv[val_idx]
        print(f"  Train: {len(X_tr)}  Val: {len(X_val)}")

        tf.keras.backend.clear_session()
        model = build_multi_dual()

        t0 = time.time()
        history = model.fit(
            X_tr, make_y_dict(y_tr),
            batch_size=BATCH_SIZE,
            validation_data=(X_val, make_y_dict(y_val)),
            epochs=EPOCHS,
            callbacks=make_callbacks(fold_i),
            shuffle=True,
            verbose=1,
        )
        elapsed = time.time() - t0

        # Best epoch = lowest val_loss (EarlyStopping monitored val_loss)
        best_ep = int(np.argmin(history.history["val_loss"])) + 1

        # Val accuracy from predictions (restore_best_weights already applied)
        best_val_acc = fusion_accuracy(model, X_val, y_val)

        fold_val_accs.append(best_val_acc)
        best_epochs.append(best_ep)

        print(f"  Fold {fold_i}: fusion_val_acc={best_val_acc*100:.2f}%  "
              f"best_epoch={best_ep}  ({elapsed/60:.1f} min)")
        del model

    # ── CV summary ────────────────────────────────────────────────────────
    accs_pct = np.array(fold_val_accs) * 100
    cv_mean  = float(accs_pct.mean())
    cv_std   = float(accs_pct.std())
    print(f"\n{'='*40}")
    print(f"CV ({N_FOLDS}-fold): {cv_mean:.2f}% ± {cv_std:.2f}%")
    print(f"Per-fold: {[f'{a:.2f}' for a in accs_pct]}")

    # ── Final model ───────────────────────────────────────────────────────
    final_epochs = int(np.median(best_epochs))
    print(f"\n--- Final model: {len(X_tv)} samples, {final_epochs} epochs ---")

    tf.keras.backend.clear_session()
    final_model = build_multi_dual()

    t0 = time.time()
    final_model.fit(
        X_tv, make_y_dict(y_tv),
        batch_size=BATCH_SIZE,
        epochs=final_epochs,
        shuffle=True,
        verbose=1,
    )
    elapsed = time.time() - t0

    test_acc = fusion_accuracy(final_model, X_te, y_te)
    print(f"Final test accuracy (fusion): {test_acc*100:.2f}%  ({elapsed/60:.1f} min)")

    final_model.save(str(RESULTS_DIR / f"{PREFIX}_final.keras"))

    results = {
        "model": "multidual",
        "n_folds": N_FOLDS,
        "cv_mean_pct":  cv_mean,
        "cv_std_pct":   cv_std,
        "fold_accuracies_pct": accs_pct.tolist(),
        "best_epochs_per_fold": best_epochs,
        "final_epochs_used": final_epochs,
        "final_test_accuracy_pct": test_acc * 100,
    }
    out = RESULTS_DIR / f"{PREFIX}_cv_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out}")
    print(f"\nSUMMARY  [Multi-Dual System]")
    print(f"  CV  : {cv_mean:.2f}% ± {cv_std:.2f}%  (5-fold on train+val)")
    print(f"  Test: {test_acc*100:.2f}%  (frozen test set, final model)")


if __name__ == "__main__":
    main()
