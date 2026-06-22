"""
Computational Statistics — GPU memory, params, inference time
=============================================================
Loads every model, runs a warm-up + timed forward pass on a small
batch, and reports:
  - Total trainable parameters
  - Peak GPU memory (MiB) during inference
  - Mean inference latency (ms) over N_RUNS batches
  - GPU / CUDA / TF environment info

Run on DGX:
  export CUDA_VISIBLE_DEVICES=0
  python compute_stats.py
"""

import os, time, json, inspect
import numpy as np
import tensorflow as tf

BASE = "/storage/data4/up1084631"

# ── GPU / environment info ─────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
print("\n=== Environment ===")
print(f"TensorFlow : {tf.__version__}")
print(f"Python     : {__import__('sys').version.split()[0]}")
print(f"GPU(s)     : {[g.name for g in gpus]}")

# nvidia-smi for driver + VRAM
os.system("nvidia-smi --query-gpu=name,memory.total,driver_version "
          "--format=csv,noheader")

# ── Custom objects for MedViT ──────────────────────────────────────────────────
import medvit_v2_architecture
_CUSTOM = {
    name: obj
    for name, obj in inspect.getmembers(medvit_v2_architecture, inspect.isclass)
    if obj.__module__ == "medvit_v2_architecture"
}

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "Binary CNN": {
        "path":   f"{BASE}/models/best_binary_model.h5",
        "custom": False,
        "ensemble": False,
    },
    "4-Class CNN": {
        "path":   f"{BASE}/best_multiclass_4class_model.h5",
        "custom": False,
        "ensemble": False,
    },
    "3-Class CNN": {
        "path":   f"{BASE}/models/best_multiclass_3class_model.h5",
        "custom": False,
        "ensemble": False,
    },
    "1-vs-1 Ensemble": {
        "path":   [
            f"{BASE}/models/ensemble/best_glioma_vs_meningioma_model.h5",
            f"{BASE}/models/ensemble/best_glioma_vs_pituitary_model.h5",
            f"{BASE}/models/ensemble/best_meningioma_vs_pituitary_model.h5",
        ],
        "custom": False,
        "ensemble": True,
    },
    "Multi-Dual System": {
        "path":   f"{BASE}/models/best_multi_dual_model.h5",
        "custom": False,
        "ensemble": False,
    },
    "ResNet-50 Dual": {
        "path":   f"{BASE}/models/best_resnet_dual_model.h5",
        "custom": False,
        "ensemble": False,
    },
    "EfficientNet-B3 Dual": {
        "path":   f"{BASE}/models/best_efnet_dual_model.h5",
        "custom": False,
        "ensemble": False,
    },
    "MedViT V2 Base": {
        "path":   f"{BASE}/medvit_v2_experiment_v2/best_medvit_v2_base.keras",
        "custom": True,
        "ensemble": False,
    },
    "MedViT V2 Dual": {
        "path":   f"{BASE}/medvit_v2_dual_experiment/best_medvit_v2_dual.keras",
        "custom": True,
        "ensemble": False,
    },
}

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 1          # single-image latency (matches paper Table 1)
N_WARMUP   = 5
N_RUNS     = 50
DUMMY      = np.random.rand(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)

results = {}

for name, cfg in MODELS.items():
    print(f"\n{'─'*55}")
    print(f"Model: {name}")

    tf.keras.backend.clear_session()
    custom_obj = _CUSTOM if cfg["custom"] else {}

    # ── Ensemble: load 3 sub-models ───────────────────────────────────────────
    if cfg["ensemble"]:
        paths = cfg["path"]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"  [SKIP] missing: {missing}")
            continue
        sub_models = [tf.keras.models.load_model(p, compile=False)
                      for p in paths]
        params = sum(m.count_params() for m in sub_models)
        print(f"  Params (total 3 sub-models): {params:,}")

        tf.config.experimental.reset_memory_stats("GPU:0")
        for m in sub_models:
            m(DUMMY, training=False)
        mem_info = tf.config.experimental.get_memory_info("GPU:0")
        peak_mib = mem_info["peak"] / (1024 ** 2)
        print(f"  Peak GPU memory: {peak_mib:.1f} MiB")

        for _ in range(N_WARMUP):
            for m in sub_models:
                m(DUMMY, training=False)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            for m in sub_models:
                m(DUMMY, training=False)
            times.append((time.perf_counter() - t0) * 1000)

        for m in sub_models:
            del m

    # ── Single model ──────────────────────────────────────────────────────────
    else:
        path = cfg["path"]
        if not os.path.exists(path):
            print(f"  [SKIP] file not found: {path}")
            continue
        model = tf.keras.models.load_model(path, compile=False,
                                           custom_objects=custom_obj)
        params = model.count_params()
        print(f"  Params: {params:,}")

        tf.config.experimental.reset_memory_stats("GPU:0")
        _ = model(DUMMY, training=False)
        mem_info = tf.config.experimental.get_memory_info("GPU:0")
        peak_mib = mem_info["peak"] / (1024 ** 2)
        print(f"  Peak GPU memory: {peak_mib:.1f} MiB")

        for _ in range(N_WARMUP):
            model(DUMMY, training=False)
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            model(DUMMY, training=False)
            times.append((time.perf_counter() - t0) * 1000)
        del model

    mean_ms = float(np.mean(times))
    std_ms  = float(np.std(times))
    print(f"  Inference latency: {mean_ms:.1f} ± {std_ms:.1f} ms "
          f"(batch={BATCH_SIZE}, n={N_RUNS})")

    results[name] = {
        "params":            params,
        "peak_gpu_mib":      round(peak_mib, 1),
        "latency_mean_ms":   round(mean_ms, 1),
        "latency_std_ms":    round(std_ms, 1),
    }

    tf.keras.backend.clear_session()

# ── Also check for training log files ─────────────────────────────────────────
print("\n\n=== Checking for training log files ===")
import glob
log_patterns = [
    f"{BASE}/*.log", f"{BASE}/*.txt",
    f"{BASE}/medvit_v2_experiment_v2/*.log",
    f"{BASE}/medvit_v2_experiment_v2/*.txt",
    f"{BASE}/medvit_v2_dual_experiment/*.log",
    f"{BASE}/medvit_v2_dual_experiment/*.txt",
]
for pat in log_patterns:
    for f in glob.glob(pat):
        size = os.path.getsize(f)
        print(f"  {f}  ({size} bytes)")

# ── Save ──────────────────────────────────────────────────────────────────────
out = "compute_stats_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\nSaved: {out}")
print("\n=== SUMMARY ===")
print(f"{'Model':<30} {'Params':>10} {'Peak MiB':>10} {'Latency (ms)':>15}")
print("─" * 68)
for name, r in results.items():
    print(f"{name:<30} {r['params']:>10,} {r['peak_gpu_mib']:>10.1f} "
          f"  {r['latency_mean_ms']:.1f} ± {r['latency_std_ms']:.1f}")
