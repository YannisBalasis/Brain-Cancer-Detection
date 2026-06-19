"""Inference time benchmark — run on DGX with GPU."""
import tensorflow as tf
import numpy as np
import time, inspect

BASE = "/storage/data4/up1084631"
models_to_bench = {
    "Binary CNN":        f"{BASE}/models/best_binary_model.h5",
    "4-Class CNN":       f"{BASE}/best_multiclass_4class_model.h5",
    "3-Class CNN":       f"{BASE}/models/best_multiclass_3class_model.h5",
    "ResNet-50 Dual":    f"{BASE}/models/best_resnet_dual_model.h5",
    "EfficientNet Dual": f"{BASE}/models/best_efnet_dual_model.h5",
    "Multi-Dual":        f"{BASE}/models/best_multi_dual_model.h5",
    "MedViT V2 Base":    f"{BASE}/medvit_v2_experiment_v2/best_medvit_v2_base.keras",
    "MedViT V2 Dual":    f"{BASE}/medvit_v2_dual_experiment/best_medvit_v2_dual.keras",
}

import medvit_v2_architecture
custom_objects = {
    name: obj
    for name, obj in inspect.getmembers(medvit_v2_architecture, inspect.isclass)
    if obj.__module__ == "medvit_v2_architecture"
}

dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
N = 200

print(f"{'Model':<25} {'Mean (ms)':>10} {'Std (ms)':>10}")
print("-" * 47)
for name, path in models_to_bench.items():
    try:
        m = tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)
        for _ in range(10):
            m.predict(dummy, verbose=0)
        times = []
        for _ in range(N):
            t0 = time.perf_counter()
            m.predict(dummy, verbose=0)
            times.append((time.perf_counter() - t0) * 1000)
        t = np.array(times)
        print(f"{name:<25} {t.mean():>10.1f} {t.std():>10.1f}")
    except Exception as e:
        print(f"{name:<25} ERROR: {e}")
