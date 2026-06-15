from __future__ import annotations

import os
import sys


def check_tensorflow_cpu() -> int:
    """Print TensorFlow devices and run a tiny operation on the CPU."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    try:
        import tensorflow as tf
    except ImportError:
        print("[ERREUR] TensorFlow n'est pas installe dans cet environnement.")
        print("Installe-le dans ton venv avec : pip install tensorflow")
        return 1

    print(f"TensorFlow version : {tf.__version__}")

    cpus = tf.config.list_physical_devices("CPU")
    gpus = tf.config.list_physical_devices("GPU")

    print(f"CPU detectes : {len(cpus)}")
    for cpu in cpus:
        print(f" - {cpu.name} ({cpu.device_type})")

    print(f"GPU detectes : {len(gpus)}")
    for gpu in gpus:
        print(f" - {gpu.name} ({gpu.device_type})")

    if not cpus:
        print("[ERREUR] Aucun CPU detecte par TensorFlow.")
        return 1

    with tf.device("/CPU:0"):
        result = tf.reduce_sum(tf.constant([1.0, 2.0, 3.0])).numpy()

    print(f"Test calcul CPU OK : 1 + 2 + 3 = {result:.1f}")
    print("[OK] TensorFlow detecte et utilise le CPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(check_tensorflow_cpu())
