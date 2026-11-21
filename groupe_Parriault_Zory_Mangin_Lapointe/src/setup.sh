#!/usr/bin/env bash
# Creates .venv, installs requirements and checks which device TensorFlow uses.
# Works on machines with or without an NVIDIA GPU.
set -euo pipefail
cd "$(dirname "$0")"

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# TensorFlow 2.21 does not search the pip-installed cuSOLVER, so preload it at interpreter startup.
# Harmless when the CUDA wheels are absent (non-Linux) or when there is no GPU.
SITE=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cat > "$SITE/preload_nvidia_cusolver.pth" <<'PTH'
import os, ctypes, sysconfig; p = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cusolver", "lib", "libcusolver.so.11"); exec("try: os.path.exists(p) and ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)\nexcept OSError: pass")
PTH

.venv/bin/python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"TensorFlow {tf.__version__}: " + (f"GPU OK -> {gpus}" if gpus else "no GPU found, running on CPU"))
PY
