#!/usr/bin/env bash
# Environment setup. torch comes from the CPU index first so a local install
# pulls the ~200 MB wheels instead of the ~2.5 GB CUDA ones; on Colab the GPU
# build is already in the image and this is a no-op.
set -euo pipefail

PY="${PYTHON:-python}"
VENV="${VENV:-.venv}"

if [ ! -d "$VENV" ]; then
  echo "==> creating $VENV"
  "$PY" -m venv "$VENV"
fi

if [ -f "$VENV/Scripts/python.exe" ]; then
  VPY="$VENV/Scripts/python.exe"        # Windows
else
  VPY="$VENV/bin/python"                # POSIX
fi

echo "==> upgrading pip"
"$VPY" -m pip install --quiet --upgrade pip setuptools wheel

echo "==> installing torch (CPU wheels)"
"$VPY" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "==> installing the rest"
"$VPY" -m pip install -r requirements.txt

echo "==> freezing exact versions"
"$VPY" -m pip freeze > requirements.lock.txt

echo "==> verifying imports"
"$VPY" - <<'PYEOF'
import importlib
mods = ["torch", "transformers", "librosa", "soundfile", "datasets",
        "gradio", "onnxruntime", "silero_vad", "sklearn", "sarvamai"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m}: {e}")
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if bad:
    raise SystemExit("FAILED imports:\n  " + "\n  ".join(bad))
print("  all imports ok")
PYEOF

echo
echo "Done. Next:"
echo "  $VPY scripts/prepare_data.py --split test --all-languages --max-rows 400"
echo "  $VPY scripts/run_baselines.py --cache data/cache/test"
