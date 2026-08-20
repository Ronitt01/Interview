"""Hugging Face Space entry point.

Spaces run the file named by `app_file` in the README front matter (default
`app.py`) from the repository root, and expect the Gradio app to be launched at
import time on 0.0.0.0:7860. That is a different contract from
``demo/app.py``, which is a CLI with argparse — so this file adapts rather than
duplicates. All the UI and inference logic lives in ``demo/app.py`` and there is
no second copy to keep in sync.

Locally you can run either:

    python demo/app.py            # CLI, picks its own port, --share, --checkpoint
    python app.py                 # exactly what the Space runs

Weights: the Space serves whatever ``find_checkpoint`` discovers in ``weights/``,
preferring an int8 ONNX graph because that is the artefact that would actually
deploy. If no weights are committed, it falls back to the E0 energy baseline and
says so in the UI rather than erroring — a Space that boots and explains itself
is more useful than one that shows a stack trace.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (str(ROOT), str(ROOT / "demo")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import the real module by path, not by name. `import app` from here would
# resolve to *this* file (both are named `app`), so the UI module is loaded
# explicitly from demo/app.py under a distinct module name.
import importlib.util

_spec = importlib.util.spec_from_file_location("turn_demo", ROOT / "demo" / "app.py")
assert _spec and _spec.loader
turn_demo = importlib.util.module_from_spec(_spec)
sys.modules["turn_demo"] = turn_demo
_spec.loader.exec_module(turn_demo)


def build():
    """Resolve weights, build the Blocks app, and return it."""
    checkpoint = os.environ.get("TURN_CHECKPOINT") or None
    onnx = os.environ.get("TURN_ONNX") or None
    if not checkpoint and not onnx:
        checkpoint, onnx = turn_demo.find_checkpoint(None)

    turn_demo.BACKEND = turn_demo.Backend(checkpoint, onnx)
    print(f"  detector: {turn_demo.BACKEND.info}", flush=True)
    return turn_demo.build_ui()


demo = build()

if __name__ == "__main__":
    # 0.0.0.0 and 7860 are what the Spaces runtime proxies. Both are overridable
    # by the GRADIO_SERVER_* env vars the Dockerfile sets, so this default only
    # applies when nothing else has an opinion.
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
