"""Download do modelo de profundidade (Depth Anything V2 Small, ONNX)."""
import os
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/"
FILES = {"fp32": "model.onnx", "fp16": "model_fp16.onnx", "int8": "model_quantized.onnx"}


def models_dir():
    root = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
    d = Path(root) / "WLSS3D" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_model(precision="fp32", progress=None):
    name = FILES[precision]
    dest = models_dir() / f"depth-anything-v2-small-{name}"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(BASE + name, timeout=30) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(done / total)
    tmp.replace(dest)
    return dest
