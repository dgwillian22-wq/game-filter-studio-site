"""Motores de upscale: IA (PyTorch + spandrel) e classico (Lanczos + nitidez)."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

from .models import ensure_model


class ClassicBackend:
    """Sem IA: Lanczos + mascara de nitidez. Funciona em qualquer maquina."""

    name = "classic"

    def upscale(self, rgb: np.ndarray, scale: int) -> np.ndarray:
        h, w = rgb.shape[:2]
        img = Image.fromarray(rgb).resize((w * scale, h * scale), Image.LANCZOS)
        if scale > 1:
            img = img.filter(ImageFilter.UnsharpMask(radius=max(1.0, scale / 2), percent=60, threshold=2))
        return np.asarray(img)


class AIBackend:
    """Qualquer modelo de super-resolucao suportado pelo spandrel (ESRGAN,
    Real-ESRGAN, SwinIR, HAT...). Processa em blocos para caber na memoria."""

    def __init__(self, model: str, tile: int = 0, device: str | None = None):
        try:
            import torch
            from spandrel import ImageModelDescriptor, ModelLoader
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "O motor de IA precisa de PyTorch e spandrel:\n  pip install torch spandrel\n"
                "(ou use --model classic, que nao precisa de nada alem de Pillow/numpy)."
            ) from e
        self.torch = torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        desc = ModelLoader().load_from_file(str(ensure_model(model)))
        if not isinstance(desc, ImageModelDescriptor):
            raise SystemExit("Esse arquivo nao e um modelo de imagem -> imagem.")
        if desc.input_channels != 3 or desc.output_channels != 3:
            raise SystemExit("Apenas modelos RGB (3 canais) sao suportados.")
        self.desc = desc.to(device).eval()
        self.half = device == "cuda" and desc.supports_half
        if self.half:
            self.desc.half()
        self.native = int(desc.scale)
        self.tile = tile or (512 if device != "cpu" else 192)
        self.name = f"ai:{model}@{device}"

    def _forward(self, x):
        req = self.desc.size_requirements
        _, _, h, w = x.shape
        mult = max(1, getattr(req, "multiple_of", 1))
        ph, pw = (-h) % mult, (-w) % mult
        if getattr(req, "square", False):
            side = max(h + ph, w + pw)
            side += (-side) % mult
            ph, pw = side - h, side - w
        if ph or pw:
            x = self.torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect" if min(h, w) > max(ph, pw) else "replicate")
        y = self.desc(x)
        if ph or pw:
            y = y[:, :, : h * self.native, : w * self.native]
        return y

    def _tiled(self, x, pad: int = 16):
        _, _, h, w = x.shape
        s, t = self.native, self.tile
        if h <= t and w <= t:
            return self._forward(x)
        out = x.new_zeros((1, 3, h * s, w * s))
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                y1, x1 = min(y0 + t, h), min(x0 + t, w)
                ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
                yb, xb = min(y1 + pad, h), min(x1 + pad, w)
                patch = self._forward(x[:, :, ya:yb, xa:xb])
                oy, ox = (y0 - ya) * s, (x0 - xa) * s
                out[:, :, y0 * s : y1 * s, x0 * s : x1 * s] = patch[:, :, oy : oy + (y1 - y0) * s, ox : ox + (x1 - x0) * s]
        return out

    def upscale(self, rgb: np.ndarray, scale: int) -> np.ndarray:
        torch = self.torch
        h, w = rgb.shape[:2]
        passes = max(1, math.ceil(math.log(scale) / math.log(self.native))) if scale > 1 else 1
        x = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1)[None].float().div(255.0)
        x = x.to(self.device)
        if self.half:
            x = x.half()
        with torch.inference_mode():
            for _ in range(passes):
                x = self._tiled(x).clamp_(0, 1)
        out = (x[0].float().permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        target = (w * scale, h * scale)
        if out.shape[1] != target[0] or out.shape[0] != target[1]:
            out = np.asarray(Image.fromarray(out).resize(target, Image.LANCZOS))
        return out


def get_backend(model: str, tile: int = 0, device: str | None = None):
    if model == "classic":
        return ClassicBackend()
    return AIBackend(model, tile=tile, device=device)
