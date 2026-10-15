"""Pipeline de textura: alfa, tileable, mapas de dados e normais.

Tudo aqui trabalha com Pillow/numpy; o modelo de IA entra pelo ``backend``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

INPUT_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff",
    ".dds", ".webp", ".pcx", ".gif",
}

_NORMAL_RE = re.compile(r"(?:^|[_\-.])(n|nrm|nm|normal|normals|normalmap)$", re.I)
_DATA_RE = re.compile(
    r"(?:^|[_\-.])(bump|height|disp|displacement|spec|specular|gloss|rough|"
    r"roughness|metal|metallic|ao|mask|lm|lightmap|opacity|orm)$",
    re.I,
)


def texture_kind(path: str | Path) -> str:
    """'normal', 'data' ou 'color', pelo sufixo do nome (``rocha_n.png``)."""
    stem = Path(path).stem
    if _NORMAL_RE.search(stem):
        return "normal"
    if _DATA_RE.search(stem):
        return "data"
    return "color"


@dataclass
class Options:
    scale: int = 4
    strength: float = 1.0       # 0 = so Lanczos, 1 = saida pura da IA
    tileable: bool = False      # textura repetivel: evita emendas
    sharpen: float = 0.0        # nitidez extra no final (0 = nenhuma)
    data_maps: str = "classic"  # normal/spec/mask: 'classic' | 'ai' | 'skip'


class Skip(Exception):
    """Arquivo deve ser ignorado (ex.: mapa de dados com data_maps='skip')."""


def resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Lanczos em HxW[xC] uint8 -> uint8. ``size`` = (largura, altura)."""
    return np.asarray(Image.fromarray(arr).resize(size, Image.LANCZOS))


def _resize_float(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    chans = [
        np.asarray(Image.fromarray(arr[..., c].astype(np.float32), "F").resize(size, Image.LANCZOS))
        for c in range(arr.shape[2])
    ]
    return np.stack(chans, axis=-1)


def bleed_colors(rgb: np.ndarray, alpha: np.ndarray, iters: int = 16) -> np.ndarray:
    """Espalha a cor das bordas opacas para os pixels transparentes.

    Sem isso, o RGB "escondido" (geralmente preto) vaza como halo escuro
    nas bordas de texturas recortadas depois do upscale.
    """
    known = alpha > 0
    if known.all() or not known.any():
        return rgb
    rgb = rgb.astype(np.float32).copy()
    for _ in range(iters):
        acc = np.zeros_like(rgb)
        cnt = np.zeros(known.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            sh_k = np.roll(known, (dy, dx), (0, 1))
            sh_c = np.roll(rgb, (dy, dx), (0, 1))
            acc += np.where(sh_k[..., None], sh_c, 0.0)
            cnt += sh_k
        fill = ~known & (cnt > 0)
        if not fill.any():
            break
        rgb[fill] = acc[fill] / cnt[fill][:, None]
        known = known | fill
        if known.all():
            break
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def _normalize_normals(arr: np.ndarray) -> np.ndarray:
    v = arr.astype(np.float32) / 127.5 - 1.0
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    v = np.where(n > 1e-6, v / np.maximum(n, 1e-6), np.array([0, 0, 1], np.float32))
    return np.clip((v + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8)


def _to_working(img: Image.Image) -> tuple[Image.Image, str]:
    """Reduz qualquer modo do Pillow para RGB/RGBA (+ lembra se era cinza)."""
    if img.mode in ("L", "I;16", "I"):
        if img.mode != "L":
            img = img.point(lambda v: v / 257.0 if img.mode == "I;16" else v / 256.0).convert("L")
        return img.convert("RGB"), "L"
    if img.mode == "LA":
        return img.convert("RGBA"), "LA"
    if img.mode == "P":
        return (img.convert("RGBA") if "transparency" in img.info else img.convert("RGB")), "P"
    if img.mode in ("RGBA", "RGB"):
        return img, img.mode
    return (img.convert("RGBA") if "A" in img.getbands() else img.convert("RGB")), img.mode


def process_image(img: Image.Image, kind: str, opts: Options, backend) -> Image.Image:
    """Aumenta ``img`` por ``opts.scale``. ``kind``: color | normal | data."""
    if opts.scale < 1:
        raise ValueError("scale deve ser >= 1")
    if kind != "color" and opts.data_maps == "skip":
        raise Skip(kind)

    work, original_mode = _to_working(img)
    arr = np.asarray(work)
    has_alpha = arr.shape[2] == 4
    rgb = arr[..., :3]
    alpha = arr[..., 3] if has_alpha else None
    if has_alpha:
        rgb = bleed_colors(rgb, alpha)

    h, w = rgb.shape[:2]
    scale = opts.scale
    pad = 0
    if opts.tileable:
        pad = max(8, min(h, w) // 8)
        rgb = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="wrap")
        if has_alpha:
            alpha = np.pad(alpha, pad, mode="wrap")

    ph, pw = rgb.shape[:2]
    target = (pw * scale, ph * scale)
    use_ai = kind == "color" or opts.data_maps == "ai"

    if kind == "normal" and not use_ai:
        out = _normalize_normals(_resize_float(rgb, target))
    elif use_ai:
        out = backend.upscale(rgb, scale)
        if opts.strength < 1.0:
            base = resize(rgb, target).astype(np.float32)
            out = np.clip(base + opts.strength * (out.astype(np.float32) - base) + 0.5, 0, 255).astype(np.uint8)
        if kind == "normal":
            out = _normalize_normals(out)
    else:
        out = resize(rgb, target)

    if has_alpha:
        out = np.dstack([out, resize(alpha, target)])

    if pad:
        c = pad * scale
        out = out[c:-c, c:-c]

    result = Image.fromarray(out)
    if opts.sharpen > 0:
        rgb_img = result.convert("RGB").filter(
            ImageFilter.UnsharpMask(radius=1.2, percent=int(opts.sharpen * 100), threshold=1)
        )
        if has_alpha:
            rgb_img.putalpha(result.getchannel("A"))
        result = rgb_img

    if original_mode in ("L", "LA"):
        gray = result.convert("L")
        if original_mode == "LA":
            gray = Image.merge("LA", (gray, result.getchannel("A")))
        result = gray
    return result
