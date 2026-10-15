"""Registro de modelos e download com verificacao de integridade."""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_REL = "https://github.com/xinntao/Real-ESRGAN/releases/download"


@dataclass(frozen=True)
class Model:
    key: str
    filename: str
    url: str
    sha256: str
    scale: int
    desc: str


MODELS = {
    "general": Model(
        "general", "realesr-general-x4v3.pth", f"{_REL}/v0.2.5.0/realesr-general-x4v3.pth",
        "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292", 4, "Leve e rapido. Bom para texturas pequenas e com artefatos de compressao (padrao).",
    ),
    "photo": Model(
        "photo", "RealESRGAN_x4plus.pth", f"{_REL}/v0.1.0/RealESRGAN_x4plus.pth",
        "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1", 4, "Qualidade maxima para texturas realistas (rocha, madeira, tecido). Pesado.",
    ),
    "anime": Model(
        "anime", "RealESRGAN_x4plus_anime_6B.pth", f"{_REL}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da", 4, "Cel shading / desenho: linhas limpas e cores chapadas.",
    ),
}


def home() -> Path:
    base = os.environ.get("TEXAI_HOME")
    return Path(base) if base else Path.home() / ".cache" / "texai" / "models"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(name_or_path: str, quiet: bool = False) -> Path:
    """Devolve o caminho do .pth/.safetensors, baixando se for modelo do registro.

    Aceita tambem um caminho para modelo proprio (ex.: baixado do OpenModelDB).
    """
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return p
    if name_or_path not in MODELS:
        raise SystemExit(
            f"Modelo desconhecido: {name_or_path!r}. Use: {', '.join(MODELS)}, 'classic' ou um arquivo .pth/.safetensors."
        )
    m = MODELS[name_or_path]
    dest = home() / m.filename
    if dest.is_file() and (not m.sha256 or _sha256(dest) == m.sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if not quiet:
        print(f"Baixando modelo '{m.key}' ({m.filename})...", file=sys.stderr)
    urllib.request.urlretrieve(m.url, tmp)
    if m.sha256 and _sha256(tmp) != m.sha256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Falha de integridade ao baixar {m.filename}: hash SHA-256 diferente do esperado.")
    tmp.replace(dest)
    return dest
