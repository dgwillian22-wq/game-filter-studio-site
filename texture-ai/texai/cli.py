"""Linha de comando: ``python -m texai upscale|models|serve``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

from .models import MODELS
from .pipeline import INPUT_EXTS, Options, Skip, process_image, texture_kind

FORMATS = {"png": ".png", "tga": ".tga", "bmp": ".bmp", "jpg": ".jpg", "dds": ".dds", "webp": ".webp"}


def save_image(img: Image.Image, path: Path) -> None:
    ext = path.suffix.lower()
    kw: dict = {}
    if ext == ".jpg":
        img = img.convert("RGB")
        kw = {"quality": 95, "subsampling": 0}
    elif ext == ".bmp" and img.mode in ("RGBA", "LA"):
        img = img.convert("RGB")
    elif ext == ".dds":
        kw = {"pixel_format": "DXT5" if "A" in img.getbands() else "DXT1"}
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    elif ext == ".webp":
        kw = {"lossless": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, **kw)


def collect(src: Path) -> list[tuple[Path, Path]]:
    """[(arquivo, caminho relativo a ``src``)] respeitando subpastas."""
    if src.is_file():
        return [(src, Path(src.name))]
    files = [p for p in sorted(src.rglob("*")) if p.is_file() and p.suffix.lower() in INPUT_EXTS]
    return [(p, p.relative_to(src)) for p in files]


def cmd_upscale(a: argparse.Namespace) -> int:
    from .backends import get_backend

    src = Path(a.input)
    if not src.exists():
        print(f"Nao encontrei: {src}", file=sys.stderr)
        return 2
    items = collect(src)
    if not items:
        print("Nenhuma textura encontrada (png, jpg, bmp, tga, dds, pcx, tif, webp...).", file=sys.stderr)
        return 2

    if a.output:
        out_root = Path(a.output)
    else:
        out_root = src.with_name(src.name + "_ai") if src.is_dir() else src.parent
    suffix = "_ai" if (src.is_file() and not a.output) else ""
    out_root_resolved = out_root.resolve()

    opts = Options(scale=a.scale, strength=a.strength, tileable=a.tileable,
                   sharpen=a.sharpen, data_maps=a.data_maps)
    backend = get_backend(a.model, tile=a.tile, device=a.device)
    print(f"Motor: {backend.name} | escala: {a.scale}x | {len(items)} arquivo(s)")

    ok = skipped = failed = 0
    t0 = time.time()
    for i, (path, rel) in enumerate(items, 1):
        if path.resolve().is_relative_to(out_root_resolved) and src.is_dir():
            continue
        ext = rel.suffix if a.format == "same" else FORMATS[a.format]
        dest = out_root / rel.with_name(rel.stem + suffix + ext)
        tag = f"[{i}/{len(items)}] {rel}"
        if dest.exists() and not a.overwrite:
            print(f"{tag}: ja existe, pulando (use --overwrite)")
            skipped += 1
            continue
        try:
            with Image.open(path) as im:
                im.load()
                kind = texture_kind(path)
                res = process_image(im, kind, opts, backend)
            save_image(res, dest)
            print(f"{tag}: {im.width}x{im.height} -> {res.width}x{res.height} ({kind})")
            ok += 1
        except Skip:
            print(f"{tag}: mapa de dados, ignorado")
            skipped += 1
        except Exception as e:  # um arquivo ruim nao derruba o lote
            print(f"{tag}: ERRO {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
    print(f"Pronto em {time.time() - t0:.1f}s: {ok} ok, {skipped} ignorados, {failed} com erro. Saida: {out_root}")
    return 1 if failed else 0


def cmd_models(_: argparse.Namespace) -> int:
    from .models import ensure_model

    print("classic  Sem IA (Lanczos + nitidez). Nao precisa de download.")
    for m in MODELS.values():
        print(f"{m.key:<8} {m.desc}")
    print("\nTambem aceita --model caminho/do/modelo.pth (ex.: modelos do OpenModelDB).")
    if _.download:
        for key in MODELS:
            print(ensure_model(key))
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from .web import serve

    serve(a.host, a.port, open_browser=not a.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="texai", description="Remasterize texturas de jogos antigos com IA.")
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("upscale", help="aumenta/melhora uma textura ou uma pasta inteira")
    u.add_argument("input", help="arquivo ou pasta de texturas")
    u.add_argument("-o", "--output", help="pasta de saida (padrao: <pasta>_ai ou <arquivo>_ai.png)")
    u.add_argument("-m", "--model", default="general", help="general | photo | anime | classic | arquivo.pth (padrao: general)")
    u.add_argument("-s", "--scale", type=int, default=4, choices=[1, 2, 3, 4, 6, 8], help="fator de escala (padrao: 4)")
    u.add_argument("--strength", type=float, default=1.0, help="0..1: mistura da IA com o upscale simples (padrao: 1)")
    u.add_argument("--tileable", action="store_true", help="textura repetivel: evita emendas nas bordas")
    u.add_argument("--sharpen", type=float, default=0.0, help="nitidez extra no final, ex.: 0.5")
    u.add_argument("--data-maps", choices=["classic", "ai", "skip"], default="classic",
                   help="como tratar normal/spec/mask (_n, _spec, _mask...): classic (padrao), ai ou skip")
    u.add_argument("--format", choices=["same", *FORMATS], default="png", help="formato de saida (padrao: png)")
    u.add_argument("--tile", type=int, default=0, help="tamanho do bloco de processamento (0 = automatico)")
    u.add_argument("--device", choices=["cpu", "cuda", "mps"], help="forca o dispositivo (padrao: automatico)")
    u.add_argument("--overwrite", action="store_true", help="sobrescreve arquivos existentes")
    u.set_defaults(fn=cmd_upscale)

    m = sub.add_parser("models", help="lista os modelos disponiveis")
    m.add_argument("--download", action="store_true", help="baixa todos os modelos agora")
    m.set_defaults(fn=cmd_models)

    s = sub.add_parser("serve", help="abre a interface grafica no navegador")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "strength", 1.0) < 0 or getattr(args, "strength", 1.0) > 1:
        print("--strength precisa estar entre 0 e 1", file=sys.stderr)
        return 2
    return args.fn(args)
