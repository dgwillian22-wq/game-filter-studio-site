"""Interface grafica local (so biblioteca padrao): ``python -m texai serve``."""

from __future__ import annotations

import io
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

from .backends import get_backend
from .models import MODELS
from .pipeline import Options, Skip, process_image

MAX_BYTES = 64 * 1024 * 1024
MAX_SIDE = 4096
UI = Path(__file__).with_name("ui") / "index.html"

_lock = threading.Lock()       # uma imagem por vez (CPU/GPU)
_backends: dict[tuple, object] = {}


def _backend(model: str):
    key = (model,)
    if key not in _backends:
        _backends[key] = get_backend(model)
    return _backends[key]


def run(data: bytes, q: dict[str, str]) -> bytes:
    scale = int(q.get("scale", 4))
    if scale not in (1, 2, 3, 4, 6, 8):
        raise ValueError("escala invalida")
    model = q.get("model", "general")
    if model != "classic" and model not in MODELS:
        raise ValueError("modelo invalido")
    kind = q.get("kind", "color")
    if kind not in ("color", "normal", "data"):
        raise ValueError("tipo invalido")
    opts = Options(
        scale=scale,
        strength=min(1.0, max(0.0, float(q.get("strength", 1)))),
        tileable=q.get("tileable") == "1",
        sharpen=min(2.0, max(0.0, float(q.get("sharpen", 0)))),
        data_maps="classic",
    )
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        if max(im.size) > MAX_SIDE:
            raise ValueError(f"imagem muito grande para o modo web (maximo {MAX_SIDE}px); use a linha de comando")
        with _lock:
            out = process_image(im, kind, opts, _backend(model))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = "TexAI"

    def log_message(self, fmt, *args):  # silencioso
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        elif path == "/api/info":
            self._json(200, {
                "models": [{"key": "classic", "desc": "Sem IA (Lanczos + nitidez). Instantaneo."}]
                + [{"key": m.key, "desc": m.desc} for m in MODELS.values()],
            })
        else:
            self._json(404, {"error": "nao encontrado"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/process":
            return self._json(404, {"error": "nao encontrado"})
        # So aceita chamadas da propria pagina (evita CSRF de outros sites).
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != host:
            return self._json(403, {"error": "origem nao permitida"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if not 0 < n <= MAX_BYTES:
                raise ValueError("arquivo vazio ou maior que 64 MB")
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            self._send(200, run(self.rfile.read(n), q), "image/png")
        except Skip:
            self._json(400, {"error": "ignorado"})
        except (ValueError, OSError) as e:
            self._json(400, {"error": str(e)})
        except SystemExit as e:  # modelo ausente / dependencia faltando
            self._json(500, {"error": str(e)})
        except Exception as e:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"TexAI rodando em {url}  (Ctrl+C para sair)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
