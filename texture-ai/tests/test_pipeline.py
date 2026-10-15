import io
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from texai import cli, web
from texai.backends import ClassicBackend
from texai.pipeline import Options, Skip, bleed_colors, process_image, texture_kind

B = ClassicBackend()


def noise(h=32, w=32, mode="RGB", seed=0):
    rng = np.random.default_rng(seed)
    ch = 4 if mode == "RGBA" else 3
    return Image.fromarray(rng.integers(0, 256, (h, w, ch), dtype=np.uint8), mode)


@pytest.mark.parametrize("name,kind", [
    ("rocha.png", "color"), ("rocha_n.png", "normal"), ("rocha_normal.tga", "normal"),
    ("rocha_spec.png", "data"), ("rocha_mask.png", "data"), ("canyon.png", "color"),
    ("wall_nm.png", "normal"), ("tronco.png", "color"),
])
def test_texture_kind(name, kind):
    assert texture_kind(name) == kind


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_scale(scale):
    out = process_image(noise(20, 30), "color", Options(scale=scale), B)
    assert out.size == (30 * scale, 20 * scale)


def test_alpha_preserved_and_no_dark_halo():
    img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for y in range(4, 12):
        for x in range(4, 12):
            img.putpixel((x, y), (255, 200, 0, 255))
    out = process_image(img, "color", Options(scale=4), B)
    assert out.mode == "RGBA" and out.size == (64, 64)
    a = np.asarray(out)
    edge = (a[..., 3] > 0) & (a[..., 3] < 255)
    assert edge.any()
    # as bordas semi-transparentes mantem a cor da textura (sem vazar preto)
    assert a[edge][:, :3].min(axis=0)[0] > 200


def test_bleed_colors():
    rgb = np.zeros((5, 5, 3), np.uint8)
    rgb[2, 2] = (10, 20, 30)
    alpha = np.zeros((5, 5), np.uint8)
    alpha[2, 2] = 255
    out = bleed_colors(rgb, alpha)
    assert (out == (10, 20, 30)).all()


def test_tileable_wraps_seamlessly():
    # gradiente periodico: a emenda esquerda/direita nao pode virar degrau
    x = np.arange(32)
    row = (127 + 120 * np.sin(2 * np.pi * x / 32)).astype(np.uint8)
    arr = np.stack([np.tile(row, (32, 1))] * 3, -1)
    img = Image.fromarray(arr)
    seam = lambda o: np.abs(np.asarray(o).astype(int)[:, 0] - np.asarray(o).astype(int)[:, -1]).mean()
    tiled = process_image(img, "color", Options(scale=4, tileable=True), B)
    plain = process_image(img, "color", Options(scale=4, tileable=False), B)
    assert tiled.size == plain.size == (128, 128)
    assert seam(tiled) <= seam(plain)
    assert seam(tiled) < 8


def test_normal_map_stays_unit_length():
    v = np.zeros((16, 16, 3), np.float32)
    v[..., 0], v[..., 2] = 0.6, 0.8
    arr = ((v + 1) * 127.5).astype(np.uint8)
    out = np.asarray(process_image(Image.fromarray(arr), "normal", Options(scale=2), B)).astype(np.float32)
    n = np.linalg.norm(out / 127.5 - 1, axis=-1)
    assert abs(n.mean() - 1) < 0.03


def test_data_maps_skip():
    with pytest.raises(Skip):
        process_image(noise(), "normal", Options(data_maps="skip"), B)


def test_gray_and_palette_modes():
    g = Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16), "L")
    assert process_image(g, "color", Options(scale=2), B).mode == "L"
    p = noise(8, 8).convert("P")
    assert process_image(p, "color", Options(scale=2), B).mode == "RGB"
    la = Image.merge("LA", (g, g))
    assert process_image(la, "color", Options(scale=2), B).mode == "LA"


def test_strength_zero_equals_plain_resize():
    img = noise(16, 16)
    out = process_image(img, "color", Options(scale=2, strength=0.0), B)
    ref = img.resize((32, 32), Image.LANCZOS)
    assert np.array_equal(np.asarray(out), np.asarray(ref))


def test_cli_batch_preserves_tree_and_formats(tmp_path):
    src = tmp_path / "jogo"
    (src / "mapa").mkdir(parents=True)
    noise().save(src / "a.png")
    noise().save(src / "mapa" / "b.bmp")
    noise(8, 8).save(src / "mapa" / "b_n.png")
    (src / "leia.txt").write_text("x")
    (src / "ruim.png").write_bytes(b"nao sou png")
    rc = cli.main(["upscale", str(src), "-m", "classic", "-s", "2", "--data-maps", "skip"])
    out = tmp_path / "jogo_ai"
    assert rc == 1                              # 1 arquivo com erro nao derruba o lote
    assert Image.open(out / "a.png").size == (64, 64)
    assert Image.open(out / "mapa" / "b.png").size == (64, 64)
    assert not (out / "mapa" / "b_n.png").exists()
    assert not (out / "leia.txt").exists()
    # rodar de novo nao sobrescreve
    assert cli.main(["upscale", str(src), "-m", "classic", "-s", "2", "--data-maps", "skip"]) == 1


def test_cli_single_file_and_tga_dds(tmp_path):
    f = tmp_path / "x.png"
    noise(16, 16, "RGBA").save(f)
    assert cli.main(["upscale", str(f), "-m", "classic", "-s", "2"]) == 0
    assert Image.open(tmp_path / "x_ai.png").size == (32, 32)
    for fmt in ("tga", "dds"):
        assert cli.main(["upscale", str(f), "-m", "classic", "-s", "2", "--format", fmt, "-o", str(tmp_path / fmt)]) == 0
        assert Image.open(tmp_path / fmt / f"x.{fmt}").size == (32, 32)


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_web_api(server):
    assert b"TexAI" in urllib.request.urlopen(server + "/").read()
    info = json.load(urllib.request.urlopen(server + "/api/info"))
    assert {m["key"] for m in info["models"]} >= {"classic", "general"}
    buf = io.BytesIO()
    noise(10, 10).save(buf, "PNG")
    req = urllib.request.Request(server + "/api/process?model=classic&scale=2", buf.getvalue(), method="POST")
    out = Image.open(io.BytesIO(urllib.request.urlopen(req).read()))
    assert out.size == (20, 20)


def test_web_rejects_bad_input_and_foreign_origin(server):
    def post(path, body, headers=None):
        req = urllib.request.Request(server + path, body, headers=headers or {}, method="POST")
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            return e.code
    assert post("/api/process?model=classic", b"lixo") == 400
    assert post("/api/process?model=../../etc/passwd", b"x") == 400
    buf = io.BytesIO()
    noise(8, 8).save(buf, "PNG")
    assert post("/api/process?model=classic", buf.getvalue(), {"Origin": "https://evil.example"}) == 403
