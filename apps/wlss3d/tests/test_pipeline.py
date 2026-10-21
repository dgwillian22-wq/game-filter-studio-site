import subprocess

import cv2
import numpy as np
import pytest

from wlss3d.depth import DepthSmoother, FallbackDepth
from wlss3d.render import Renderer, fit_viewport, make_context, view_at


@pytest.fixture(scope="module")
def gl():
    ctx = make_context()
    return ctx, Renderer(ctx)


def render(gl, frame, depth, mode, view=0.0, strength=0.05, conv=0.5):
    ctx, r = gl
    h, w = frame.shape[:2]
    fbo = ctx.simple_framebuffer((w, h), components=3)
    r.upload_frame(frame)
    r.upload_depth(depth)
    fbo.clear(0, 0, 0, 1)
    r.draw(fbo, (0, 0, w, h), mode, strength, conv, view)
    return np.frombuffer(fbo.read(components=3), np.uint8).reshape(h, w, 3)[::-1]


def scene():
    """Fundo cinza com um quadrado vermelho (BGR) no meio; o quadrado esta perto."""
    frame = np.full((64, 128, 3), 90, np.uint8)
    frame[20:44, 52:76] = (0, 0, 255)
    depth = np.full((64, 128), 0.2, np.float32)
    depth[20:44, 52:76] = 0.9
    return frame, depth


def red_columns(img):
    mask = (img[..., 0] > 200) & (img[..., 1] < 60)
    xs = np.where(mask.any(axis=0))[0]
    return xs.min(), xs.max()


def red_center(img):
    mask = (img[..., 0] > 200) & (img[..., 1] < 60)
    return np.where(mask)[1].mean()


def test_original_keeps_orientation_and_color(gl):
    frame, depth = scene()
    out = render(gl, frame, depth, "original")
    assert (out == 0).all(axis=2).sum() == 0  # cobre a tela inteira
    assert red_columns(out) == (52, 75)
    assert out[2, 2].tolist() == [90, 90, 90]
    assert out[30, 60, 0] > 200  # RGB: vermelho


def test_pisca_views_move_near_object_in_opposite_directions(gl):
    frame, depth = scene()
    assert (render(gl, frame, depth, "pisca", view=-1.0) == 0).all(axis=2).sum() == 0
    left = red_columns(render(gl, frame, depth, "pisca", view=-1.0))
    right = red_columns(render(gl, frame, depth, "pisca", view=1.0))
    assert left[0] > right[0]  # olho esquerdo ve o objeto perto mais a direita
    # deslocamento maximo: 2 * 0.05 * (0.9 - 0.5) * 128 px = 5,12 px; as bordas do objeto se
    # esticam (parede de heightfield) e o centro anda um pouco menos que isso
    assert 2.5 <= red_center(render(gl, frame, depth, "pisca", view=-1.0)) - red_center(
        render(gl, frame, depth, "pisca", view=1.0)) <= 6.5


def test_object_at_convergence_does_not_move(gl):
    frame, depth = scene()
    depth[:] = 0.5
    a = render(gl, frame, depth, "pisca", view=-1.0)
    b = render(gl, frame, depth, "pisca", view=1.0)
    assert np.array_equal(a, b)


def test_sbs_halves_match_eye_views(gl):
    frame, depth = scene()
    sbs = render(gl, frame, depth, "sbs")
    half = sbs.shape[1] // 2
    assert not np.array_equal(sbs[:, :half], sbs[:, half:])


def test_anaglyph_channels(gl):
    frame, depth = scene()
    out = render(gl, frame, depth, "anaglifo")
    assert out[..., 0].astype(int).std() > 0 and not np.array_equal(out[..., 0], out[..., 1])


def test_depth_view_shows_depth(gl):
    frame, depth = scene()
    out = render(gl, frame, depth, "profundidade")
    assert out[30, 60, 0] > out[2, 2, 0] + 100


def test_view_at():
    assert view_at("pisca", 0.0, 3.0) == -1.0
    assert view_at("pisca", 1 / 6 + 0.01, 3.0) == 1.0
    assert abs(view_at("vaievem", 1 / 12, 3.0) - 1.0) < 1e-6
    assert view_at("sbs", 5.0, 3.0) == 0.0


def test_fit_viewport_letterbox():
    assert fit_viewport(1000, 1000, 1600, 900) == (0, 219, 1000, 562)


def test_fallback_depth_range_and_smoother():
    frame = np.random.randint(0, 255, (72, 128, 3), np.uint8)
    d = FallbackDepth().infer(frame)
    assert d.dtype == np.float32 and 0.0 <= d.min() and d.max() <= 1.0 and d.std() > 0
    s = DepthSmoother()
    first = s(d, frame)
    second = s(np.zeros_like(d), frame)  # mesma cena: mistura, nao salta direto
    assert second.mean() > 0 and second.mean() < first.mean()
    cut = s(np.zeros_like(d), np.full_like(frame, 255))  # corte de cena: reinicia
    assert cut.mean() == 0


def test_export_roundtrip(tmp_path):
    from wlss3d.export import export_video

    src = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    dst = tmp_path / "out.mp4"
    n = export_video(src, dst, FallbackDepth(), mode="pisca", hz=2.0)
    assert n == 10
    cap = cv2.VideoCapture(str(dst))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    ok, a = cap.read()
    for _ in range(3):
        ok, b = cap.read()
    assert ok and not np.array_equal(a, b)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                            "stream=codec_name", "-of", "csv=p=0", str(dst)], capture_output=True, text=True)
    assert "aac" in probe.stdout


def test_decoder_and_audio_clock(tmp_path):
    from wlss3d.source import AudioPlayer, VideoDecoder

    src = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    dec = VideoDecoder(src)
    import time

    time.sleep(0.5)
    item = dec.get(0.0)
    assert item and item[0] == 0.0 and item[1].shape == (90, 160, 3)
    dec.seek(1.0)
    time.sleep(0.5)
    item = dec.get(1.05)
    assert item and abs(item[0] - 1.0) < 0.21
    dec.close()

    a = AudioPlayer(src, start=0.5, device=False)
    assert a.ready.wait(5) and a.has_audio
    a.resume()
    got = a.pull(4800)
    assert len(got) == 4800 * 4 and any(got)
    assert abs(a.position() - (0.5 + 0.1)) < 1e-6
    a.close()

    silent = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)], check=True)
    s = AudioPlayer(silent, device=False)
    assert s.ready.wait(5) and not s.has_audio
    s.close()
