"""Exporta o video ja convertido em 3D (offline, mesma imagem do player)."""
import subprocess

import cv2
import numpy as np

from .depth import DepthSmoother
from .render import Renderer, make_context, view_at
from .source import find_ffmpeg


def export_video(src, dst, depth_model, mode="pisca", strength=0.04, conv=0.5, hz=3.0, progress=None):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"Nao consegui abrir o video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    ctx = make_context()
    renderer = Renderer(ctx)
    fbo = ctx.simple_framebuffer((w, h), components=3)
    smoother = DepthSmoother()
    ff = subprocess.Popen(
        [find_ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", f"{fps}", "-i", "-", "-i", str(src), "-map", "0:v", "-map", "1:a?",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(dst)],
        stdin=subprocess.PIPE)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            renderer.upload_frame(frame)
            renderer.upload_depth(smoother(depth_model.infer(frame), frame))
            fbo.clear(0, 0, 0, 1)
            renderer.draw(fbo, (0, 0, w, h), mode, strength, conv, view_at(mode, i / fps, hz))
            ff.stdin.write(_flip(fbo.read(components=3), w, h))
            i += 1
            if progress and total:
                progress(i / total)
    finally:
        ff.stdin.close()
        ff.wait()
        cap.release()
    return i


def _flip(raw, w, h):
    """glReadPixels devolve a ultima linha primeiro."""
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)[::-1].tobytes()
