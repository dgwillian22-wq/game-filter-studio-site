"""Estimativa de profundidade por video: modelo ONNX na GPU (DirectML/CUDA) com fallback."""
import threading
import time

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _normalize(d, state):
    """Normaliza 0..1 (1 = perto) com limites suavizados no tempo para nao piscar."""
    lo, hi = np.percentile(d, (2, 98))
    if state.get("lo") is None:
        state["lo"], state["hi"] = lo, hi
    else:
        state["lo"] += 0.15 * (lo - state["lo"])
        state["hi"] += 0.15 * (hi - state["hi"])
    return np.clip((d - state["lo"]) / (state["hi"] - state["lo"] + 1e-6), 0.0, 1.0).astype(np.float32)


class DepthModel:
    """Depth Anything V2 via ONNX Runtime."""

    def __init__(self, path, long_side=308, invert=False):
        import onnxruntime as ort

        wanted = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in wanted if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.provider = self.session.get_providers()[0].replace("ExecutionProvider", "")
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.fp16 = "float16" in inp.type
        self.fixed = tuple(inp.shape[2:]) if all(isinstance(s, int) for s in inp.shape[2:]) else None
        self.long_side = long_side
        self.invert = invert
        self._norm = {}

    def _size(self, h, w):
        if self.fixed:
            return self.fixed[1], self.fixed[0]
        s = self.long_side / max(h, w)
        return max(56, round(w * s / 14) * 14), max(56, round(h * s / 14) * 14)

    def infer(self, bgr):
        w, h = self._size(*bgr.shape[:2])
        x = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = ((x.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        if self.fp16:
            x = x.astype(np.float16)
        d = np.squeeze(self.session.run(None, {self.input_name: x})[0]).astype(np.float32)
        d = _normalize(d, self._norm)
        return 1.0 - d if self.invert else d


class FallbackDepth:
    """Profundidade aproximada (sem IA): parte de baixo mais perto + brilho. Usada se nao ha modelo."""

    provider = "Sem IA (demo)"

    def __init__(self, invert=False):
        self.invert = invert
        self._norm = {}

    def infer(self, bgr):
        h, w = bgr.shape[:2]
        sw, sh = 192, max(8, int(192 * h / w))
        small = cv2.GaussianBlur(cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA), (0, 0), 3)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        ramp = np.linspace(0.0, 1.0, sh, dtype=np.float32)[:, None]
        d = _normalize(0.7 * ramp + 0.3 * gray, self._norm)
        return 1.0 - d if self.invert else d


class DepthSmoother:
    """Mistura no tempo; reinicia em corte de cena."""

    def __init__(self, alpha=0.6, cut=0.18):
        self.alpha, self.cut = alpha, cut
        self.prev = None
        self.prev_thumb = None

    def __call__(self, depth, frame):
        thumb = cv2.cvtColor(cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        thumb = thumb.astype(np.float32) / 255.0
        is_cut = self.prev_thumb is not None and float(np.abs(thumb - self.prev_thumb).mean()) > self.cut
        self.prev_thumb = thumb
        if self.prev is None or self.prev.shape != depth.shape or is_cut:
            self.prev = depth
        else:
            self.prev = self.alpha * depth + (1.0 - self.alpha) * self.prev
        return self.prev


class AsyncDepth:
    """Roda a IA em outra thread; o player sempre usa o ultimo mapa pronto (sem travar o video)."""

    def __init__(self, model):
        self.model = model
        self.smoother = DepthSmoother()
        self.version = 0
        self.depth = None
        self.ms = 0.0
        self._frame = None
        self._ev = threading.Event()
        self._lock = threading.Lock()
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, frame):
        with self._lock:
            self._frame = frame
        self._ev.set()

    def latest(self):
        with self._lock:
            return self.version, self.depth

    def stop(self):
        self._stop = True
        self._ev.set()

    def _run(self):
        while not self._stop:
            self._ev.wait()
            self._ev.clear()
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                continue
            t0 = time.perf_counter()
            depth = self.smoother(self.model.infer(frame), frame)
            self.ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self.depth = depth
                self.version += 1


def load_depth_model(path=None, long_side=308, invert=False, progress=None, log=print):
    """Tenta o modelo ONNX; se nao der (sem internet, sem onnxruntime), usa o fallback sem IA."""
    try:
        if path is None:
            from .models import ensure_model

            path = ensure_model(progress=progress)
        return DepthModel(path, long_side, invert)
    except Exception as e:  # noqa: BLE001
        log(f"[aviso] modelo de IA indisponivel ({e}); usando profundidade aproximada sem IA")
        return FallbackDepth(invert)
