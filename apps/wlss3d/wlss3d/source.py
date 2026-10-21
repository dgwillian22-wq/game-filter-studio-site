"""Decodificacao de video (thread propria) e audio (ffmpeg -> sounddevice) com relogio mestre."""
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2


def find_ffmpeg():
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for base in (getattr(sys, "_MEIPASS", None), Path(sys.executable).parent):
        if base and (Path(base) / exe).exists():
            return str(Path(base) / exe)
    return shutil.which("ffmpeg")


class VideoDecoder(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise IOError(f"Nao consegui abrir o video: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration = n / self.fps if n and n > 0 else 0.0
        self.q = queue.Queue(maxsize=6)
        self.eof = False
        self._gen = 0
        self._seek = None
        self._idx = 0
        self._held = None
        self._lock = threading.Lock()
        self._stop = False
        self.start()

    def seek(self, t):
        with self._lock:
            self._gen += 1
            self._seek = max(0.0, t)
            self._held = None
            self.eof = False
        self._drain()

    def _drain(self):
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    def close(self):
        self._stop = True

    def run(self):
        while not self._stop:
            with self._lock:
                req, self._seek = self._seek, None
                gen = self._gen
            if req is not None:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, req * 1000.0)
                self._idx = int(round(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
            ok, frame = self.cap.read()
            if not ok:
                self.eof = True
                time.sleep(0.01)
                continue
            item = (self._idx / self.fps, frame, gen)
            self._idx += 1
            while not self._stop:
                try:
                    self.q.put(item, timeout=0.05)
                    break
                except queue.Full:
                    if gen != self._gen:
                        break
        self.cap.release()

    def get(self, t_now):
        """Devolve (t, frame) do quadro que deve estar na tela agora (descarta os atrasados)."""
        latest = None
        slack = 0.5 / self.fps
        while True:
            if self._held is None:
                try:
                    self._held = self.q.get_nowait()
                except queue.Empty:
                    break
            if self._held[2] != self._gen:
                self._held = None
                continue
            if self._held[0] <= t_now + slack:
                latest, self._held = self._held, None
            else:
                break
        return (latest[0], latest[1]) if latest else None

    @property
    def at_end(self):
        return self.eof and self.q.empty() and self._held is None


class AudioPlayer:
    """Toca o audio do arquivo via ffmpeg e serve de relogio (posicao = amostras ja tocadas)."""

    SR = 48000

    def __init__(self, path, start=0.0, device=True):
        self.start = start
        self.has_audio = True
        self.ready = threading.Event()
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._played = 0
        self._paused = True
        self._stop = False
        self._eof = False
        self._proc = subprocess.Popen(
            [find_ffmpeg(), "-v", "quiet", "-ss", f"{start:.3f}", "-i", str(path), "-vn",
             "-f", "s16le", "-ac", "2", "-ar", str(self.SR), "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        threading.Thread(target=self._read, daemon=True).start()
        self._stream = None
        if device:
            import sounddevice as sd

            self._stream = sd.RawOutputStream(samplerate=self.SR, channels=2, dtype="int16",
                                              blocksize=1024, callback=self._callback)
            self._stream.start()

    def _read(self):
        total = 0
        prime = int(0.3 * self.SR) * 4
        cap = 5 * self.SR * 4
        while not self._stop:
            with self._lock:
                full = len(self._buf) > cap
            if full:
                time.sleep(0.02)
                continue
            chunk = self._proc.stdout.read(65536)
            if not chunk:
                break
            total += len(chunk)
            with self._lock:
                self._buf += chunk
                if len(self._buf) >= prime:
                    self.ready.set()
        self._eof = True
        if total == 0:
            self.has_audio = False
        self.ready.set()

    def _callback(self, outdata, frames, _time, _status):
        n = frames * 4
        chunk = b""
        with self._lock:
            if not self._paused:
                chunk = bytes(self._buf[:n])
                del self._buf[:len(chunk)]
                self._played += len(chunk) // 4
        outdata[:n] = chunk + b"\x00" * (n - len(chunk))

    def pull(self, frames):
        """Mesmo caminho do callback, para testes sem placa de som."""
        out = bytearray(frames * 4)
        self._callback(out, frames, None, None)
        return bytes(out)

    def position(self):
        return self.start + self._played / self.SR

    def resume(self):
        self._paused = False

    def pause(self):
        self._paused = True

    def close(self):
        self._stop = True
        self._proc.kill()
        if self._stream:
            self._stream.close()


class Playback:
    """Relogio unico do player: audio quando existe, senao tempo de parede."""

    def __init__(self, path, use_audio=True):
        self.path = path
        self.decoder = VideoDecoder(path)
        self.use_audio = bool(use_audio and find_ffmpeg())
        self.audio = None
        self._playing = True
        self._t0 = 0.0
        self._w0 = None
        self._open_audio(0.0)

    def _open_audio(self, start):
        if self.audio:
            self.audio.close()
            self.audio = None
        if self.use_audio:
            try:
                self.audio = AudioPlayer(self.path, start)
            except Exception as e:  # noqa: BLE001
                print(f"[aviso] sem audio ({e})")
                self.use_audio = False

    @property
    def playing(self):
        return self._playing

    def time(self):
        if not self._playing or self._w0 is None:
            return self._t0
        if self.audio and self.audio.has_audio:
            return self.audio.position()
        return self._t0 + time.perf_counter() - self._w0

    def update(self):
        """Chamar a cada quadro: so inicia o relogio quando o audio tem buffer."""
        if self._playing and self._w0 is None and (self.audio is None or self.audio.ready.is_set()):
            self._w0 = time.perf_counter()
            if self.audio:
                self.audio.resume()

    def toggle(self):
        if self._playing:
            self._t0 = self.time()
            self._playing, self._w0 = False, None
            if self.audio:
                self.audio.pause()
        else:
            self._playing = True

    def seek(self, t):
        if self.decoder.duration:
            t = min(max(0.0, t), max(0.0, self.decoder.duration - 0.1))
        t = max(0.0, t)
        self._t0, self._w0 = t, None
        self.decoder.seek(t)
        self._open_audio(t)

    def seek_by(self, dt):
        self.seek(self.time() + dt)

    def close(self):
        self.decoder.close()
        if self.audio:
            self.audio.close()
