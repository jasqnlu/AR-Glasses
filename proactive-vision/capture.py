"""Camera / video capture into a single latest-frame slot (never queued)."""
import threading
import time

import cv2

import config


class Capture:
    def __init__(self, source):
        self.is_file = not str(source).isdigit()
        self.source = source if self.is_file else int(source)
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source {source!r}")
        if not self.is_file:
            self._set_size(*config.CAPTURE_SIZE) or self._set_size(*config.CAPTURE_FALLBACK_SIZE)
        self.file_fps = (self.cap.get(cv2.CAP_PROP_FPS) or 30.0) if self.is_file else None
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self.running = True
        self.ended = False
        threading.Thread(target=self._loop, daemon=True, name="capture").start()

    def _set_size(self, w, h):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == w

    def _loop(self):
        if not self.is_file:  # first frames are dark while auto-exposure settles
            for _ in range(config.WARMUP_FRAMES):
                self.cap.read()
        period = 1.0 / self.file_fps if self.is_file else 0
        while self.running:
            t0 = time.time()
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:
                    self.ended = True
                    break
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1
            if period:  # play files in real time so dwell/age math matches reality
                time.sleep(max(0.0, period - (time.time() - t0)))

    def latest(self):
        """Return (seq, frame). seq lets consumers skip frames they've already processed."""
        with self._lock:
            return self._seq, self._frame

    def stop(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()
