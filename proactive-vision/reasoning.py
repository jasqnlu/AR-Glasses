"""Qwen3-VL via mlx-vlm, plus the single MLX worker thread.

All MLX work (VLM load/generate AND Whisper) runs on ONE thread: MLX binds GPU streams to
threads, and serialising also enforces rule 3 (one VLM call at a time; extra triggers dropped).
"""
import os
import queue
import tempfile
import threading
import time

import cv2
import numpy as np

import config

PROACTIVE_PROMPT = ("You are a wearable assistant. In under 15 words, say the single most useful "
                    "thing about this. If nothing is useful, reply exactly: SKIP.")
QUESTION_PROMPT = "The user asks: '{q}'. Recent memory: {rows}. Answer in under 15 words."


class MLXWorker:
    def __init__(self):
        self.q = queue.Queue()
        self.vlm_pending = threading.Event()   # set while a VLM job is queued or running
        threading.Thread(target=self._loop, daemon=True, name="mlx").start()

    def _loop(self):
        while True:
            fn = self.q.get()
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f"[mlx] job error: {e!r}")

    def submit(self, fn):
        self.q.put(fn)

    def call(self, fn):
        """Run fn on the worker and block until done (used for startup load/warm-up)."""
        box, done = {}, threading.Event()

        def job():
            try:
                box["v"] = fn()
            except Exception as e:  # noqa: BLE001
                box["e"] = e
            done.set()
        self.submit(job)
        done.wait()
        if "e" in box:
            raise box["e"]
        return box.get("v")


class Reasoner:
    def __init__(self, worker: MLXWorker, model_id=config.VLM_MODEL):
        self.w = worker
        self._tmp = os.path.join(tempfile.gettempdir(), "pva_vlm_input.jpg")
        self.last_ms = 0.0
        self.w.call(lambda: self._load(model_id))

    def _load(self, model_id):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        t0 = time.time()
        try:
            self.model, self.processor = load(model_id)
            self.model_id = model_id
        except Exception as e:  # noqa: BLE001
            print(f"[vlm] {model_id} failed ({e}); falling back to {config.VLM_FALLBACK}")
            self.model, self.processor = load(config.VLM_FALLBACK)
            self.model_id = config.VLM_FALLBACK
        self.cfg = load_config(self.model_id)
        print(f"[vlm] loaded {self.model_id} in {time.time()-t0:.1f}s")

    def warmup(self):
        out = self.w.call(lambda: self._run(np.full((256, 256, 3), 128, np.uint8), "Say OK."))
        print(f"[vlm] warm-up -> {out!r} in {self.last_ms:.0f} ms")

    def _run(self, image_bgr, prompt):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        h, w = image_bgr.shape[:2]
        s = config.VLM_IMG_MAX / max(h, w)
        if s < 1:
            image_bgr = cv2.resize(image_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(self._tmp, image_bgr)
        formatted = apply_chat_template(self.processor, self.cfg, prompt, num_images=1)
        t0 = time.perf_counter()
        out = generate(self.model, self.processor, formatted, [self._tmp],
                       max_tokens=config.VLM_MAX_TOKENS, temperature=0.0, verbose=False)
        self.last_ms = (time.perf_counter() - t0) * 1000
        text = getattr(out, "text", out)  # newer mlx-vlm returns a GenerationResult
        return text.strip().strip('"').strip()

    def busy(self):
        return self.w.vlm_pending.is_set()

    def ask(self, image_bgr, prompt, on_done):
        """Queue a VLM call. Returns False (dropped, not queued) if one is already pending."""
        if self.w.vlm_pending.is_set():
            return False
        self.w.vlm_pending.set()
        img = image_bgr.copy()

        def job():
            try:
                res = self._run(img, prompt)
            except Exception as e:  # noqa: BLE001
                print(f"[vlm] error: {e!r}")
                res = None
            finally:
                self.w.vlm_pending.clear()
            on_done(res)
        self.w.submit(job)
        return True

    def proactive(self, crop, on_done):
        return self.ask(crop, PROACTIVE_PROMPT, on_done)

    def question(self, frame, q, rows, on_done):
        return self.ask(frame, QUESTION_PROMPT.format(q=q, rows="; ".join(rows) or "none"), on_done)


def is_skip(text):
    return not text or "SKIP" in text.upper()[:12]
