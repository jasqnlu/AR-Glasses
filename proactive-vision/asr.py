"""Push-to-talk speech-to-text: start()/stop() records 16 kHz mono; Whisper runs on the MLX worker.

OpenCV windows don't report key-up, so push-to-talk is a toggle: SPACE to start, SPACE to stop.
"""
import numpy as np

import config

SR = 16000


class PushToTalk:
    def __init__(self, worker):
        import sounddevice as sd
        self.sd, self.w = sd, worker
        self.recording = False
        self._chunks, self._stream = [], None

    def warmup(self):
        import mlx_whisper
        self.w.call(lambda: mlx_whisper.transcribe(np.zeros(SR, np.float32),
                                                   path_or_hf_repo=config.WHISPER_MODEL))

    def start(self):
        if self.recording:
            return
        self._chunks = []
        self._stream = self.sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                           callback=lambda d, *_: self._chunks.append(d.copy()))
        self._stream.start()
        self.recording = True

    def stop(self, on_text):
        """Stop recording, transcribe on the MLX worker (queued behind any VLM call), call on_text(str)."""
        if not self.recording:
            return
        self._stream.stop()
        self._stream.close()
        self.recording = False
        audio = np.concatenate(self._chunks)[:, 0] if self._chunks else np.zeros(0, np.float32)
        if len(audio) < SR * 0.3:
            on_text("")
            return

        def job():
            import mlx_whisper
            out = mlx_whisper.transcribe(audio, path_or_hf_repo=config.WHISPER_MODEL, language="en")
            on_text(out.get("text", "").strip())
        self.w.submit(job)
