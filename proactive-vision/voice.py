"""Speech out. Non-blocking; a new line interrupts the previous one.

Backends:
  elevenlabs — streams 16 kHz PCM from ElevenLabs and plays chunks as they arrive (~0.3–0.5 s to first audio).
               Only the TEXT of the line leaves the machine; frames and audio never do.
  say        — macOS `say`, fully offline. Automatic fallback if ElevenLabs fails.
"""
import os
import subprocess
import threading

import numpy as np

import config

_lock = threading.Lock()
_gen = 0                 # bumped on every speak(); playback stops when it no longer matches
_say_proc = None
_eleven_ok = True        # turned off for the session on a bad key


def backend():
    if config.TTS_BACKEND == "say":
        return "say"
    if config.TTS_BACKEND in ("auto", "elevenlabs") and os.environ.get("ELEVENLABS_API_KEY") and _eleven_ok:
        return "elevenlabs"
    return "say"


def speak(text):
    global _gen
    if not config.SPEAK or not text:
        return
    with _lock:
        _gen += 1
        gen = _gen
    _stop_say()
    if backend() == "elevenlabs":
        threading.Thread(target=_eleven, args=(text, gen), daemon=True, name="tts").start()
    else:
        _say(text)


def _say(text):
    global _say_proc
    _say_proc = subprocess.Popen(["say", text])


def _stop_say():
    if _say_proc and _say_proc.poll() is None:
        _say_proc.terminate()


def _eleven(text, gen):
    global _eleven_ok
    import requests
    import sounddevice as sd
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.ELEVEN_VOICE_ID}/stream"
    try:
        r = requests.post(url, params={"output_format": "pcm_16000"}, stream=True, timeout=config.ELEVEN_TIMEOUT_S,
                          headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                          json={"text": text, "model_id": config.ELEVEN_MODEL})
        if r.status_code == 401:
            _eleven_ok = False
            raise RuntimeError("ElevenLabs rejected the API key (401); using `say` from now on")
        r.raise_for_status()
        with sd.RawOutputStream(samplerate=16000, channels=1, dtype="int16") as out:
            carry = b""
            for chunk in r.iter_content(chunk_size=3200):       # 0.1 s of audio per chunk
                if gen != _gen:                                 # a newer line started: stop talking
                    r.close()
                    return
                chunk = carry + chunk
                cut = len(chunk) - len(chunk) % 2               # keep int16 samples whole
                carry = chunk[cut:]
                if cut:
                    out.write(np.frombuffer(chunk[:cut], np.int16))
    except Exception as e:  # noqa: BLE001 — never go silent: fall back to `say` for this line
        print(f"[tts] ElevenLabs failed ({e}); falling back to say")
        if gen == _gen:
            _say(text)
