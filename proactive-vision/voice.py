"""Speech out via macOS `say` (non-blocking). A new line interrupts the previous one."""
import subprocess

import config

_proc = None


def speak(text):
    global _proc
    if not config.SPEAK or not text:
        return
    if _proc and _proc.poll() is None:
        _proc.terminate()
    _proc = subprocess.Popen(["say", text])
