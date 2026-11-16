"""Proactive Vision Assistant — entry point, router, and thread wiring.

  python main.py                     # webcam 0
  python main.py --source clip.mp4   # recorded POV clip (backup demo)
  python main.py --no-vlm --no-asr   # fast iteration on perception + relevance
  python main.py --headless --source clip.mp4 --ask "where's my water bottle?"

Threads: main = HUD + keys | perception = YOLO + memory + relevance | mlx = VLM + Whisper
"""
import argparse
import re
import threading
import time

import cv2

import config
import voice
from capture import Capture
from memory import Memory
from perception import Perception
from relevance import Relevance


# ------------------------------------------------------------------ router
def parse_where(q):
    """'where is my water bottle?' → ('bottle', None) | (None, 'keys') | (None, None)."""
    q = q.lower()
    if not re.search(r"\bwhere\b", q):
        return None, None
    for phrase in sorted(config.SYNONYMS, key=len, reverse=True):   # longest match first
        if re.search(rf"\b{re.escape(phrase)}s?\b", q):
            return config.SYNONYMS[phrase], None
    for thing in config.UNTRACKABLE:
        if re.search(rf"\b{re.escape(thing)}\b", q):
            return None, thing
    return None, None


def best_landmark(labels):
    """Pick the most useful 'near the X' anchor: static furniture/devices first, never people."""
    labels = [l for l in labels if l and l != "person"]
    rank = {l: i for i, l in enumerate(config.LANDMARKS)}
    return min(labels, key=lambda l: rank.get(l, len(rank))) if labels else None


def memory_answer(label, memory, visible_labels):
    if label in visible_labels:
        return f"Your {label} is right in front of you."
    hit = memory.last_seen(label)
    if not hit:
        return f"I haven't seen your {label} yet."
    mins = int(hit["age_s"] // 60)
    when = "less than a minute ago" if mins < 1 else f"about {mins} minute{'s' if mins != 1 else ''} ago"
    near = best_landmark(hit["near"].split(", ") if hit["near"] else [])
    where = f"near the {near}" if near else "where you left it"
    return f"Your {label} was {where}, {when}."


# ------------------------------------------------------------------ app
class App:
    def __init__(self, args):
        self.args = args
        self.t0 = time.time()
        self.mem = Memory(fresh=args.fresh)
        self.per = Perception()
        self.rel = Relevance()
        self.worker = self.reasoner = self.ptt = None
        self.hud = None
        self.tracks, self.scores = [], {}
        self.frame = None
        self.running = True
        self._seq = -1

        if not args.no_vlm or not args.no_asr:
            from reasoning import MLXWorker
            self.worker = MLXWorker()
        if not args.no_vlm:
            from reasoning import Reasoner
            print("[init] loading VLM…")
            self.reasoner = Reasoner(self.worker, config.VLM_MODEL)
            self.reasoner.warmup()
        if not args.no_asr:
            try:
                from asr import PushToTalk
                self.ptt = PushToTalk(self.worker)
                print("[init] warming Whisper…")
                self.ptt.warmup()
            except Exception as e:  # noqa: BLE001 — cut list #2: fall back to typed questions
                print(f"[asr] disabled ({e!r}); use T to type questions")
                self.ptt = None

        # warm YOLO (first MPS call ~2 s), THEN open the source — a clip must not start playing
        # while models load, or its first seconds are silently skipped.
        import numpy as np
        self.per.warmup(np.zeros((480, 640, 3), np.uint8))
        self.cap = Capture(args.source)
        self.log(f"ready (init {time.time()-self.t0:.1f}s)")

    def log(self, msg):
        print(f"[{time.time()-self.t0:6.1f}s] {msg}", flush=True)

    # ---- output
    def say(self, text, kind, trigger):
        self.log(f"SAY ({trigger}) {text}")
        if self.hud:
            self.hud.card(text, kind)
        voice.speak(text)
        self.rel.mark_spoken()
        self.mem.log_utterance(text, trigger)

    def set_status(self, s):
        if self.hud:
            self.hud.status = s

    # ---- perception thread
    def perception_loop(self):
        while self.running:
            seq, frame = self.cap.latest()
            if frame is None or seq == self._seq:     # latest frame only; skip if nothing new
                if self.cap.ended:
                    break
                time.sleep(0.005)
                continue
            self._seq = seq
            events = self.per.step(frame)
            visible = self.per.visible()
            labels = [t.label for t in self.per.current]
            for t in self.per.current:
                t.near.update(labels)
            for ev, tr in events:
                if ev in ("APPEAR", "MOVE"):
                    self.mem.record(tr, labels)
                elif ev == "GONE":   # use what it was seen WITH, not what's in view now
                    self.mem.update_last_seen(tr, tr.near)
                if self.args.verbose:
                    self.log(f"{ev} {tr.label}#{tr.track_id} conf={tr.conf:.2f}")
            self.mem.purge()

            best = self.rel.tick(visible)
            self.tracks, self.scores, self.frame = visible, dict(self.rel.scores), frame
            if best is not None:
                self.nudge(best)

    def nudge(self, track):
        s = self.rel.scores[track.track_id][1]
        if self.reasoner is None:   # --no-vlm: still prove the engine fires
            self.rel.mark_nudged(track.label)
            self.say(f"Noticed your {track.label}.", "nudge", f"relevance:{s:.2f}")
            return
        if self.reasoner.busy():
            return                  # rule 3: dropped, try again next tick
        self.rel.mark_nudged(track.label)
        self.log(f"nudge {track.label}#{track.track_id} S={s:.2f} → VLM")
        self.set_status(f"looking at {track.label}…")
        t0 = time.time()

        def done(text):
            from reasoning import is_skip
            self.set_status("")
            self.log(f"vlm {track.label}: {text!r} ({(time.time()-t0)*1000:.0f} ms)")
            if not is_skip(text):
                self.say(text, "nudge", f"relevance:{track.label}:{s:.2f}")
        self.reasoner.proactive(track.crop, done)

    # ---- questions
    def handle_question(self, q):
        q = q.strip()
        if not q:
            self.set_status("")
            return
        self.log(f"ask {q!r}")
        label, untrackable = parse_where(q)
        if label:
            self.say(memory_answer(label, self.mem, {t.label for t in self.tracks}), "answer", "memory")
        elif untrackable:
            self.say(f"I can't track {untrackable} yet, only common objects.", "answer", "memory")
        elif self.reasoner is None:
            self.say("Visual questions need the VLM enabled.", "answer", "router")
        elif self.frame is not None:
            self.set_status("thinking…")
            ok = self.reasoner.question(self.frame, q, self.mem.recent(3),
                                        lambda a: (self.set_status(""),
                                                   self.say(a or "Sorry, I couldn't tell.", "answer", "vlm")))
            if not ok:
                self.set_status("")
                self.say("One sec, still thinking.", "answer", "busy")

    # ---- main loop (HUD + keys)
    def run(self):
        threading.Thread(target=self.perception_loop, daemon=True, name="perception").start()
        if self.args.headless:
            return self.run_headless()
        from hud import HUD
        self.hud = HUD()
        while self.running:
            seq, frame = self.cap.latest()
            if frame is not None:
                stats = {"yolo_ms": self.per.ms,
                         "vlm_ms": self.reasoner.last_ms if self.reasoner else 0,
                         "quiet_s": time.time() - self.rel.last_spoken if self.rel.last_spoken else 0}
                self.hud.draw(frame, list(self.tracks), dict(self.scores), stats)
            key = cv2.waitKey(30) & 0xFF
            if key != 255:
                self.on_key(key)
            if self.cap.ended and not self.args.loop_hold:
                break
        self.shutdown()

    def on_key(self, key):
        h = self.hud
        if h.typing is not None:                     # typed-question mode
            if key in (13, 10):
                q, h.typing = h.typing, None
                self.handle_question(q)
            elif key == 27:
                h.typing = None
            elif key in (8, 127):
                h.typing = h.typing[:-1]
            elif 32 <= key < 127:
                h.typing += chr(key)
            return
        if key in (ord("q"), 27):
            self.running = False
        elif key == ord("d"):
            h.debug = not h.debug
        elif key == ord("t"):
            h.typing = ""
        elif key == ord(" ") and self.ptt:
            if not self.ptt.recording:
                self.ptt.start()
                self.set_status("● listening… (SPACE to stop)")
            else:
                self.set_status("transcribing…")
                self.ptt.stop(lambda text: (print(f"[asr] {text!r}"), self.handle_question(text)))

    def run_headless(self):
        """Smoke-test path: run the clip, optionally ask a question at the end, print everything."""
        while not self.cap.ended and self.running:
            time.sleep(0.2)
        time.sleep(0.5)
        for q in self.args.ask or []:
            self.handle_question(q)
        if self.reasoner:
            while self.reasoner.busy():
                time.sleep(0.1)
        time.sleep(0.5)
        self.shutdown()

    def shutdown(self):
        self.running = False
        self.cap.stop()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--no-vlm", action="store_true", help="skip Qwen3-VL (template nudges only)")
    ap.add_argument("--no-asr", action="store_true", help="skip Whisper (typed questions only)")
    ap.add_argument("--small", action="store_true", help="use the 2B VLM")
    ap.add_argument("--mute", action="store_true", help="HUD only, no `say`")
    ap.add_argument("--fresh", action="store_true", help="wipe memory.db on start")
    ap.add_argument("--headless", action="store_true", help="no window; for smoke tests on a clip")
    ap.add_argument("--ask", action="append", help="(headless) question(s) to ask after the clip")
    ap.add_argument("--loop-hold", action="store_true", help="keep window open after clip ends")
    ap.add_argument("-v", "--verbose", action="store_true", help="print APPEAR/MOVE/GONE events")
    args = ap.parse_args()
    if args.small:
        config.VLM_MODEL = config.VLM_FALLBACK
    if args.mute:
        config.SPEAK = False
    App(args).run()


if __name__ == "__main__":
    main()
