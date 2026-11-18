"""Proactive Vision Assistant — entry point, router, and thread wiring.

  python main.py                     # webcam 0
  python main.py --source clip.mp4   # recorded POV clip (backup demo)
  python main.py --no-vlm --no-asr   # fast iteration on perception + relevance
  python main.py --headless --source clip.mp4 --ask "where's my water bottle?"

Threads: main = HUD + keys | perception = YOLO + memory + relevance | mlx = VLM + Whisper
"""
import argparse
import os
import re
import threading
import time
from collections import deque

import cv2

import config
import voice
from capture import Capture
from memory import Memory
from perception import Perception
from relevance import Relevance
import rag


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


def _when(age_s):
    mins = int(age_s // 60)
    return "less than a minute ago" if mins < 1 else f"about {mins} minute{'s' if mins != 1 else ''} ago"


def memory_answer(label, memory, visible_labels):
    if label in visible_labels:
        return f"Your {label} is right in front of you."
    hit = memory.last_seen(label)
    m = memory.latest_memory(label)
    if not hit and not m:
        return f"I haven't seen your {label} yet."
    # A "put down here" memory is the best answer, unless the object was seen somewhere else after it.
    if m and (m["place"] or m["relation"]) and (not hit or m["age_s"] <= hit["age_s"] + 5):
        where = m["place"] or m["relation"]
        if m["place"] and m["relation"] and m["relation"].split()[-1] not in m["place"]:
            where += f", {m['relation']}"          # add geometry if the VLM didn't name that landmark
        return f"Your {label} was {where}, {_when(m['age_s'])}."
    when = _when(hit["age_s"])
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
        self.embed = None
        self.captions = deque(maxlen=8)   # (memory_id, label, outlined_frame) waiting for an idle VLM
        self.question_waiting = False     # a user question outranks captions for the VLM
        self._last_remembered = {}        # label → ts of last memory (dedupe twin detections)

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

        if not args.no_rag:
            try:
                print("[init] loading embedder…")
                self.embed = rag.Embedder()
                self.embed("warm up")
            except Exception as e:  # noqa: BLE001 — fall back to recency-only memory
                print(f"[rag] disabled ({e!r})")

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
            # co-occurrence only from confident tracks, so a faint false detection never becomes a landmark
            labels = [t.label for t in self.per.current if t.conf_ema >= config.NUDGE_MIN_CONF]
            for t in self.per.current:
                t.near.update(labels)
            moved = {tr.track_id for ev, tr in events if ev == "MOVE"}
            for tr in self.per.current:       # write on first trusted sighting, then on MOVE
                if tr.conf >= config.MEMORY_MIN_CONF and (not tr.recorded or tr.track_id in moved):
                    self.mem.record(tr, labels)
                    tr.recorded = True
            confident = [t for t in self.per.current if t.conf_ema >= config.NUDGE_MIN_CONF]
            for tr in confident:              # settled personal object → keep a "this is where it is" snapshot
                if (tr.label in config.REMEMBER_LABELS and tr.recorded
                        and time.time() - tr.still_since >= config.SETTLE_S
                        and (tr.snapshot is None or time.time() - tr.snapshot[4] > 0.5)):
                    tr.snapshot = (frame, tr.bbox, tr.poly, rag.relation(tr, confident), time.time())
            for ev, tr in events:
                if ev == "GONE" and tr.recorded:   # use what it was seen WITH, not what's in view now
                    self.mem.update_last_seen(tr, tr.near)
                    if tr.snapshot is not None:
                        self.remember(tr)
                if self.args.verbose:
                    self.log(f"{ev} {tr.label}#{tr.track_id} conf={tr.conf:.2f}")
            self.mem.purge()

            best = self.rel.tick(visible)
            self.tracks, self.scores, self.frame = visible, dict(self.rel.scores), frame
            if best is not None:
                self.nudge(best)
            else:
                self.pump_captions()          # nudges first; captions only when nothing else wants the VLM

    # ---- object memory (RAG)
    def remember(self, tr):
        """Object left view after sitting still: store where it was now, describe it when the VLM is idle."""
        frame, bbox, poly, rel, ts = tr.snapshot
        last = self._last_remembered.get(tr.label, 0)
        self._last_remembered[tr.label] = ts
        if ts - last < 5:              # duplicate detection of the same object — one memory is enough
            return
        text = rag.memory_text(tr.label, None, None, rel)
        vec = self.embed(text) if self.embed else None
        x, y, w, h = bbox
        mem_id = self.mem.add_memory(tr.label, ts, rel, text, vec, frame[max(y, 0):y + h, max(x, 0):x + w])
        self.log(f"remember {tr.label}#{tr.track_id}: {text!r}")
        if self.reasoner is not None:
            self.captions.append((mem_id, tr.label, rel, rag.outlined(frame, bbox, poly)))

    def pump_captions(self):
        if not self.captions or self.reasoner is None or self.reasoner.busy() or self.question_waiting:
            return
        mem_id, label, rel, img = self.captions.popleft()

        def done(out):
            app, place = rag.parse_description(out, label)
            text = rag.memory_text(label, app, place, rel)
            vec = self.embed(text) if self.embed else None
            self.mem.describe_memory(mem_id, app, place, text, vec)
            self.log(f"described {label}: {out!r} -> {text!r}")
        self.reasoner.ask(img, rag.DESCRIBE_PROMPT.format(label=label), done)

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
            self.question_waiting = True
            threading.Thread(target=self._ask_vlm, args=(q, self.frame), daemon=True).start()

    def context_rows(self, q):
        """RAG: the memories most relevant to the question (falls back to most recent sightings)."""
        if self.embed:
            hits = self.mem.search(self.embed(q), config.RAG_K, config.RAG_MIN_SIM)
            if hits:
                self.log("rag " + " | ".join(f"{h['sim']:.2f} {h['text']}" for h in hits))
                return [f"{h['text']} ({_when(h['age_s'])})" for h in hits]
        return self.mem.recent(3)

    def _ask_vlm(self, q, frame):
        rows = self.context_rows(q)
        deadline = time.time() + 8            # wait out a caption / nudge already running (≤ ~1.5 s)
        while self.reasoner.busy() and time.time() < deadline:
            time.sleep(0.05)
        ok = self.reasoner.question(frame, q, rows,
                                    lambda a: (self.set_status(""),
                                               self.say(a or "Sorry, I couldn't tell.", "answer", "vlm")))
        self.question_waiting = False
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
        while self.captions and self.reasoner:   # let pending "where it was" descriptions finish
            self.pump_captions()
            time.sleep(0.1)
        for q in self.args.ask or []:
            self.handle_question(q)
            time.sleep(0.2)
            while self.reasoner and (self.reasoner.busy() or self.question_waiting):
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
    ap.add_argument("--no-rag", action="store_true", help="skip the embedding model (recency-only memory)")
    ap.add_argument("--small", action="store_true", help="use the 2B VLM")
    ap.add_argument("--mute", action="store_true", help="HUD only, no speech")
    ap.add_argument("--eleven", action="store_true",
                    help="speak with ElevenLabs instead of macOS `say` (needs ELEVENLABS_API_KEY in .env)")
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
    if args.eleven:
        config.TTS_BACKEND = "elevenlabs"
        if not os.environ.get("ELEVENLABS_API_KEY"):
            print("[tts] --eleven given but no ELEVENLABS_API_KEY in .env or the environment; using say")
    print(f"[tts] voice: {voice.backend() if config.SPEAK else 'muted'}")
    App(args).run()


if __name__ == "__main__":
    main()
