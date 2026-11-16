"""OpenCV HUD: video + tracked boxes + fading answer card + debug panel + typed-question box.
Must be drawn/shown from the main thread (macOS GUI rule)."""
import time

import cv2

import config

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = "Proactive Vision"


def _text(img, s, org, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


class HUD:
    def __init__(self):
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        self.card_text, self.card_t, self.card_kind = "", 0.0, "nudge"
        self.debug = True
        self.typing = None          # str while typed-question box is open
        self.status = ""            # e.g. "listening…", "thinking…"
        self._fps_t, self.fps = time.time(), 0.0

    def card(self, text, kind="nudge"):
        t = text if len(text) <= config.CARD_MAX_CHARS else text[:config.CARD_MAX_CHARS - 1] + "…"
        self.card_text, self.card_t, self.card_kind = t, time.time(), kind

    def draw(self, frame, tracks, scores, stats):
        img = frame.copy()
        H, W = img.shape[:2]
        now = time.time()
        dt = now - self._fps_t
        self._fps_t = now
        self.fps = 0.9 * self.fps + 0.1 * (1 / dt if dt > 0 else 0)

        # tracked boxes
        for t in tracks:
            x, y, w, h = t.bbox
            sc = scores.get(t.track_id)
            hot = sc and sc[1] > config.THETA
            col = (0, 200, 255) if hot else (120, 220, 120)
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
            lab = f"{t.label} #{t.track_id}"
            if self.debug and sc:
                lab += f"  S={sc[1]:.2f} d={t.dwell_s:.1f}s"
            _text(img, lab, (x + 4, max(y - 8, 16)), 0.5, col)

        # answer / nudge card, fades after CARD_SECONDS
        age = now - self.card_t
        if self.card_text and age < config.CARD_SECONDS:
            alpha = 1.0 if age < config.CARD_SECONDS - 1 else config.CARD_SECONDS - age
            (tw, th), _ = cv2.getTextSize(self.card_text, FONT, 0.9, 2)
            cw, ch = min(tw + 40, W - 40), th + 36
            x0, y0 = (W - cw) // 2, H - ch - 30
            overlay = img.copy()
            accent = (255, 170, 60) if self.card_kind == "answer" else (60, 200, 255)
            cv2.rectangle(overlay, (x0, y0), (x0 + cw, y0 + ch), (25, 25, 25), -1)
            cv2.rectangle(overlay, (x0, y0), (x0 + 6, y0 + ch), accent, -1)
            _text(overlay, self.card_text, (x0 + 22, y0 + th + 16), 0.9, (255, 255, 255), 2)
            img = cv2.addWeighted(overlay, 0.85 * alpha, img, 1 - 0.85 * alpha, 0)

        # status line (top-left)
        top = f"{self.fps:4.1f} fps"
        if self.status:
            top += f"   {self.status}"
        _text(img, top, (12, 26), 0.6, (200, 255, 200))

        # debug panel (top-right)
        if self.debug:
            lines = [f"yolo {stats.get('yolo_ms', 0):.0f} ms   vlm {stats.get('vlm_ms', 0):.0f} ms",
                     f"theta {config.THETA:.2f}   quiet {stats.get('quiet_s', 0):.0f}s"]
            for label, s, p in sorted(scores.values(), key=lambda v: -v[1])[:6]:
                lines.append(f"{label[:12]:12s} {s:5.2f} n{p['n']:.1f} d{p['d']:.1f} "
                             f"c{p['c']:.1f} p{p['p']:.1f} r{p['r']:.1f}")
            pw, x0 = 400, max(W - 410, 0)
            panel = img.copy()
            cv2.rectangle(panel, (x0, 6), (x0 + pw, 14 + 20 * len(lines)), (20, 20, 20), -1)
            img = cv2.addWeighted(panel, 0.7, img, 0.3, 0)
            for i, ln in enumerate(lines):
                _text(img, ln, (x0 + 8, 24 + 20 * i), 0.42, (255, 255, 0) if i < 2 else (230, 230, 230))

        # typed question box
        if self.typing is not None:
            cv2.rectangle(img, (20, 50), (W - 20, 95), (30, 30, 30), -1)
            _text(img, "Ask: " + self.typing + "_", (32, 82), 0.75, (255, 255, 255), 2)
        else:
            _text(img, "SPACE talk  T type  D debug  Q quit", (12, H - 12), 0.45, (180, 180, 180))

        cv2.imshow(WIN, img)
