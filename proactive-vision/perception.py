"""YOLO tracking → per-track state + APPEAR / MOVE / GONE events."""
import time
from collections import Counter
from dataclasses import dataclass, field

from ultralytics import YOLO

import config


@dataclass
class Track:
    track_id: int
    label: str
    conf: float
    bbox: tuple            # x, y, w, h (pixels)
    first_seen: float
    last_seen: float
    dwell_s: float = 0.0   # continuous seconds near frame centre
    anchor: tuple = None   # centre at last APPEAR/MOVE event
    near: set = field(default_factory=set)   # labels co-visible while this track was visible
    conf_ema: float = 0.0  # smoothed confidence (in-hand objects flicker as fingers move)
    recorded: bool = False # has a trusted sighting been written to memory yet
    poly: object = field(default=None, repr=False)   # Nx2 int32 outline from the seg mask
    votes: Counter = field(default_factory=Counter, repr=False)  # conf-weighted label votes
    crop: object = field(default=None, repr=False)

    @property
    def center(self):
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)


class Perception:
    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL)
        self.tracks: dict[int, Track] = {}
        self.ms = 0.0
        self._last_t = None
        self.current = []

    def warmup(self, frame):
        self.model.track(frame, persist=True, device="mps", conf=config.YOLO_CONF, verbose=False)

    def step(self, frame):
        """Run tracking on one frame. Returns list of (event, Track)."""
        now = time.time()
        dt = 0.0 if self._last_t is None else min(now - self._last_t, 0.5)
        self._last_t = now
        t0 = time.perf_counter()
        r = self.model.track(frame, persist=True, device="mps", conf=config.YOLO_CONF,
                             imgsz=config.YOLO_IMGSZ, verbose=False)[0]
        self.ms = (time.perf_counter() - t0) * 1000

        H, W = frame.shape[:2]
        cx0, cx1 = W * (0.5 - config.CENTER_FRAC / 2), W * (0.5 + config.CENTER_FRAC / 2)
        cy0, cy1 = H * (0.5 - config.CENTER_FRAC / 2), H * (0.5 + config.CENTER_FRAC / 2)
        move_px = W * config.MOVE_FRAC
        events = []
        self.current = []      # tracks matched in THIS frame (for co-occurrence)

        boxes = r.boxes
        if boxes is not None and boxes.id is not None:
            polys = r.masks.xy if r.masks is not None else [None] * len(boxes)
            for xyxy, tid, cls, conf, poly in zip(boxes.xyxy.tolist(), boxes.id.int().tolist(),
                                                  boxes.cls.int().tolist(), boxes.conf.tolist(), polys):
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                bbox = (x1, y1, x2 - x1, y2 - y1)
                raw = r.names[cls]
                tr = self.tracks.get(tid)
                if tr is None:
                    tr = Track(tid, raw, conf, bbox, now, now, conf_ema=conf)
                    tr.votes[raw] += conf
                    self.tracks[tid] = tr
                    tr.anchor = tr.center
                    events.append(("APPEAR", tr))
                else:
                    # label = conf-weighted majority over the track's life (stops phone→remote flicker)
                    tr.votes[raw] += conf
                    tr.label = tr.votes.most_common(1)[0][0]
                    tr.conf, tr.bbox, tr.last_seen = conf, bbox, now
                    tr.conf_ema += config.CONF_EMA * (conf - tr.conf_ema)
                tr.poly = poly.astype("int32") if poly is not None and len(poly) >= 3 else None
                self.current.append(tr)
                cx, cy = tr.center
                tr.dwell_s = tr.dwell_s + dt if (cx0 <= cx <= cx1 and cy0 <= cy <= cy1) else 0.0
                tr.crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
                ax, ay = tr.anchor
                if ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5 > move_px:
                    tr.anchor = (cx, cy)
                    events.append(("MOVE", tr))

        for tid, tr in list(self.tracks.items()):
            if now - tr.last_seen > config.GONE_AFTER_S:
                events.append(("GONE", tr))
                del self.tracks[tid]
        return events

    def visible(self, max_age=0.3):
        now = time.time()
        return [t for t in self.tracks.values() if now - t.last_seen <= max_age]
