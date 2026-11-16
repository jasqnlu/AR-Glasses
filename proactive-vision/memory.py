"""SQLite scene memory. Written only on APPEAR/MOVE. Auto-forgets after RETENTION_S."""
import os
import sqlite3
import threading
import time

import cv2

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sightings (
  id INTEGER PRIMARY KEY,
  ts REAL, label TEXT, track_id INTEGER, conf REAL,
  bbox TEXT, near TEXT, crop_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_label_ts ON sightings(label, ts);
CREATE TABLE IF NOT EXISTS utterances (ts REAL, text TEXT, trigger TEXT);
"""


class Memory:
    def __init__(self, path=config.DB_PATH, fresh=False):
        if fresh and os.path.exists(path):
            os.remove(path)
        os.makedirs(config.CROPS_DIR, exist_ok=True)
        # one connection shared across threads, guarded by a lock
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self._last_purge = 0.0

    def record(self, track, near_labels):
        x, y, w, h = track.bbox
        crop_path = None
        if track.crop is not None and track.crop.size:
            crop_path = os.path.join(config.CROPS_DIR, f"{int(time.time()*1000)}_{track.track_id}.jpg")
            cv2.imwrite(crop_path, track.crop)
        near = ", ".join(sorted(set(near_labels) - {track.label}))
        with self.lock:
            self.db.execute(
                "INSERT INTO sightings (ts,label,track_id,conf,bbox,near,crop_path) VALUES (?,?,?,?,?,?,?)",
                (time.time(), track.label, track.track_id, track.conf, f"{x},{y},{w},{h}", near, crop_path))
            self.db.commit()

    def update_last_seen(self, track, near_labels):
        """Refresh ts/near of this track's newest row (on GONE) so 'last seen' is accurate."""
        near = ", ".join(sorted(set(near_labels) - {track.label}))
        with self.lock:
            self.db.execute(
                "UPDATE sightings SET ts=?, near=CASE WHEN ?<>'' THEN ? ELSE near END "
                "WHERE id=(SELECT MAX(id) FROM sightings WHERE track_id=? AND label=?)",
                (track.last_seen, near, near, track.track_id, track.label))
            self.db.commit()

    def last_seen(self, label):
        """→ dict(label, near, age_s, crop_path) or None."""
        with self.lock:
            row = self.db.execute(
                "SELECT ts, near, crop_path FROM sightings WHERE label=? ORDER BY ts DESC LIMIT 1",
                (label,)).fetchone()
        if not row:
            return None
        return {"label": label, "near": row[1], "age_s": time.time() - row[0], "crop_path": row[2]}

    def recent(self, n=3):
        with self.lock:
            rows = self.db.execute(
                "SELECT ts, label, near FROM sightings ORDER BY ts DESC LIMIT ?", (n,)).fetchall()
        now = time.time()
        return [f"{lab} near {near or 'nothing notable'}, {int(now-ts)}s ago" for ts, lab, near in rows]

    def log_utterance(self, text, trigger):
        with self.lock:
            self.db.execute("INSERT INTO utterances VALUES (?,?,?)", (time.time(), text, trigger))
            self.db.commit()

    def purge(self):
        """Privacy: delete rows + crops older than RETENTION_S. Cheap; call often."""
        now = time.time()
        if now - self._last_purge < 30:
            return
        self._last_purge = now
        cutoff = now - config.RETENTION_S
        with self.lock:
            paths = self.db.execute(
                "SELECT crop_path FROM sightings WHERE ts<? AND crop_path IS NOT NULL", (cutoff,)).fetchall()
            self.db.execute("DELETE FROM sightings WHERE ts<?", (cutoff,))
            self.db.execute("DELETE FROM utterances WHERE ts<?", (cutoff,))
            self.db.commit()
        for (p,) in paths:
            try:
                os.remove(p)
            except OSError:
                pass
