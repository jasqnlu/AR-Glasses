"""Fast tests for the demo-critical logic (router, memory recall, relevance). No camera/models.
Run: .venv/bin/python -m pytest -q test_logic.py
"""
import time

import config
from main import memory_answer, parse_where
from memory import Memory
from perception import Track
from relevance import Relevance


def mk(label, conf=0.8, dwell=0.0, tid=1):
    now = time.time()
    return Track(tid, label, conf, (100, 100, 50, 80), now, now, dwell_s=dwell, conf_ema=conf)


# ---- router
def test_where_synonyms():
    assert parse_where("Where's my water bottle?") == ("bottle", None)
    assert parse_where("where did I put my phone") == ("cell phone", None)
    assert parse_where("Where are my keys?") == (None, "keys")
    assert parse_where("what is this book about?") == (None, None)


def test_memory_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CROPS_DIR", str(tmp_path / "crops"))
    m = Memory(str(tmp_path / "m.db"))
    assert memory_answer("bottle", m, set()) == "I haven't seen your bottle yet."
    m.record(mk("bottle"), ["bottle", "laptop", "cup"])
    m.db.execute("UPDATE sightings SET ts = ts - 130")
    assert memory_answer("bottle", m, set()) == "Your bottle was near the laptop, about 2 minutes ago."
    assert "right in front of you" in memory_answer("bottle", m, {"bottle"})


# ---- relevance: beat 1 (nudge) and beat 3 (restraint)
def test_held_book_triggers():
    r = Relevance()
    assert r.tick([mk("book", 0.8, dwell=2.0)]) is not None


def test_glance_does_not_trigger():
    r = Relevance()
    assert r.tick([mk("book", 0.9, dwell=0.3)]) is None


def test_ordinary_objects_stay_silent():
    r = Relevance()
    scene = [mk(lbl, 0.9, dwell=5.0, tid=i) for i, lbl in enumerate(["chair", "cup", "tv", "person"])]
    assert r.tick(scene) is None


def test_cooldown_and_recent_speech_suppress():
    r = Relevance()
    book = mk("book", 0.8, dwell=2.0)
    assert r.tick([book]) is not None
    r.mark_nudged("book"); r.mark_spoken()
    assert r.tick([book]) is None                       # same label: cooldown
    assert r.tick([mk("cell phone", 0.8, 2.0, 2)]) is None  # just spoke: recency penalty
