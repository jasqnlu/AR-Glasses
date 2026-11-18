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


# ---- object memory / RAG
import numpy as np

import rag


def box(label, bbox, tid):
    t = mk(label, tid=tid)
    t.bbox = bbox
    return t


def test_relation_on_surface_and_beside_landmark():
    table = box("dining table", (0, 200, 640, 280), 1)
    laptop = box("laptop", (300, 150, 200, 150), 2)
    bottle = box("bottle", (200, 180, 40, 100), 3)          # bottom at y=280, inside the table box
    assert rag.relation(bottle, [table, laptop, bottle]) == "on the dining table, left of the laptop"


def test_relation_ignores_people_and_empty_scene():
    bottle = box("bottle", (200, 180, 40, 100), 3)
    assert rag.relation(bottle, [bottle, box("person", (0, 0, 640, 480), 9)]) == ""


def test_parse_description():
    assert rag.parse_description("Blue metal water bottle | On the kitchen counter.", "bottle") == \
        ("Blue metal water bottle", "on the kitchen counter")
    assert rag.parse_description("SKIP", "bottle") == (None, None)
    assert rag.memory_text("bottle", "blue Hydro Flask", "on the desk", "left of the laptop") == \
        "blue Hydro Flask bottle on the desk; left of the laptop"


def test_where_prefers_put_down_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CROPS_DIR", str(tmp_path / "crops"))
    m = Memory(str(tmp_path / "m.db"))
    m.record(mk("bottle"), ["bottle", "laptop"])
    mid = m.add_memory("bottle", time.time(), "left of the laptop", "bottle left of the laptop")
    assert memory_answer("bottle", m, set()) == "Your bottle was left of the laptop, less than a minute ago."
    m.describe_memory(mid, "blue bottle", "on the kitchen counter by the sink", "blue bottle ...")
    assert memory_answer("bottle", m, set()) == \
        "Your bottle was on the kitchen counter by the sink, left of the laptop, less than a minute ago."


def test_search_ranks_by_similarity(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CROPS_DIR", str(tmp_path / "crops"))
    m = Memory(str(tmp_path / "m.db"))
    e = np.eye(4, dtype=np.float32)
    m.add_memory("book", time.time(), "", "red book on the couch", e[0])
    m.add_memory("bottle", time.time(), "", "bottle on the desk", e[1])
    q = (0.9 * e[1] + 0.1 * e[0]).astype(np.float32)
    hits = m.search(q / np.linalg.norm(q), k=2, min_sim=0.2)
    assert [h["label"] for h in hits] == ["bottle"]       # book is below min_sim


def test_parse_drops_marker_and_non_places():
    assert rag.parse_description("Green outline bottle | On table, in dimly lit room", "bottle") == \
        (None, "on table, in dimly lit room")
    assert rag.parse_description("green | wooden surface", "cell phone") == (None, "on wooden surface")
    assert rag.parse_description("Pink phone | Held in woman's hands", "cell phone") == ("Pink phone", None)
