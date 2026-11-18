"""Object memory with retrieval.

When a personal object has been sitting still and then leaves view, we save WHERE it was:
  1. relation(): deterministic phrase from geometry   → "on the dining table, left of the laptop"
  2. VLM description (when idle), object outlined     → "blue metal water bottle | on the kitchen counter by the sink"
  3. text embedding (MiniLM, CPU)                     → stored in SQLite, searched by cosine similarity
"""
import threading

import cv2
import numpy as np

import config

DESCRIBE_PROMPT = (
    "A green line marks one {label}; it is only a pointer, never mention it. Reply in exactly this format: "
    "<the {label}'s own colour, material or brand, max 6 words> | "
    "<where it is, starting with 'on', 'in' or 'next to', max 10 words>")
# place phrases that aren't a location (held objects, model hedging) are dropped, geometry is used instead
_NOT_A_PLACE = ("not visible", "unclear", "unknown", "hand", "held", "holding", "outline")
_PREPOSITIONS = ("on ", "in ", "at ", "by ", "near ", "next ", "under ", "inside ", "beside ", "behind ",
                 "against ", "atop ", "underneath ", "between ", "left ", "right ", "in front")


def _center(b):
    x, y, w, h = b
    return x + w / 2, y + h / 2


def relation(obj, others):
    """Spatial phrase for obj relative to other confident tracks. '' if nothing useful is in view."""
    x, y, w, h = obj.bbox
    cx, cy, bottom = x + w / 2, y + h / 2, y + h
    cands = [o for o in others
             if o.track_id != obj.track_id and o.label not in ("person", obj.label)]
    parts = []
    # resting ON a surface: object's bottom-centre lies inside the surface box, above its lower edge
    surfaces = [o for o in cands if o.label in config.SURFACES]
    for s in sorted(surfaces, key=lambda o: -o.bbox[2] * o.bbox[3]):
        sx, sy, sw, sh = s.bbox
        if sx <= cx <= sx + sw and sy <= bottom <= sy + 0.9 * sh:
            parts.append(f"on the {s.label}")
            break
    # nearest non-surface landmark → left of / right of / next to
    rest = [o for o in cands if o.label not in config.SURFACES]
    if rest:
        n = min(rest, key=lambda o: (_center(o.bbox)[0] - cx) ** 2 + (_center(o.bbox)[1] - cy) ** 2)
        nx, ny = _center(n.bbox)
        dx, dy = cx - nx, cy - ny
        side = ("left of" if dx < 0 else "right of") if abs(dx) > max(abs(dy), (w + n.bbox[2]) / 4) \
            else "next to"
        parts.append(f"{side} the {n.label}")
    return ", ".join(parts)


def outlined(frame, bbox, poly):
    """Copy of frame with the object outlined in green, so the VLM knows which object we mean."""
    img = frame.copy()
    if poly is not None:
        cv2.polylines(img, [poly], True, (0, 255, 0), 4, cv2.LINE_AA)
    else:
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return img


def parse_description(text, label):
    """'blue bottle | on the desk.' → ('blue bottle', 'on the desk'). Tolerates a missing '|'."""
    if not text or "SKIP" in text.upper()[:6]:
        return None, None
    a, _, p = text.partition("|")
    clean = lambda s: s.strip().strip(".\"'").strip() or None
    a, p = clean(a), clean(p)
    if a and ("outline" in a.lower() or a.lower() == "green"):   # the marker leaking into the answer
        a = None
    if p and any(w in p.lower() for w in _NOT_A_PLACE):
        p = None
    if p:
        p = p[0].lower() + p[1:]
        if not p.startswith(_PREPOSITIONS):      # "wooden surface" → "on wooden surface"
            p = "on " + p
    return a, p


def memory_text(label, appearance, place, rel):
    """The sentence we embed and show to the VLM: what it looks like + where it was."""
    if not appearance:
        what = label
    elif label.split()[-1] in appearance.lower():   # "blue water bottle" already names it
        what = appearance
    else:                                           # "blue Hydro Flask" → add the class word
        what = f"{appearance} {label}"
    where = "; ".join(s for s in (place, rel) if s)
    return f"{what} {where}".strip()


class Embedder:
    """all-MiniLM-L6-v2 on CPU (keeps MPS free for YOLO). Returns L2-normalised float32 vectors."""

    def __init__(self, model=None):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model or config.EMBED_MODEL, device="cpu")
        self.lock = threading.Lock()

    def __call__(self, text):
        with self.lock:
            return self.m.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
