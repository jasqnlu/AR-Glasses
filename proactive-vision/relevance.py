"""Relevance engine — decides WHEN to speak.

S = w_n·novelty + w_d·dwell + w_c·conf + w_p·prior[label] − w_r·exp(−Δt/τ)

Plus two hard gates so the demo's "restraint" beat is reliable:
  * dwell ≥ MIN_DWELL_S   (a passing glance never triggers)
  * same label not nudged in LABEL_COOLDOWN_S
"""
import math
import time

import config


class Relevance:
    def __init__(self):
        self.last_spoken = 0.0             # any utterance (nudge or answer)
        self.last_nudged: dict[str, float] = {}
        self.scores: dict[int, tuple] = {}  # track_id → (label, S, parts) for the debug panel

    def score(self, track, now):
        last = self.last_nudged.get(track.label)
        novelty = 1.0 if last is None or now - last > config.NOVELTY_WINDOW_S else 0.0
        dwell = min(track.dwell_s / config.DWELL_CAP_S, 1.0)
        prior = config.PRIORS.get(track.label, config.DEFAULT_PRIOR)
        recent = math.exp(-(now - self.last_spoken) / config.TAU_S) if self.last_spoken else 0.0
        parts = {
            "n": config.W_NOVELTY * novelty,
            "d": config.W_DWELL * dwell,
            "c": config.W_CONF * track.conf,
            "p": config.W_PRIOR * prior,
            "r": -config.W_RECENT * recent,
        }
        return sum(parts.values()), parts

    def tick(self, tracks):
        """Score all visible tracks; return the single best track that should trigger, or None."""
        now = time.time()
        self.scores.clear()
        best, best_s = None, config.THETA
        for t in tracks:
            s, parts = self.score(t, now)
            self.scores[t.track_id] = (t.label, s, parts)
            gated = (t.dwell_s < config.MIN_DWELL_S or
                     now - self.last_nudged.get(t.label, -1e9) < config.LABEL_COOLDOWN_S)
            if s > best_s and not gated:
                best, best_s = t, s
        return best

    def mark_nudged(self, label):
        """Call when a nudge is *attempted* (even if the VLM says SKIP) so we don't re-ask every tick."""
        self.last_nudged[label] = time.time()

    def mark_spoken(self):
        self.last_spoken = time.time()
