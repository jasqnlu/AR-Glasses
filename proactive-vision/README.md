# Proactive Vision Assistant

A head-mounted camera assistant that sees what you see, remembers it, and speaks up **only when something is worth saying**. It runs 100% locally on a MacBook: no cloud APIs, and footage never leaves the device.

```
Camera ─► latest-frame slot ─┬─► HUD (main thread, ~30 fps)
                             └─► YOLO11s-seg track (~33 ms) ─┬─► memory.py  (SQLite, write on APPEAR/MOVE/GONE)
                                                         └─► relevance.py ─(S > θ)─► Qwen3-VL 4B ─► say + HUD card
 Question (SPACE / T) ─► Whisper ─► router ─┬─ "where is X?" ─► memory template (instant, no hallucination)
                                            └─ anything else ─► Qwen3-VL 4B with current frame
```

**Facts come from memory; language comes from the model.**

## Setup (Apple Silicon, Python 3.12)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python main.py            # first run downloads YOLO + Qwen3-VL 4B + whisper-small (~3 GB)
```

macOS will ask for **camera** and **microphone** access for your terminal on first run.

## Run

| Command | What |
| --- | --- |
| `python main.py` | Live webcam 0, full pipeline |
| `python main.py --source clips/pov.mp4` | Backup demo: a recorded POV clip through the same pipeline |
| `python main.py --fresh` | Wipe memory first (do this before a demo run) |
| `python main.py --small` | 2B VLM if memory pressure goes yellow |
| `python main.py --no-vlm --no-asr` | Fast iteration on perception and relevance |
| `python main.py --headless --source clip.mp4 --ask "where's my water bottle?"` | Scripted smoke test |

**Keys:** `SPACE` start/stop talking · `T` type a question · `D` debug panel · `Q` quit.
(OpenCV doesn't report key-up, so push-to-talk is a toggle.)

## Relevance engine

```
S = w_n·novelty + w_d·dwell + w_c·conf + w_p·prior[label] − w_r·exp(−Δt/τ)
```
plus hard gates: **dwell ≥ 0.8 s** (a glance never triggers), **smoothed conf ≥ 0.35**, and a **per-label 120 s cooldown**.

**θ = 2.8, not the 1.8 in the spec.** Novelty + full dwell alone add up to 2.0, so with θ = 1.8 a chair held for 2 s would trigger a nudge. θ has to sit above the class-independent terms (~2.5) so the **prior** decides:

| Held 2 s, conf 0.8 | S | |
| --- | --- | --- |
| book | 3.40 | speaks |
| bottle / cell phone | 3.20 | speaks |
| laptop | 3.00 | speaks |
| cup | 2.50 | silent |
| chair / tv / person | 2.40 | silent |

After speaking, the recency term keeps the next nudge ≥ ~25 s away. The VLM can still veto with `SKIP`. All weights and priors are in `config.py`; the `D` panel shows each term live.

## Design notes

- **One MLX thread.** The VLM and Whisper share one worker thread (MLX binds GPU streams to threads). A VLM trigger that arrives while the worker is busy is *dropped*, not queued.
- **Source opens after models load.** A clip doesn't start playing until YOLO, the VLM and Whisper are warm, so the first seconds of the backup demo aren't skipped.
- **Memory context belongs to the track.** Each track remembers which labels it was seen *with*. On GONE, the sighting's timestamp is refreshed, but its location ("near") comes from where the object actually was, not whatever is in view 2 s later.
- **Landmark ranking.** "Near the X" prefers static objects (laptop > table > couch > tv…) and never uses a person.
- **Polygon outlines.** `yolo11s-seg` gives a per-object mask; the HUD draws its outline, not a box. People (including your own arm in a head-mounted view) get an outline only, no fill.
- **Objects in your hand.** Fingers cover part of the object and drop YOLO's confidence, so the tracker gets boxes down to 0.1 (ByteTrack uses weak boxes to keep tracks alive). Confidence bars apply only where it matters: memory writes need 0.45 (a false "bottle" can't poison recall) and nudges need a smoothed 0.35.
- **Label voting.** Each track's label is the confidence-weighted majority over its lifetime, so a phone in the hand doesn't flip to "remote" and get nudged twice.
- **Privacy.** Rows and crops older than 60 min are deleted automatically.

## Measured (M5, 16 GB)

| Stage | Target | Measured |
| --- | --- | --- |
| YOLO11s-seg track | ≤ 60 ms | ~33 ms median (first MPS call ~2 s, done at warm-up) |
| HUD | ≥ 15 fps | ~26 fps |
| VLM, warm | ≤ 2 s | 0.5–1.5 s |
| Startup (warm cache) | — | ~10 s |

## Tests

```bash
.venv/bin/python -m pytest -q test_logic.py   # router, memory recall, nudge / glance / restraint / cooldown
```

## Limits

YOLO only knows the 80 COCO classes (no keys, wallet or posters). The router answers those honestly ("I can't track keys yet"). Next step: YOLO-World with custom classes.
