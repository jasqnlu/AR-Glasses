# Proactive Vision Assistant

A head-mounted camera assistant that sees what you see, remembers it, and speaks up **only when something is worth saying**. It runs locally on a MacBook. Footage, audio and memory never leave the device; the only optional cloud call is ElevenLabs text-to-speech, which receives just the text of each spoken line (`--tts say` keeps it 100% offline).

```
Camera ─► latest-frame slot ─┬─► HUD (main thread, ~30 fps)
                             └─► YOLO11s-seg track (~33 ms) ─┬─► memory.py  (SQLite, write on APPEAR/MOVE/GONE)
                                                             ├─► rag.py: settled object leaves view → "where it was" memory
                                                             └─► relevance.py ─(S > θ)─► Qwen3-VL 4B ─► say + HUD card
 Question (SPACE / T) ─► Whisper ─► router ─┬─ "where is X?" ─► put-down memory / sightings → template (no hallucination)
                                            └─ anything else ─► top-k memories (MiniLM) + current frame ─► Qwen3-VL 4B
```

**Facts come from memory; language comes from the model.**

## Setup (Apple Silicon, Python 3.12)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python main.py            # first run downloads YOLO + Qwen3-VL 4B + whisper-small (~3 GB)
```

macOS will ask for **camera** and **microphone** access for your terminal on first run.

**Voice (optional):** for ElevenLabs instead of macOS `say`, create `proactive-vision/.env` (gitignored):

```
ELEVENLABS_API_KEY=sk_...
# ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb   # optional, default premade voice "George"
```

With a key set, `--tts auto` (the default) streams `eleven_flash_v2_5` audio (~0.3–0.5 s to first sound). Any network or quota error falls back to `say` for that line; a rejected key switches to `say` for the session.

## Run

| Command | What |
| --- | --- |
| `python main.py` | Live webcam 0, full pipeline |
| `python main.py --source clips/pov.mp4` | Backup demo: a recorded POV clip through the same pipeline |
| `python main.py --fresh` | Wipe memory first (do this before a demo run) |
| `python main.py --small` | 2B VLM if memory pressure goes yellow |
| `python main.py --tts say` | Fully offline voice (macOS `say`) even if an ElevenLabs key is set |
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

## Object memory (text RAG)

When a personal object (bottle, phone, book, laptop, bag, cup…) sits still for 1.5 s and then leaves view, it's remembered:

1. **Geometry, instant and deterministic:** `on the dining table, left of the laptop` (`rag.relation`).
2. **VLM description, when the VLM is idle:** the frame is sent with the object outlined in green → `clear plastic bottle | on the kitchen counter`. Captions queue behind nudges and questions, so they never delay a reply.
3. **Embedding:** `all-MiniLM-L6-v2` on CPU (~10 ms). The vector is stored in the same SQLite file; search is brute-force cosine (an hour of memories is a few hundred rows, so no vector DB is needed).

"Where's my bottle?" uses the latest put-down memory: *"Your bottle was on the kitchen counter, left of the laptop, about 3 minutes ago."* Any other question retrieves the **most relevant** memories (not just the latest 3) and gives them to the VLM with the current frame, so "what did my bottle look like?" works with the bottle out of view.

Guards: the VLM is told the outline is only a marker; place text describing a hand/held/"not visible" is discarded in favour of geometry; twin detections within 5 s make a single memory. `--no-rag` turns embeddings off.

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
| ElevenLabs first audio | — | ~0.3–0.5 s (streamed PCM) |
| Startup (warm cache) | — | ~10 s |

## Tests

```bash
.venv/bin/python -m pytest -q test_logic.py   # router, recall, relevance, geometry, RAG search (12 tests)
```

## Limits

YOLO only knows the 80 COCO classes (no keys, wallet or posters). The router answers those honestly ("I can't track keys yet"). Next step: YOLO-World with custom classes.
