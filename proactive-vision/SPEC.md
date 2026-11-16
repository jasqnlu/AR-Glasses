# SPEC — Proactive Vision Assistant (VexoHacks, Nov 14 2026)

**One line:** A head-mounted camera assistant that sees what you see, remembers it, and speaks up only when something is worth saying — running 100% locally on a MacBook.

**Builder:** Jason (solo) · **Machine:** MacBook M5, 16 GB · **Build window:** 9 AM–9 PM

## Locked decisions (tested Fri night)

| Piece | Choice | Tested |
| --- | --- | --- |
| Detection | YOLO nano, `model.track(persist=True)`, `device="mps"` | ✅ |
| Demo object | **water bottle** (COCO class `bottle`) | — |
| VLM | `mlx-community/Qwen3-VL-4B-Instruct-4bit` (fallback: 2B) | ✅ 7.9 s total incl. load |
| Speech-to-text | `mlx-whisper`, `mlx-community/whisper-small-mlx` | ✅ |
| Speech out | macOS `say` | ✅ |
| Cloud APIs | **None** (all local) | — |

Note: YOLO only knows the 80 COCO classes (bottle, cup, cell phone, book, laptop, backpack, remote…). No "keys", no "poster". Stretch: YOLO-World with custom classes.

---

## 1. Goals

| Must have (demo-critical) | Nice to have | Out of scope |
| --- | --- | --- |
| Live camera → tracked detections | Vexo bracelet voice input | Real AR glasses hardware |
| Scene memory + "where is X?" recall | YOLO-World custom classes | Spatially anchored overlays |
| Relevance engine → proactive nudges | Memory timeline in HUD | Cloud APIs / accounts |
| HUD card + spoken output | Voice question via Whisper | Training/fine-tuning |

---

## 2. Demo scenario (three beats, ~90 s)

1. **Proactive nudge** — a useful object/text (book cover, label, sign) enters view and is held → assistant says one helpful line *unprompted*.
2. **Memory recall** — water bottle set down early. Later: "Where's my water bottle?" → "Your bottle is near the laptop, about 2 minutes ago."
3. **Restraint** — look around at ordinary objects (chair, cup, monitor) → **silent**. Tell judges: knowing when *not* to talk is the hard problem.

Backup: recorded POV clip run through the same pipeline (`--source clip.mp4`).

---

## 3. Architecture

```
Camera 30 fps ─► [latest-frame slot] (overwritten, never queued)
                     │
     ┌───────────────┼──────────────────────┐
     ▼               ▼
 HUD @ 30 fps    YOLO track @ ~5–10 fps (latest frame only)
                     │  objects with stable IDs, dwell, crops
             ┌───────┴────────┐
             ▼                ▼
   memory.py (SQLite)    relevance.py — score every tick
   write on appear/move        │ S > θ and not cooling down (rare)
                               ▼
 Question ─► asr.py ─► router ─┬─ "where is <label>?" → memory → TEMPLATE answer
 (push-to-talk)                └─ other → reasoning.py (Qwen3-VL 4B: crop + memory)
                               ▼
                    voice.py: HUD card + `say` (non-blocking)
```

**Threading rules**
1. Latest frame wins — one shared slot (or `Queue(maxsize=1)`), slow stages skip frames.
2. Nothing slow in the video loop — VLM, Whisper, `say` in their own threads.
3. One VLM call at a time — new triggers while busy are **dropped**, not queued.

---

## 4. Modules

### `capture.py`
- `cv2.VideoCapture`, 1280×720 if available else 640×480.
- **Discard first ~30 frames** (auto-exposure warm-up — first frames are dark).
- Background thread writes only the latest frame. `--source` = camera index or video path.

### `perception.py`
- `model.track(frame, persist=True, device="mps", conf=0.4)` → stable `track_id` per object.
- Per track: `label, conf, bbox, first_seen, last_seen, dwell_s` (seconds near frame center), `crop`.
- Emits events: `APPEAR(track)`, `MOVE(track)` (center shifted > X px), `GONE(track)`.

### `memory.py` — SQLite
```sql
CREATE TABLE sightings (
  id INTEGER PRIMARY KEY,
  ts REAL, label TEXT, track_id INTEGER, conf REAL,
  bbox TEXT,          -- x,y,w,h
  near TEXT,          -- co-occurring labels, e.g. "laptop, cup"
  crop_path TEXT
);
CREATE TABLE utterances (ts REAL, text TEXT, trigger TEXT);
```
- Write **only on APPEAR / MOVE** events (not every frame).
- `last_seen(label)` → newest row + `near` + age.
- Privacy: rows/crops older than 60 min auto-deleted.

### `relevance.py` — core custom piece
```
S = w_n·novelty + w_d·dwell + w_c·conf + w_p·prior[label] − w_r·exp(−Δt/τ)
```
- novelty: label/track not seen in last N min · dwell: seconds near center (capped) · conf: YOLO · prior: hand-set interest per class · Δt: seconds since last spoken line.
- Start: `w_n=1, w_d=1, w_c=0.5, w_p=1, w_r=2, τ=20s, θ=1.8`. Priors: book/bottle/cell phone/laptop high; chair/person/tv/couch low.
- Log scores to console (debug HUD panel) for live tuning on the demo clip.

### `router` (in `main.py`)
- Question contains "where" + a COCO label (with synonyms: "water bottle"→`bottle`, "phone"→`cell phone`) → **memory path**:
  `f"Your {label} is near the {near}, {mins} minutes ago."` — instant, no hallucination.
- Label never seen → "I haven't seen your {label} yet."
- Anything else → **VLM path** with current frame.

### `reasoning.py`
- Load Qwen3-VL 4B **once** at startup + one warm-up call. Model ID in `config.py`.
- Input: crop/frame resized to ≤ 384–512 px + ≤ 3 memory rows as text. `max_tokens=40`.
- Proactive prompt: *"You are a wearable assistant. In under 15 words, say the single most useful thing about this. If nothing is useful, reply exactly: SKIP."*
- Question prompt: *"The user asks: '{q}'. Recent memory: {rows}. Answer in under 15 words."*
- `SKIP` → silent.

### `asr.py`
- Push-to-talk key → `sounddevice` record @ 16 kHz mono → `mlx_whisper.transcribe(..., path_or_hf_repo="mlx-community/whisper-small-mlx")`.
- Fallback: typed question box. Stretch: Vexo bracelet.

### `voice.py` / `hud.py`
- `subprocess.Popen(["say", text])` (non-blocking) + HUD card ≤ 60 chars, fades after ~5 s.
- HUD: always-on-top window (PySide6, or OpenCV window fallback). Debug toggle: tracks, scores, latency.

---

## 5. Performance targets

| Stage | Target |
| --- | --- |
| Capture → HUD | ≥ 15 FPS, never freezes |
| YOLO track | ≤ 60 ms/frame |
| VLM (4B, crop, warm) | ≤ 2 s |
| Trigger → spoken line | ≤ 3 s |
| Memory pressure | green (else switch to 2B) |

---

## 6. Timeline (solo)

| Time | Milestone | Done when |
| --- | --- | --- |
| 9:00–9:15 | Repo, venv, SPEC committed | First commit |
| 9:15–11:00 | capture → YOLO track → HUD | Tracked labels live in HUD |
| 11:00–1:00 | memory + router + template recall + VLM | "Where's my water bottle?" correct |
| 1:00–3:30 | relevance engine + `say` | Nudge fires; ordinary scene silent |
| 3:30–4:30 | push-to-talk Whisper (or typed) | Spoken question works |
| 4:30–5:00 | buffer / YOLO-World / Vexo | — |
| **5:00** | **Feature freeze** | — |
| 5:00–7:00 | Tune θ on clip, rehearse 5× | 3 beats back-to-back |
| 7:00–9:00 | Backup video, README, pitch | Submitted |

**30-minute rule:** stuck > 30 min → cut or fallback.

---

## 7. Cut list (in order)

1. Vexo bracelet → push-to-talk
2. Whisper → typed question
3. PySide6 HUD → OpenCV overlay
4. Live camera → recorded clip
5. **Never cut:** memory recall + relevance engine

---

## 8. Pitch (3 min)

1. **Problem (20 s):** assistants wait for prompts — or narrate everything.
2. **Demo (90 s):** nudge → recall → silence.
3. **How (40 s):** cheap tracked YOLO every frame; custom relevance engine decides *when*; small local VLM decides *what*. Facts come from memory, language from the model.
4. **Why local (15 s):** entirely on a laptop; footage never leaves the device; memory expires.
5. **Next (15 s):** the brain for AI smart glasses.

---

## 9. Repo layout

```
main.py  capture.py  perception.py  memory.py  relevance.py
reasoning.py  asr.py  voice.py  hud.py  config.py
requirements.txt  README.md
.gitignore   # .venv/, *.pt, *.db, clips/, crops/
```

Commit at every milestone.
