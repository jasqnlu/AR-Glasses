# AR-Glasses

## Proactive Vision Assistant

A head-mounted camera assistant that sees what you see, remembers it, and speaks up **only when something is worth saying**. It runs 100% locally on a MacBook by default: footage, audio and memory never leave the device. The only optional cloud call is ElevenLabs text-to-speech (`--eleven`), which receives just the text of each spoken line.

```
Camera ─► latest-frame slot ─┬─► HUD (main thread, ~30 fps)
                             └─► YOLO11s-seg track (~33 ms) ─┬─► memory.py  (SQLite, write on APPEAR/MOVE/GONE)
                                                             ├─► rag.py: settled object leaves view → "where it was" memory
                                                             └─► relevance.py ─(S > θ)─► Qwen3-VL 4B ─► say + HUD card
 Question (SPACE / T) ─► Whisper ─► router ─┬─ "where is X?" ─► put-down memory / sightings → template (no hallucination)
                                            └─ anything else ─► top-k memories (MiniLM) + current frame ─► Qwen3-VL 4B
```

**Facts come from memory; language comes from the model.**

### Framework

Three layers, each running at its own speed:

| Layer | Question it answers | Runs | Cost |
| --- | --- | --- | --- |
| **Perception** | *What's in front of me?* | every frame | ~33 ms |
| **Relevance engine** | *Is it worth speaking?* | every frame | < 1 ms (arithmetic) |
| **Reasoning** | *What should I say?* | rarely, only when the engine fires or you ask | 0.5–1.5 s |

The cheap layers decide **when**; the expensive model only decides **what**. That's what lets it stay quiet most of the time.

#### Tech stack

| Part | Framework / library | Model | Runs on | Why |
| --- | --- | --- | --- | --- |
| Capture & HUD | OpenCV 5 | — | CPU | Camera/video I/O and the overlay window in one library |
| Detection + tracking | Ultralytics 8.4 (PyTorch 2.15, MPS) | `yolo11s-seg` + ByteTrack | Apple GPU (MPS) | Stable track IDs and mask polygons from one call; 80 COCO classes |
| Vision-language model | MLX 0.32 + `mlx-vlm` 0.8 | `Qwen3-VL-4B-Instruct-4bit` (2B fallback) | Apple GPU (MLX) | 4-bit model that fits next to everything else in 16 GB; ~1 s answers |
| Speech-to-text | `mlx-whisper` 0.4 | `whisper-small-mlx` | Apple GPU (MLX) | ~0.2 s for a short question, fully local |
| Memory store | SQLite (stdlib) | — | CPU | Sightings, put-down memories, utterances and embedding vectors in one file; auto-expires after 60 min |
| Retrieval (RAG) | `sentence-transformers` 6.3 + NumPy | `all-MiniLM-L6-v2` (384-d) | CPU | ~10 ms per sentence; brute-force cosine search, no vector DB needed |
| Relevance engine | plain Python (`relevance.py`) | hand-tuned formula | CPU | Transparent and tunable live; no training data needed |
| Voice | macOS `say` (default) · ElevenLabs API + `sounddevice` (`--eleven`) | `eleven_flash_v2_5` | CPU / cloud (text only) | Offline by default; streamed PCM for ~0.3–0.5 s to first sound |
| Microphone | `sounddevice` (PortAudio) | — | CPU | 16 kHz mono push-to-talk recording |

Python 3.12 on Apple Silicon (tested on an M5 MacBook, 16 GB).

#### Threads

| Thread | Does | Rule |
| --- | --- | --- |
| `main` | HUD drawing + keyboard | macOS GUI calls must stay on the main thread |
| `capture` | Reads the camera into a single latest-frame slot | Frames are overwritten, never queued |
| `perception` | YOLO → memory writes → relevance score → triggers | Always works on the newest frame |
| `mlx` | All VLM and Whisper calls, one at a time | MLX binds GPU streams to a thread; a nudge arriving while busy is dropped |
| `tts` | Streams ElevenLabs audio to the speaker | A new line interrupts the old one |

#### Modules

| File | Role |
| --- | --- |
| `main.py` | Entry point, CLI flags, question router, thread wiring |
| `capture.py` | Camera / video source |
| `perception.py` | YOLO tracking, per-track state (dwell, stillness, label votes, polygon), APPEAR/MOVE/GONE events |
| `relevance.py` | The scoring formula and hard gates that decide when to speak |
| `memory.py` | SQLite tables: `sightings`, `memories` (with vectors), `utterances`; privacy purge |
| `rag.py` | Spatial relations, VLM place descriptions, MiniLM embedder |
| `reasoning.py` | Qwen3-VL loading, prompts, the single MLX worker thread |
| `asr.py` | Push-to-talk recording + Whisper |
| `voice.py` | `say` / ElevenLabs output with fallback |
| `hud.py` | Overlay: polygons, answer card, debug panel, typed-question box |
| `config.py` | Every tunable (thresholds, priors, models) + `.env` loader |

## Quick start

```bash
cd proactive-vision
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python main.py --fresh            # add --eleven for ElevenLabs voice (key in .env)
```

**Keys:** `SPACE` start/stop talking · `T` type a question · `D` debug panel · `Q` quit.

Full details (run options, relevance engine, object memory, measurements, tests): **[proactive-vision/README.md](proactive-vision/README.md)**
