"""All tunables in one place. Tune the relevance numbers live with the debug panel (press D)."""

# ---- models ----
YOLO_MODEL = "yolo11n.pt"                       # auto-downloads on first run
YOLO_CONF = 0.4
YOLO_IMGSZ = 640
VLM_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
VLM_FALLBACK = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
VLM_MAX_TOKENS = 40
VLM_IMG_MAX = 448                               # px, longest side sent to the VLM
WHISPER_MODEL = "mlx-community/whisper-small-mlx"

# ---- capture ----
CAPTURE_SIZE = (1280, 720)
CAPTURE_FALLBACK_SIZE = (640, 480)
WARMUP_FRAMES = 30                              # auto-exposure warm-up, discarded

# ---- perception ----
CENTER_FRAC = 0.5          # box centre within the middle 50% of the frame counts as "looked at"
MOVE_FRAC = 0.15           # centre shift > 15% of frame width counts as MOVE
GONE_AFTER_S = 2.0         # track unseen this long counts as GONE

# ---- memory ----
DB_PATH = "memory.db"
CROPS_DIR = "crops"
RETENTION_S = 60 * 60      # privacy: forget after 60 min

# ---- relevance ----
# S = w_n*novelty + w_d*dwell + w_c*conf + w_p*prior - w_r*exp(-dt/tau)
W_NOVELTY = 1.0
W_DWELL = 1.0
W_CONF = 0.5
W_PRIOR = 1.0
W_RECENT = 2.0
TAU_S = 20.0
# Spec said 1.8, but novelty+dwell alone = 2.0, so a held chair would fire. θ must exceed the
# class-independent terms (~2.5) so the prior decides: book 3.4 / bottle 3.2 fire, cup 2.55 / chair 2.45 don't.
THETA = 2.8
NOVELTY_WINDOW_S = 5 * 60  # label counts as novel if not *nudged about* in this window
DWELL_CAP_S = 2.0          # dwell term saturates at 1.0 after this many seconds
MIN_DWELL_S = 1.0          # hard gate: never nudge on a passing glance
LABEL_COOLDOWN_S = 120     # never nudge about the same label twice in this window

PRIORS = {
    "book": 1.0, "bottle": 0.8, "cell phone": 0.8, "laptop": 0.6,
    "backpack": 0.6, "remote": 0.5, "scissors": 0.5, "clock": 0.2,
    "cup": 0.1, "chair": 0.0, "person": 0.0, "tv": 0.0, "couch": 0.0,
    "dining table": 0.0, "bed": 0.0, "potted plant": 0.1, "keyboard": 0.1, "mouse": 0.1,
}
DEFAULT_PRIOR = 0.2

# ---- router synonyms → COCO labels ----
SYNONYMS = {
    "water bottle": "bottle", "bottle": "bottle", "flask": "bottle",
    "phone": "cell phone", "cellphone": "cell phone", "cell phone": "cell phone", "iphone": "cell phone",
    "laptop": "laptop", "computer": "laptop", "macbook": "laptop",
    "book": "book", "notebook": "book",
    "bag": "backpack", "backpack": "backpack",
    "mug": "cup", "cup": "cup",
    "remote": "remote", "keyboard": "keyboard", "mouse": "mouse",
    "scissors": "scissors", "umbrella": "umbrella", "clock": "clock",
    "chair": "chair",
}
# Preferred "near the X" anchors for memory answers, best first (people are never used)
LANDMARKS = ["laptop", "dining table", "couch", "bed", "tv", "keyboard", "sink", "refrigerator",
             "microwave", "oven", "chair", "backpack", "book", "cup", "potted plant"]
# Non-COCO things people will ask about → honest "I can't track that" answer
UNTRACKABLE = ["keys", "key", "wallet", "glasses", "sunglasses", "airpods", "charger", "poster"]

# ---- output ----
CARD_SECONDS = 5.0
CARD_MAX_CHARS = 60
SPEAK = True
