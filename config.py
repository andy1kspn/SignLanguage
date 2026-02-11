"""
Configurație centrală pentru sistemul de recunoaștere a Limbajului Semnelor Românesc (LSR).

Arhitectura vectorului de intrare per cadru:
  - Mâna stângă:  21 landmarks × 3 (x, y, z) = 63 valori
  - Mâna dreaptă: 21 landmarks × 3 (x, y, z) = 63 valori
  - Pose (trunchi): 6 landmarks × 3 (x, y, z) = 18 valori
      (umeri, coate, încheieturi — indici MediaPipe: 11-16)
  - Față (expresii): 20 landmarks × 3 (x, y, z) = 60 valori
      (buze, sprâncene — componente non-manuale)
  ──────────────────────────────────────────────────
  Total per cadru: 204 valori
  Secvență: 30 cadre (~1 secundă la 30 FPS)
  Input model: (batch, 30, 204)
"""

import os

# ──────────────────────── Căi de fișiere ────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lsr_model.keras")

# ──────────────────────── Parametri secvență ────────────────────
SEQUENCE_LENGTH = 30       # cadre per secvență (≈1s la 30 FPS)
FPS_TARGET = 30

# ──────────────────────── Dimensiuni features ───────────────────
NUM_HAND_LANDMARKS = 21    # per mână (MediaPipe Hands)
NUM_POSE_LANDMARKS = 6     # subset trunchi superior
NUM_FACE_LANDMARKS = 20    # subset expresii faciale
COORDS_PER_LANDMARK = 3    # x, y, z

# Calculul dimensiunii vectorului per cadru
HAND_FEATURES = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK * 2   # 126 (ambele mâini)
POSE_FEATURES = NUM_POSE_LANDMARKS * COORDS_PER_LANDMARK        # 18
FACE_FEATURES = NUM_FACE_LANDMARKS * COORDS_PER_LANDMARK        # 60
TOTAL_FEATURES = HAND_FEATURES + POSE_FEATURES + FACE_FEATURES  # 204

# ──────────────────────── Indici MediaPipe ──────────────────────
# Pose: umăr stâng/drept, cot stâng/drept, încheietură stângă/dreaptă
POSE_INDICES = [11, 12, 13, 14, 15, 16]

# Face Mesh: puncte cheie pentru componente non-manuale
# Buze (contur exterior + interior): 10 puncte
# Sprâncene (stânga + dreapta): 10 puncte
FACE_INDICES = [
    # Buze — contur exterior
    61, 291, 0, 17, 269, 405,
    # Buze — contur interior
    78, 308, 13, 14,
    # Sprânceană stângă
    70, 63, 105, 66, 107,
    # Sprânceană dreaptă
    336, 296, 334, 293, 300,
]

# ──────────────────────── Clase (semne LSR) ─────────────────────
# Dicționar extensibil — adaugă semne noi aici
CLASSES = [
    "buna",           # Bună / Salut
    "multumesc",      # Mulțumesc
    "te_rog",         # Te rog
    "da",             # Da
    "nu",             # Nu
    "ajutor",         # Ajutor
    "cum_te_cheama",  # Cum te cheamă?
    "bine",           # Bine
    "rau",            # Rău
    "eu",             # Eu
    "tu",             # Tu
    "iubire",         # Iubire
    "familie",        # Familie
    "apa",            # Apă
    "mancare",        # Mâncare
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls for i, cls in enumerate(CLASSES)}

# ──────────────────────── Parametri antrenare ───────────────────
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 15
REDUCE_LR_PATIENCE = 7
VALIDATION_SPLIT = 0.2

# ──────────────────────── Parametri augmentare ──────────────────
AUG_NOISE_STD = 0.005       # deviație standard zgomot Gaussian
AUG_SCALE_RANGE = (0.9, 1.1)  # scalare spațială
AUG_ROTATION_MAX = 15       # grade, rotație 2D
AUG_TIME_STRETCH = (0.8, 1.2)  # variație viteză temporală
AUG_MIRROR_PROB = 0.5       # probabilitate oglindire orizontală

# ──────────────────────── Colectare date ────────────────────────
SAMPLES_PER_CLASS = 50      # secvențe de colectat per semn
COLLECTION_COUNTDOWN = 3    # secunde numărătoare inversă înainte de înregistrare
