"""
Strategii de augmentare a datelor pentru secvențe de landmark-uri.

Provocarea: Nu există un dataset masiv pentru LSR (Limbajul Semnelor Românesc).
Soluția: Augmentare agresivă + Transfer Learning de la dataset-uri internaționale.

Tehnici implementate:
  1. Zgomot Gaussian — simulează imprecizia detecției
  2. Scalare spațială — simulează variații în distanța de la cameră
  3. Rotație 2D — simulează înclinarea corpului
  4. Oglindire orizontală — inversare stânga-dreapta (cu swap mâini)
  5. Variație temporală — accelerare/decelerare a execuției semnului
  6. Dropout de landmarks — simulează ocluzii parțiale
  test

Transfer Learning:
  - Se antrenează pe WLASL (Word-Level ASL) pentru semne similare
  - Se fac freeze pe straturile inferioare (feature extraction)
  - Se fine-tune pe datele LSR colectate
"""

import numpy as np
from scipy.interpolate import interp1d

from config import (
    SEQUENCE_LENGTH, TOTAL_FEATURES,
    NUM_HAND_LANDMARKS, COORDS_PER_LANDMARK,
    AUG_NOISE_STD, AUG_SCALE_RANGE, AUG_ROTATION_MAX,
    AUG_TIME_STRETCH, AUG_MIRROR_PROB
)


def augment_noise(sequence, std=AUG_NOISE_STD):
    """
    Adaugă zgomot Gaussian pe coordonate.

    Motivație: MediaPipe are o precizie finită; zgomotul face modelul
    robust la variațiile de detecție cadru-cu-cadru.

    Args:
        sequence: (seq_len, 204)
        std: deviația standard a zgomotului
    Returns:
        secvență cu zgomot adăugat
    """
    noise = np.random.normal(0, std, sequence.shape).astype(np.float32)
    # Nu adaugă zgomot pe coordonatele care sunt deja zero (missing landmarks)
    mask = (sequence != 0).astype(np.float32)
    return sequence + noise * mask


def augment_scale(sequence, scale_range=AUG_SCALE_RANGE):
    """
    Scalare uniformă a tuturor coordonatelor.

    Simulează variația distanței față de cameră.
    Se aplică separat pe x, y (z rămâne relativ la cameră).
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    scaled = sequence.copy()
    # Scalare pe x și y (indici 0, 1 din fiecare grupă de 3)
    for i in range(0, TOTAL_FEATURES, 3):
        scaled[:, i] *= scale      # x
        scaled[:, i + 1] *= scale  # y
    return scaled


def augment_rotation_2d(sequence, max_angle=AUG_ROTATION_MAX):
    """
    Rotație 2D (în planul imaginii) a tuturor landmark-urilor.

    Se rotesc coordonatele (x, y) în jurul centrului.
    Simulează înclinarea capului/corpului.
    """
    angle = np.random.uniform(-max_angle, max_angle)
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    rotated = sequence.copy()
    for i in range(0, TOTAL_FEATURES, 3):
        x = sequence[:, i]
        y = sequence[:, i + 1]
        # Rotație în jurul originii (0.5, 0.5) — centrul cadrului normalizat
        rotated[:, i] = cos_t * (x - 0.5) - sin_t * (y - 0.5) + 0.5
        rotated[:, i + 1] = sin_t * (x - 0.5) + cos_t * (y - 0.5) + 0.5

    return rotated


def augment_mirror(sequence):
    """
    Oglindire orizontală (stânga ↔ dreapta).

    IMPORTANT: La oglindire, mâna stângă devine dreapta și invers.
    Se inversează coordonatele x (1 - x) și se fac swap pe segmentele
    mâinilor din vectorul de features.
    """
    mirrored = sequence.copy()

    hand_size = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK  # 63

    # Swap mâna stângă (0:63) cu mâna dreaptă (63:126)
    left_hand = mirrored[:, :hand_size].copy()
    right_hand = mirrored[:, hand_size:hand_size * 2].copy()
    mirrored[:, :hand_size] = right_hand
    mirrored[:, hand_size:hand_size * 2] = left_hand

    # Inversare coordonata x (1 - x) pentru toate landmark-urile
    for i in range(0, TOTAL_FEATURES, 3):
        mask = mirrored[:, i] != 0  # nu oglindi zerouri (missing)
        mirrored[mask, i] = 1.0 - mirrored[mask, i]

    return mirrored


def augment_time_stretch(sequence, stretch_range=AUG_TIME_STRETCH):
    """
    Variație temporală — simulează execuția mai rapidă sau mai lentă a semnului.

    Strategie: Interpolarea secvenței la o lungime diferită, apoi
    re-sampling la SEQUENCE_LENGTH cadre.

    Ex: stretch=0.8 → semnul e executat cu 20% mai repede
        stretch=1.2 → semnul e executat cu 20% mai lent
    """
    factor = np.random.uniform(stretch_range[0], stretch_range[1])
    seq_len = sequence.shape[0]
    new_len = max(2, int(seq_len * factor))

    # Interpolarea pe axa temporală
    x_original = np.linspace(0, 1, seq_len)
    x_new = np.linspace(0, 1, new_len)

    stretched = np.zeros((new_len, TOTAL_FEATURES), dtype=np.float32)
    for feat_idx in range(TOTAL_FEATURES):
        if np.all(sequence[:, feat_idx] == 0):
            continue
        f = interp1d(x_original, sequence[:, feat_idx], kind='linear',
                     fill_value='extrapolate')
        stretched[:, feat_idx] = f(x_new)

    # Re-sample la lungimea originală
    if new_len == seq_len:
        return stretched

    x_stretched = np.linspace(0, 1, new_len)
    x_target = np.linspace(0, 1, SEQUENCE_LENGTH)
    result = np.zeros((SEQUENCE_LENGTH, TOTAL_FEATURES), dtype=np.float32)
    for feat_idx in range(TOTAL_FEATURES):
        if np.all(stretched[:, feat_idx] == 0):
            continue
        f = interp1d(x_stretched, stretched[:, feat_idx], kind='linear',
                     fill_value='extrapolate')
        result[:, feat_idx] = f(x_target)

    return result


def augment_landmark_dropout(sequence, drop_prob=0.05):
    """
    Dropout aleatoriu de landmarks individuale.

    Simulează ocluzii parțiale (mâna acoperă parțial fața, etc.).
    Se setează la zero grupuri de (x, y, z) cu probabilitatea drop_prob.
    """
    dropped = sequence.copy()
    num_landmarks = TOTAL_FEATURES // 3

    for t in range(dropped.shape[0]):
        for lm in range(num_landmarks):
            if np.random.random() < drop_prob:
                idx = lm * 3
                dropped[t, idx:idx + 3] = 0.0

    return dropped


def apply_augmentation(sequence, augmentations=None):
    """
    Aplică un pipeline de augmentări aleatorii pe o secvență.

    Args:
        sequence: np.ndarray shape (SEQUENCE_LENGTH, TOTAL_FEATURES)
        augmentations: lista de augmentări de aplicat (None = toate)
    Returns:
        secvență augmentată
    """
    if augmentations is None:
        augmentations = ['noise', 'scale', 'rotation', 'mirror', 'time_stretch',
                         'dropout']

    aug = sequence.copy()

    for aug_name in augmentations:
        if aug_name == 'noise' and np.random.random() < 0.7:
            aug = augment_noise(aug)
        elif aug_name == 'scale' and np.random.random() < 0.5:
            aug = augment_scale(aug)
        elif aug_name == 'rotation' and np.random.random() < 0.5:
            aug = augment_rotation_2d(aug)
        elif aug_name == 'mirror' and np.random.random() < AUG_MIRROR_PROB:
            aug = augment_mirror(aug)
        elif aug_name == 'time_stretch' and np.random.random() < 0.5:
            aug = augment_time_stretch(aug)
        elif aug_name == 'dropout' and np.random.random() < 0.3:
            aug = augment_landmark_dropout(aug)

    return aug


def generate_augmented_dataset(X, y, augment_factor=5):
    """
    Generează un dataset augmentat.

    Args:
        X: np.ndarray shape (N, seq_len, features) — date originale
        y: np.ndarray shape (N,) — etichete
        augment_factor: de câte ori se multiplică datasetul

    Returns:
        X_aug, y_aug: dataset augmentat
    """
    X_aug = [X]
    y_aug = [y]

    for _ in range(augment_factor - 1):
        X_batch = np.array([apply_augmentation(seq) for seq in X])
        X_aug.append(X_batch)
        y_aug.append(y.copy())

    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


# ────────────────────── Transfer Learning ──────────────────────

def map_wlasl_to_lsr():
    """
    Mapare între semnele WLASL (ASL) și LSR (Limba Semnelor Românească).

    Multe semne de bază (da, nu, mulțumesc, ajutor) au forme similare
    în diferite limbi ale semnelor. Această mapare permite transfer
    learning parțial.

    Returns:
        dict: {semn_lsr: semn_wlasl} pentru semne cu echivalent ASL
    """
    return {
        "da":        "yes",
        "nu":        "no",
        "multumesc": "thank you",
        "ajutor":    "help",
        "bine":      "good",
        "rau":       "bad",
        "eu":        "me",
        "tu":        "you",
        "iubire":    "love",
        "familie":   "family",
        "apa":       "water",
        "mancare":   "food",
    }


def prepare_transfer_data(wlasl_data_dir, mapping):
    """
    Pregătește datele WLASL pre-procesate pentru transfer learning.

    Presupune că datele WLASL au fost deja procesate prin MediaPipe
    și salvate ca .npy cu aceeași structură de features.

    Pașii:
      1. Încarcă secvențele WLASL pentru semnele mapate
      2. Le re-etichetează cu indicii LSR corespunzători
      3. Returnează (X_transfer, y_transfer) gata de antrenare

    Args:
        wlasl_data_dir: calea către datele WLASL procesate
        mapping: dict de la map_wlasl_to_lsr()
    Returns:
        (X, y): date pregătite sau (None, None) dacă nu există date
    """
    import os
    from config import CLASS_TO_IDX

    X_transfer = []
    y_transfer = []

    for lsr_sign, asl_sign in mapping.items():
        asl_dir = os.path.join(wlasl_data_dir, asl_sign)
        if not os.path.exists(asl_dir):
            continue

        lsr_idx = CLASS_TO_IDX.get(lsr_sign)
        if lsr_idx is None:
            continue

        for npy_file in os.listdir(asl_dir):
            if not npy_file.endswith('.npy'):
                continue
            filepath = os.path.join(asl_dir, npy_file)
            seq = np.load(filepath)
            if seq.shape == (SEQUENCE_LENGTH, TOTAL_FEATURES):
                X_transfer.append(seq)
                y_transfer.append(lsr_idx)

    if not X_transfer:
        return None, None

    return np.array(X_transfer), np.array(y_transfer)
