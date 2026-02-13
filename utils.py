"""
Utilități pentru extracția și procesarea landmark-urilor MediaPipe Holistic.

Pipeline de transformare coordonate -> vector de intrare:
  1. MediaPipe Holistic proceseaza cadrul BGR -> detecteaza maini, pose, fata
  2. Se extrag coordonatele (x, y, z) normalizate [0, 1] pentru fiecare landmark
  3. Se aplica normalizare relativa la punctul de referinta (umar drept)
  4. Landmark-urile lipsa sunt completate cu zerouri
  5. Vectorul final per cadru: [mana_stanga(63) | mana_dreapta(63) | pose(18) | fata(60)]

NOTA: Importurile MediaPipe sunt lazy (doar in functiile care au nevoie de camera).
      Functiile de normalizare/padding functioneaza fara MediaPipe instalat.
"""

import numpy as np
from config import (
    POSE_INDICES, FACE_INDICES,
    NUM_HAND_LANDMARKS, COORDS_PER_LANDMARK,
    TOTAL_FEATURES
)

# ═══════════════════════════════════════════════════════════════
#  MediaPipe — importuri lazy (doar pentru camera/vizualizare)
# ═══════════════════════════════════════════════════════════════

_mp_holistic = None
_mp_drawing = None
_mp_drawing_styles = None


def _patch_protobuf():
    """
    Patch de compatibilitate: protobuf 5.x a eliminat SymbolDatabase.GetPrototype()
    care e folosit intern de mediapipe 0.10.14. Acest patch il readauga.
    """
    from google.protobuf import symbol_database
    if not hasattr(symbol_database.SymbolDatabase, 'GetPrototype'):
        from google.protobuf import message_factory
        def _get_prototype(self, descriptor):
            return message_factory.GetMessageClass(descriptor)
        symbol_database.SymbolDatabase.GetPrototype = _get_prototype


def _load_mediapipe():
    """Incarca MediaPipe la prima utilizare (lazy import)."""
    global _mp_holistic, _mp_drawing, _mp_drawing_styles
    if _mp_holistic is not None:
        return

    _patch_protobuf()
    import mediapipe as mp
    _mp_holistic = mp.solutions.holistic
    _mp_drawing = mp.solutions.drawing_utils
    _mp_drawing_styles = mp.solutions.drawing_styles


def init_holistic(static_mode=False, min_detection_confidence=0.5,
                  min_tracking_confidence=0.5):
    """Initializeaza MediaPipe Holistic cu parametrii specificati."""
    _load_mediapipe()
    return _mp_holistic.Holistic(
        static_image_mode=static_mode,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_landmarks(results):
    """
    Extrage si concateneaza landmark-urile din rezultatele MediaPipe Holistic.

    Transformarea coordonatelor:
      - MediaPipe returneaza (x, y, z) normalizate:
        x, y in [0, 1] relativ la dimensiunea imaginii
        z = profunzime relativa la camera (valori mici = aproape)
      - Le concatenam intr-un vector plat de 204 valori per cadru
      - Landmark-urile nedetectate -> vector de zerouri

    Returns:
        np.ndarray: vector shape (204,) sau (TOTAL_FEATURES,)
    """
    # Mana stanga — 21 landmarks x 3
    if results.left_hand_landmarks:
        left_hand = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.left_hand_landmarks.landmark
        ]).flatten()
    else:
        left_hand = np.zeros(NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK)

    # Mana dreapta — 21 landmarks x 3
    if results.right_hand_landmarks:
        right_hand = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.right_hand_landmarks.landmark
        ]).flatten()
    else:
        right_hand = np.zeros(NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK)

    # Pose (trunchi superior) — 6 landmarks x 3
    if results.pose_landmarks:
        pose = np.array([
            [results.pose_landmarks.landmark[i].x,
             results.pose_landmarks.landmark[i].y,
             results.pose_landmarks.landmark[i].z]
            for i in POSE_INDICES
        ]).flatten()
    else:
        pose = np.zeros(len(POSE_INDICES) * COORDS_PER_LANDMARK)

    # Fata (expresii) — 20 landmarks x 3
    if results.face_landmarks:
        face = np.array([
            [results.face_landmarks.landmark[i].x,
             results.face_landmarks.landmark[i].y,
             results.face_landmarks.landmark[i].z]
            for i in FACE_INDICES
        ]).flatten()
    else:
        face = np.zeros(len(FACE_INDICES) * COORDS_PER_LANDMARK)

    # Concatenare: [mana_stanga | mana_dreapta | pose | fata]
    feature_vector = np.concatenate([left_hand, right_hand, pose, face])

    assert feature_vector.shape[0] == TOTAL_FEATURES, (
        f"Dimensiune incorecta: {feature_vector.shape[0]} != {TOTAL_FEATURES}"
    )
    return feature_vector


# ═══════════════════════════════════════════════════════════════
#  Functii pure (numpy only — nu necesita MediaPipe)
# ═══════════════════════════════════════════════════════════════

def normalize_sequence(sequence):
    """
    Normalizare relativa a unei secvente de landmark-uri.

    Strategia:
      1. Se foloseste centrul umerilor ca punct de referinta (origin)
      2. Se scade originea din toate coordonatele -> invarianta la translatie
      3. Se normalizeaza scala folosind distanta inter-umeri

    Args:
        sequence: np.ndarray shape (seq_len, 204)
    Returns:
        np.ndarray: secventa normalizata, shape (seq_len, 204)
    """
    normalized = sequence.copy().astype(np.float32)

    for t in range(normalized.shape[0]):
        frame = normalized[t]
        if np.all(frame == 0):
            continue

        # Extrage coordonatele umerilor din sectiunea pose
        # Pose incepe la indexul 126 (dupa 63+63 pentru maini)
        pose_start = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK * 2  # 126
        # Umar stang = POSE_INDICES[0] -> primele 3 valori din pose
        shoulder_left = frame[pose_start:pose_start + 3]
        # Umar drept = POSE_INDICES[1] -> urmatoarele 3 valori
        shoulder_right = frame[pose_start + 3:pose_start + 6]

        # Punct de referinta: centrul umerilor
        origin = (shoulder_left + shoulder_right) / 2.0

        # Scala: distanta inter-umeri
        shoulder_dist = np.linalg.norm(shoulder_left - shoulder_right)
        if shoulder_dist < 1e-6:
            shoulder_dist = 1.0  # evita impartirea la zero

        # Aplica normalizarea pe grupe de 3 (x, y, z)
        for i in range(0, len(frame), 3):
            if not np.all(frame[i:i+3] == 0):  # nu normaliza zerouri (missing)
                normalized[t, i:i+3] = (frame[i:i+3] - origin) / shoulder_dist

    return normalized


def draw_landmarks_on_frame(frame, results):
    """Deseneaza landmark-urile detectate pe cadru pentru feedback vizual."""
    _load_mediapipe()
    # Pose
    if results.pose_landmarks:
        _mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, _mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=_mp_drawing_styles.get_default_pose_landmarks_style()
        )
    # Mana stanga
    if results.left_hand_landmarks:
        _mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, _mp_holistic.HAND_CONNECTIONS,
            _mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
            _mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1),
        )
    # Mana dreapta
    if results.right_hand_landmarks:
        _mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, _mp_holistic.HAND_CONNECTIONS,
            _mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            _mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1),
        )
    # Fata (simplificat — doar conturul)
    if results.face_landmarks:
        _mp_drawing.draw_landmarks(
            frame, results.face_landmarks, _mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=_mp_drawing_styles.get_default_face_mesh_contours_style()
        )
    return frame


def pad_or_truncate_sequence(sequence, target_length):
    """
    Asigura ca secventa are exact target_length cadre.
    - Prea scurta -> padding cu zerouri la final
    - Prea lunga -> trunchiere la final
    """
    current_length = len(sequence)
    if current_length == target_length:
        return np.array(sequence)
    elif current_length > target_length:
        return np.array(sequence[:target_length])
    else:
        padding = np.zeros((target_length - current_length, TOTAL_FEATURES))
        return np.vstack([sequence, padding])