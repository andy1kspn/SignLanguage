"""
Utilități pentru extracția și procesarea landmark-urilor MediaPipe.

Pipeline de transformare coordonate -> vector de intrare:
  1. MediaPipe proceseaza cadrul RGB -> detecteaza maini, pose, fata
  2. Se extrag coordonatele (x, y, z) normalizate [0, 1] pentru fiecare landmark
  3. Se aplica normalizare relativa la punctul de referinta (umar drept)
  4. Landmark-urile lipsa sunt completate cu zerouri
  5. Vectorul final per cadru: [mana_stanga(63) | mana_dreapta(63) | pose(18) | fata(60)]

NOTA: Suportă atât MediaPipe 0.10.14 (Holistic) cât și 0.10.30+ (task API separat).
"""

import numpy as np
import cv2
from config import (
    POSE_INDICES, FACE_INDICES,
    NUM_HAND_LANDMARKS, COORDS_PER_LANDMARK,
    TOTAL_FEATURES
)

# ═══════════════════════════════════════════════════════════════
#  MediaPipe — importuri lazy (doar pentru camera/vizualizare)
# ═══════════════════════════════════════════════════════════════

_hand_landmarker = None
_pose_landmarker = None
_face_landmarker = None
_mp_drawing = None
_mp_version = None


def _load_mediapipe():
    """Incarca MediaPipe la prima utilizare (lazy import)."""
    global _hand_landmarker, _pose_landmarker, _face_landmarker, _mp_drawing, _mp_version
    if _hand_landmarker is not None:
        return

    try:
        import mediapipe as mp
        _mp_version = mp.__version__
        
        # MediaPipe 0.10.30+ folosește task API
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe import solutions
        from mediapipe.framework.formats import landmark_pb2
        
        # Configurare HandLandmarker
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=None),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Configurare PoseLandmarker
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=None),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Configurare FaceLandmarker
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=None),
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        _hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        _pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        _face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        _mp_drawing = solutions.drawing_utils
        
        print(f"✓ MediaPipe {_mp_version} încărcat cu succes (task API)")
        
    except Exception as e:
        print(f"⚠ Eroare la încărcarea MediaPipe: {e}")
        _hand_landmarker = None
        _pose_landmarker = None
        _face_landmarker = None
        _mp_drawing = None


class HolisticResults:
    """Clasă pentru a stoca rezultatele de la toate landmarker-ele."""
    def __init__(self):
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.pose_landmarks = None
        self.face_landmarks = None


class HolisticProcessor:
    """Procesor unificat care combină Hand, Pose și Face landmarkers."""
    
    def __init__(self):
        _load_mediapipe()
        self.frame_count = 0
    
    def process(self, image_rgb):
        """Procesează un frame RGB și returnează rezultate unificate."""
        if _hand_landmarker is None:
            return None
        
        self.frame_count += 1
        timestamp_ms = self.frame_count * 33  # ~30 FPS
        
        # Convertește la MediaPipe Image
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        results = HolisticResults()
        
        try:
            # Detectează mâini
            hand_result = _hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            if hand_result.hand_landmarks and len(hand_result.hand_landmarks) > 0:
                # Identifică mâna stângă și dreaptă
                for idx, handedness in enumerate(hand_result.handedness):
                    if idx < len(hand_result.hand_landmarks):
                        hand_label = handedness[0].category_name
                        if hand_label == "Left":
                            results.left_hand_landmarks = hand_result.hand_landmarks[idx]
                        elif hand_label == "Right":
                            results.right_hand_landmarks = hand_result.hand_landmarks[idx]
            
            # Detectează pose
            pose_result = _pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                results.pose_landmarks = pose_result.pose_landmarks[0]
            
            # Detectează față
            face_result = _face_landmarker.detect_for_video(mp_image, timestamp_ms)
            if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
                results.face_landmarks = face_result.face_landmarks[0]
                
        except Exception as e:
            print(f"⚠ Eroare procesare: {e}")
            return None
        
        return results
    
    def close(self):
        """Închide toate landmarker-ele."""
        global _hand_landmarker, _pose_landmarker, _face_landmarker
        if _hand_landmarker:
            _hand_landmarker.close()
        if _pose_landmarker:
            _pose_landmarker.close()
        if _face_landmarker:
            _face_landmarker.close()
        _hand_landmarker = None
        _pose_landmarker = None
        _face_landmarker = None


def init_holistic(static_mode=False, min_detection_confidence=0.5,
                  min_tracking_confidence=0.5):
    """Initializeaza procesor unificat pentru landmark detection."""
    return HolisticProcessor()


def extract_landmarks(results):
    """
    Extrage si concateneaza landmark-urile din rezultatele MediaPipe.

    Transformarea coordonatelor:
      - MediaPipe returneaza (x, y, z) normalizate:
        x, y in [0, 1] relativ la dimensiunea imaginii
        z = profunzime relativa la camera (valori mici = aproape)
      - Le concatenam intr-un vector plat de 204 valori per cadru
      - Landmark-urile nedetectate -> vector de zerouri

    Returns:
        np.ndarray: vector shape (204,) sau (TOTAL_FEATURES,)
    """
    # Dacă results este None (MediaPipe nu e disponibil), returnează vector gol
    if results is None:
        return np.zeros(TOTAL_FEATURES, dtype=np.float32)

    # Mana stanga — 21 landmarks x 3
    if results.left_hand_landmarks:
        left_hand = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.left_hand_landmarks
        ]).flatten()
    else:
        left_hand = np.zeros(NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK)

    # Mana dreapta — 21 landmarks x 3
    if results.right_hand_landmarks:
        right_hand = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.right_hand_landmarks
        ]).flatten()
    else:
        right_hand = np.zeros(NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK)

    # Pose (trunchi superior) — 6 landmarks x 3
    if results.pose_landmarks:
        pose = np.array([
            [results.pose_landmarks[i].x,
             results.pose_landmarks[i].y,
             results.pose_landmarks[i].z]
            for i in POSE_INDICES
        ]).flatten()
    else:
        pose = np.zeros(len(POSE_INDICES) * COORDS_PER_LANDMARK)

    # Fata (expresii) — 20 landmarks x 3
    if results.face_landmarks:
        face = np.array([
            [results.face_landmarks[i].x,
             results.face_landmarks[i].y,
             results.face_landmarks[i].z]
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
    
    # Dacă results este None sau drawing nu e disponibil, returnează frame-ul nemodificat
    if results is None or _mp_drawing is None:
        return frame
    
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
