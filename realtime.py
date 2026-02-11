"""
Inferenta in timp real -- Traducere Limbaj Semne Romanesc -> Text.

Functionare:
  1. Deschide camera web (OpenCV)
  2. La fiecare cadru: extrage landmarks cu MediaPipe Holistic
  3. Acumuleaza un buffer de 30 cadre (sliding window)
  4. La fiecare 30 de cadre noi: ruleaza predictia prin model
  5. Afiseaza traducerea pe ecran cu nivel de confidenta

Utilizare:
  python realtime.py
  python realtime.py --model models/lsr_model.keras --threshold 0.7
  python realtime.py --demo
"""

import os
import sys
import time
import collections

import cv2
import numpy as np
import tensorflow as tf

from config import (
    SEQUENCE_LENGTH, TOTAL_FEATURES, CLASSES, IDX_TO_CLASS,
    MODEL_PATH, NUM_CLASSES
)
from utils import (
    init_holistic, extract_landmarks, normalize_sequence,
    draw_landmarks_on_frame
)


class RealtimeTranslator:
    """Motor de traducere in timp real pentru limbajul semnelor."""

    def __init__(self, model_path, prediction_interval=30,
                 confidence_threshold=0.6, history_size=5):
        print(f"Se incarca modelul: {model_path}")
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects=self._get_custom_objects()
        )
        print("Model incarcat cu succes!")

        self.prediction_interval = prediction_interval
        self.confidence_threshold = confidence_threshold
        self.buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
        self.frame_count = 0
        self.current_prediction = None
        self.current_confidence = 0.0
        self.history = collections.deque(maxlen=history_size)
        self.last_prediction_time = 0

        # Warmup -- prima predictie e mai lenta (compilare graf)
        dummy = np.zeros((1, SEQUENCE_LENGTH, TOTAL_FEATURES), dtype=np.float32)
        self.model.predict(dummy, verbose=0)
        print("Model warmup complet.")

    def _get_custom_objects(self):
        """Returneaza obiectele custom necesare pentru load_model."""
        from model import TemporalAttention, PositionalEncoding, GraphConvLayer
        return {
            'TemporalAttention': TemporalAttention,
            'PositionalEncoding': PositionalEncoding,
            'GraphConvLayer': GraphConvLayer,
        }

    def process_frame(self, landmarks):
        """
        Proceseaza un cadru nou.

        Args:
            landmarks: vector de features (204,) din extract_landmarks()
        Returns:
            (prediction_text, confidence) sau (None, 0) daca nu e gata
        """
        self.buffer.append(landmarks)
        self.frame_count += 1

        # Verifica daca avem suficiente cadre si e timpul pentru predictie
        if (len(self.buffer) == SEQUENCE_LENGTH and
                self.frame_count % self.prediction_interval == 0):
            return self._predict()

        return self.current_prediction, self.current_confidence

    def _predict(self):
        """Ruleaza predictia pe buffer-ul curent."""
        # Construieste secventa din buffer
        sequence = np.array(list(self.buffer), dtype=np.float32)

        # Normalizare (invarianta pozitie/scala)
        sequence = normalize_sequence(sequence)

        # Predictie
        input_data = sequence[np.newaxis, ...]  # (1, 30, 204)
        start_time = time.time()
        probs = self.model.predict(input_data, verbose=0)[0]
        inference_time = time.time() - start_time

        # Extrage clasa cu cea mai mare probabilitate
        predicted_idx = np.argmax(probs)
        confidence = probs[predicted_idx]

        if confidence >= self.confidence_threshold:
            prediction = IDX_TO_CLASS[predicted_idx]

            # Adauga la istoric doar daca difera de ultima predictie
            if (not self.history or self.history[-1] != prediction):
                self.history.append(prediction)

            self.current_prediction = prediction
            self.current_confidence = float(confidence)
        else:
            self.current_prediction = None
            self.current_confidence = float(confidence)

        self.last_prediction_time = inference_time
        return self.current_prediction, self.current_confidence


def draw_ui(frame, translator, fps):
    """
    Deseneaza interfata utilizator pe cadru.

    Componente:
      - Banner superior: predictia curenta + confidenta
      - Bara de confidenta vizuala
      - Istoric traduceri
      - FPS si timp inferenta
    """
    h, w = frame.shape[:2]

    # -- Banner superior (fundal semi-transparent) --
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    if translator.current_prediction:
        # Numele semnului -- mare, vizibil
        label = translator.current_prediction.upper().replace('_', ' ')
        conf_pct = translator.current_confidence * 100
        text = f"{label} ({conf_pct:.1f}%)"

        # Culoare bazata pe confidenta
        if translator.current_confidence > 0.85:
            color = (0, 255, 0)     # verde -- confidenta mare
        elif translator.current_confidence > 0.7:
            color = (0, 255, 255)   # galben -- confidenta medie
        else:
            color = (0, 165, 255)   # portocaliu -- confidenta scazuta

        cv2.putText(frame, text, (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Bara de confidenta
        bar_width = int(translator.current_confidence * (w - 30))
        cv2.rectangle(frame, (15, 65), (15 + bar_width, 80), color, -1)
        cv2.rectangle(frame, (15, 65), (w - 15, 80), (100, 100, 100), 1)
    else:
        cv2.putText(frame, "Asteapta detectie...", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        # Bara de progres buffer
        progress = len(translator.buffer) / SEQUENCE_LENGTH
        bar_width = int(progress * (w - 30))
        cv2.rectangle(frame, (15, 65), (15 + bar_width, 80), (100, 100, 100), -1)
        cv2.rectangle(frame, (15, 65), (w - 15, 80), (60, 60, 60), 1)

    # -- Banner inferior: istoric + info --
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 80), (w, h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0)

    # Istoric traduceri
    if translator.history:
        history_text = " -> ".join(list(translator.history)[-5:])
        cv2.putText(frame, f"Istoric: {history_text}", (15, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # FPS si timp inferenta
    info = f"FPS: {fps:.0f}"
    if translator.last_prediction_time > 0:
        info += f" | Inferenta: {translator.last_prediction_time*1000:.0f}ms"
    cv2.putText(frame, info, (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Instructiuni
    cv2.putText(frame, "Q/ESC: Iesire | R: Reset buffer",
                (w - 300, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    return frame


def run_realtime(model_path, camera_idx=0, threshold=0.6):
    """
    Bucla principala de inferenta in timp real.

    Args:
        model_path: calea catre modelul .keras
        camera_idx: indexul camerei (default: 0)
        threshold: prag minim de confidenta
    """
    # Verifica existenta modelului
    if not os.path.exists(model_path):
        print(f"EROARE: Modelul nu a fost gasit la: {model_path}")
        print("Antreneaza mai intai modelul cu: python train_model.py")
        print("\nSau ruleaza in modul DEMO (fara model antrenat):")
        print("  python realtime.py --demo")
        sys.exit(1)

    # Initializare
    translator = RealtimeTranslator(
        model_path=model_path,
        confidence_threshold=threshold
    )
    holistic = init_holistic()
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print("EROARE: Nu se poate deschide camera!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nSistem pornit! Fa semne in fata camerei.")
    print("Taste: Q/ESC = iesire, R = reset buffer\n")

    fps_timer = time.time()
    fps = 0
    fps_frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # oglindire pentru naturalete

            # Extractie landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Extrage features
            landmarks = extract_landmarks(results)

            # Procesare predictie
            translator.process_frame(landmarks)

            # Vizualizare landmarks
            frame = draw_landmarks_on_frame(frame, results)

            # Calcul FPS
            fps_frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = time.time()

            # Desenare UI
            frame = draw_ui(frame, translator, fps)

            # Afisare
            cv2.imshow('LSR - Traducere Limbaj Semne', frame)

            # Control tastatura
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                translator.buffer.clear()
                translator.current_prediction = None
                translator.current_confidence = 0.0
                translator.frame_count = 0
                print("Buffer resetat.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
        print("\nSistem oprit.")


def run_demo_mode(camera_idx=0):
    """
    Mod DEMO -- afiseaza landmarks fara model antrenat.
    Util pentru a verifica ca pipeline-ul de extractie functioneaza.
    """
    print("MOD DEMO -- Vizualizare landmarks fara predictie")
    print("Acest mod verifica pipeline-ul MediaPipe.\n")

    holistic = init_holistic()
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print("EROARE: Nu se poate deschide camera!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    fps_timer = time.time()
    fps = 0
    fps_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            landmarks = extract_landmarks(results)
            buffer.append(landmarks)

            frame = draw_landmarks_on_frame(frame, results)

            # Info overlay
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
            frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

            cv2.putText(frame, "MOD DEMO -- Pipeline Verificare",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Afiseaza ce a detectat
            detected = []
            if results.left_hand_landmarks:
                detected.append("Mana stanga")
            if results.right_hand_landmarks:
                detected.append("Mana dreapta")
            if results.pose_landmarks:
                detected.append("Pose")
            if results.face_landmarks:
                detected.append("Fata")

            det_text = "Detectat: " + ", ".join(detected) if detected else "Nimic detectat"
            cv2.putText(frame, det_text, (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Features non-zero
            non_zero = np.count_nonzero(landmarks)
            cv2.putText(frame, f"Features active: {non_zero}/{TOTAL_FEATURES} | "
                        f"Buffer: {len(buffer)}/{SEQUENCE_LENGTH}",
                        (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            # FPS
            fps_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_count / (time.time() - fps_timer)
                fps_count = 0
                fps_timer = time.time()
            cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('LSR - Demo Mode', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Inferenta timp real -- Limbajul Semnelor Romanesc"
    )
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help=f'Calea modelului (default: {MODEL_PATH})')
    parser.add_argument('--camera', type=int, default=0,
                        help='Index camera (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Prag confidenta (default: 0.6)')
    parser.add_argument('--demo', action='store_true',
                        help='Mod demo -- doar vizualizare landmarks, fara model')
    args = parser.parse_args()

    if args.demo:
        run_demo_mode(args.camera)
    else:
        run_realtime(args.model, args.camera, args.threshold)


if __name__ == "__main__":
    main()
