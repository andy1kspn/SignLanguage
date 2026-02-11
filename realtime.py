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
import keras

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
        self.model = keras.models.load_model(
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
    
    # Inițializează MediaPipe
    holistic = init_holistic()
    
    # Încearcă să găsească o cameră funcțională
    cap = None
    camera_sources = []
    
    # Dacă camera_idx este string, încearcă ca IP camera
    if isinstance(camera_idx, str):
        print(f"Incercare conectare IP camera: {camera_idx}")
        cap = cv2.VideoCapture(camera_idx)
        camera_sources.append(camera_idx)
    else:
        # Încearcă mai multe indici de cameră
        camera_sources = [camera_idx, 0, 1, 2, 3]  # Încearcă camera specificată + backup-uri
        
        for idx in camera_sources:
            print(f"Incercare camera index {idx}...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # Test dacă camera poate citi cadre
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"OK Camera {idx} functioneaza!")
                    camera_idx = idx
                    break
                else:
                    print(f"Camera {idx} se deschide dar nu poate citi cadre")
                    cap.release()
                    cap = None
            else:
                print(f"Camera {idx} nu se poate deschide")
                if cap:
                    cap.release()
                cap = None

    if not cap or not cap.isOpened():
        print("\nATENTIE: Nu s-a gasit nicio camera functionala!")
        print("\nSolutii pentru camera de telefon:")
        print("1. DroidCam (recomandat):")
        print("   - Instaleaza DroidCam pe telefon si PC")
        print("   - Conecteaza prin USB sau WiFi")
        print("   - Ruleaza: python realtime.py --camera http://192.168.1.100:4747/video")
        print("\n2. IP Webcam:")
        print("   - Instaleaza IP Webcam pe telefon")
        print("   - Noteaza IP-ul afisat in aplicatie")
        print("   - Ruleaza: python realtime.py --camera http://IP:8080/video")
        print("\n3. Camera USB:")
        print("   - Conecteaza telefonul prin cablu USB")
        print("   - Activeaza 'USB Debugging' si 'Camera USB' in setari")
        print("   - Ruleaza din nou aplicatia")
        print(f"\nCamere incercate: {camera_sources}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Upscale la rezolutie mai mare pentru afisare
    display_width, display_height = 1280, 960  # 2x upscale

    # Fereastra adaptiva cu pastrarea aspectului
    cv2.namedWindow('LSR - Traducere Limbaj Semne', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow('LSR - Traducere Limbaj Semne', display_width, display_height)

    print("\nSistem pornit! Fa semne in fata camerei.")
    print("Taste: Q/ESC = iesire | R = reset | +/- = zoom | L = landmarks | H = hide UI | F = fullscreen\n")

    fps_timer = time.time()
    fps = 0
    fps_frame_count = 0
    zoom_level = 1.0
    show_landmarks = True
    show_ui = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # oglindire pentru naturalete
            
            # Upscale imaginea pentru calitate mai buna (2x)
            frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_CUBIC)
            
            # Imbunatatire contrast adaptiv (CLAHE) - face imaginea mai clara
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            frame = cv2.merge([l, a, b])
            frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
            
            # Sharpening subtil pentru claritate
            kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
            frame = cv2.filter2D(frame, -1, kernel)

            # Aplica zoom daca e necesar
            if zoom_level != 1.0:
                h, w = frame.shape[:2]
                new_w, new_h = int(w * zoom_level), int(h * zoom_level)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Crop la centru pentru a pastra dimensiunea originala
                if zoom_level > 1.0:
                    start_x = (new_w - w) // 2
                    start_y = (new_h - h) // 2
                    frame = frame[start_y:start_y+h, start_x:start_x+w]
                else:
                    # Padding pentru zoom out
                    pad_x = (w - new_w) // 2
                    pad_y = (h - new_h) // 2
                    frame = cv2.copyMakeBorder(frame, pad_y, pad_y, pad_x, pad_x, 
                                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)

            # Extractie landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Extrage features
            landmarks = extract_landmarks(results)

            # Procesare predictie
            translator.process_frame(landmarks)

            # Vizualizare landmarks doar daca e activat
            if show_landmarks:
                frame = draw_landmarks_on_frame(frame, results)

            # Calcul FPS
            fps_frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = time.time()

            # Desenare UI doar daca e activat
            if show_ui:
                frame = draw_ui(frame, translator, fps)
                
                # Adauga info zoom si landmarks
                h, w = frame.shape[:2]
                cv2.putText(frame, f"Zoom: {zoom_level*100:.0f}% | Landmarks: {'ON' if show_landmarks else 'OFF'}",
                            (w - 350, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

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
            elif key == ord('+') or key == ord('='):
                zoom_level = min(2.0, zoom_level + 0.1)
                print(f"Zoom: {zoom_level*100:.0f}%")
            elif key == ord('-') or key == ord('_'):
                zoom_level = max(0.5, zoom_level - 0.1)
                print(f"Zoom: {zoom_level*100:.0f}%")
            elif key == ord('l'):
                show_landmarks = not show_landmarks
                print(f"Landmarks: {'ON' if show_landmarks else 'OFF'}")
            elif key == ord('h'):
                show_ui = not show_ui
                print(f"UI: {'ON' if show_ui else 'OFF'}")
            elif key == ord('f'):
                # Toggle fullscreen
                prop = cv2.getWindowProperty('LSR - Traducere Limbaj Semne', cv2.WND_PROP_FULLSCREEN)
                if prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty('LSR - Traducere Limbaj Semne', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty('LSR - Traducere Limbaj Semne', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
        print("\nSistem oprit.")


def run_demo_mode(camera_idx=0):
    """
    Mod DEMO -- afiseaza landmarks fara model antrenat.
    Util pentru a verifica ca pipeline-ul de extractie functioneaza.
    Suporta camere multiple si IP camere (DroidCam, IP Webcam).
    """
    print("MOD DEMO -- Vizualizare landmarks fara predictie")
    print("Acest mod verifica pipeline-ul MediaPipe.\n")

    # Inițializează MediaPipe
    holistic = init_holistic()

    # Încearcă să găsească o cameră funcțională
    cap = None
    camera_sources = []

    # Dacă camera_idx este string, încearcă ca IP camera
    if isinstance(camera_idx, str):
        print(f"Incercare conectare IP camera: {camera_idx}")
        cap = cv2.VideoCapture(camera_idx)
        camera_sources.append(camera_idx)
    else:
        # Încearcă mai multe indici de cameră
        camera_sources = [camera_idx, 0, 1, 2, 3]  # Încearcă camera specificată + backup-uri

        for idx in camera_sources:
            print(f"Incercare camera index {idx}...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # Test dacă camera poate citi cadre
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"OK Camera {idx} functioneaza!")
                    camera_idx = idx
                    break
                else:
                    print(f"Camera {idx} se deschide dar nu poate citi cadre")
                    cap.release()
                    cap = None
            else:
                print(f"Camera {idx} nu se poate deschide")
                if cap:
                    cap.release()
                cap = None

    if not cap or not cap.isOpened():
        print("\nATENTIE: Nu s-a gasit nicio camera functionala!")
        print("\nSolutii pentru camera de telefon:")
        print("1. DroidCam (recomandat):")
        print("   - Instaleaza DroidCam pe telefon si PC")
        print("   - Conecteaza prin USB sau WiFi")
        print("   - Ruleaza: python realtime.py --demo --camera http://192.168.1.100:4747/video")
        print("\n2. IP Webcam:")
        print("   - Instaleaza IP Webcam pe telefon")
        print("   - Noteaza IP-ul afisat in aplicatie")
        print("   - Ruleaza: python realtime.py --demo --camera http://IP:8080/video")
        print("\n3. Camera USB:")
        print("   - Conecteaza telefonul prin cablu USB")
        print("   - Activeaza 'USB Debugging' si 'Camera USB' in setari")
        print("   - Ruleaza din nou aplicatia")
        print(f"\nCamere incercate: {camera_sources}")
        sys.exit(1)

    # Configurează camera pentru performanță optimă
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Verifică rezoluția efectivă
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera configurata: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

    # Upscale la rezolutie mai mare pentru afisare
    display_width, display_height = 1280, 960  # 2x upscale

    # Fereastra adaptiva cu pastrarea aspectului
    cv2.namedWindow('LSR - Demo Mode', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow('LSR - Demo Mode', display_width, display_height)

    buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    fps_timer = time.time()
    fps = 0
    fps_count = 0
    zoom_level = 1.0  # Control zoom (0.5 = 50%, 1.0 = 100%, 1.5 = 150%)
    show_landmarks = True  # Toggle pentru landmarks
    show_ui = True  # Toggle pentru interfata (text overlay)

    print("Taste: Q/ESC = iesire | +/- = zoom | L = landmarks | H = hide UI | F = fullscreen")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ATENTIE: Nu se pot citi cadre din camera!")
                break

            frame = cv2.flip(frame, 1)

            # Upscale imaginea pentru calitate mai buna (2x)
            frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_CUBIC)

            # Imbunatatire contrast adaptiv (CLAHE) - face imaginea mai clara
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            frame = cv2.merge([l, a, b])
            frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)

            # Sharpening subtil pentru claritate
            kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
            frame = cv2.filter2D(frame, -1, kernel)

            # Aplica zoom daca e necesar
            if zoom_level != 1.0:
                h, w = frame.shape[:2]
                new_w, new_h = int(w * zoom_level), int(h * zoom_level)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                # Crop la centru pentru a pastra dimensiunea originala
                if zoom_level > 1.0:
                    start_x = (new_w - w) // 2
                    start_y = (new_h - h) // 2
                    frame = frame[start_y:start_y+h, start_x:start_x+w]
                else:
                    # Padding pentru zoom out
                    pad_x = (w - new_w) // 2
                    pad_y = (h - new_h) // 2
                    frame = cv2.copyMakeBorder(frame, pad_y, pad_y, pad_x, pad_x,
                                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            landmarks = extract_landmarks(results)
            buffer.append(landmarks)

            # Deseneaza landmarks doar daca e activat
            if show_landmarks:
                frame = draw_landmarks_on_frame(frame, results)

            # Info overlay doar daca UI e activat
            if show_ui:
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 180), (30, 30, 30), -1)
                frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

                cv2.putText(frame, "MOD DEMO -- Pipeline Verificare",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Afiseaza sursa camerei
                if isinstance(camera_idx, str):
                    source_text = f"IP Camera: {camera_idx}"
                else:
                    source_text = f"Camera USB: Index {camera_idx}"
                cv2.putText(frame, source_text, (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

                # Afiseaza ce a detectat cu detalii despre fata
                detected = []
                face_details = []
                if results is not None:
                    if results.left_hand_landmarks:
                        detected.append("Mana stanga")
                    if results.right_hand_landmarks:
                        detected.append("Mana dreapta")
                    if results.pose_landmarks:
                        detected.append("Pose")
                    if results.face_landmarks:
                        detected.append("Fata")
                        # Analiza detaliata a fetei
                        face_landmarks = results.face_landmarks.landmark
                        
                        # Verifica daca ochii sunt deschisi (distanta intre pleoapele de sus si jos)
                        left_eye_top = face_landmarks[159].y  # pleoapă sus ochi stâng
                        left_eye_bottom = face_landmarks[145].y  # pleoapă jos ochi stâng
                        right_eye_top = face_landmarks[386].y  # pleoapă sus ochi drept
                        right_eye_bottom = face_landmarks[374].y  # pleoapă jos ochi drept
                        
                        left_eye_open = abs(left_eye_top - left_eye_bottom) > 0.01
                        right_eye_open = abs(right_eye_top - right_eye_bottom) > 0.01
                        
                        if left_eye_open and right_eye_open:
                            face_details.append("Ochi deschisi")
                        elif not left_eye_open and not right_eye_open:
                            face_details.append("Ochi inchisi")
                        else:
                            face_details.append("Ochi clipesc")
                        
                        # Verifica daca gura este deschisa
                        mouth_top = face_landmarks[13].y  # buza de sus
                        mouth_bottom = face_landmarks[14].y  # buza de jos
                        mouth_open = abs(mouth_top - mouth_bottom) > 0.015
                        
                        if mouth_open:
                            face_details.append("Gura deschisa")
                        else:
                            face_details.append("Gura inchisa")
                        
                        # Verifica inclinarea capului (folosind nasul si centrul fetei)
                        nose_tip = face_landmarks[1]
                        chin = face_landmarks[175]
                        head_tilt = abs(nose_tip.x - chin.x)
                        
                        if head_tilt > 0.05:
                            if nose_tip.x > chin.x:
                                face_details.append("Cap inclinat dreapta")
                            else:
                                face_details.append("Cap inclinat stanga")
                        else:
                            face_details.append("Cap drept")

                det_text = "Detectat: " + ", ".join(detected) if detected else "Nimic detectat"
                cv2.putText(frame, det_text, (15, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Afiseaza detaliile faciale
                if face_details:
                    face_text = "Mimica: " + " | ".join(face_details)
                    cv2.putText(frame, face_text, (15, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 1)

                # Features non-zero
                non_zero = np.count_nonzero(landmarks)
                cv2.putText(frame, f"Features active: {non_zero}/{TOTAL_FEATURES} | "
                            f"Buffer: {len(buffer)}/{SEQUENCE_LENGTH}",
                            (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

                # Zoom level
                cv2.putText(frame, f"Zoom: {zoom_level*100:.0f}% | Landmarks: {'ON' if show_landmarks else 'OFF'}",
                            (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

                # FPS
                cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Calculeaza FPS chiar daca UI e ascuns
                h, w = frame.shape[:2]

            # Calcul FPS
            fps_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_count / (time.time() - fps_timer)
                fps_count = 0
                fps_timer = time.time()

            cv2.imshow('LSR - Demo Mode', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('+') or key == ord('='):
                zoom_level = min(2.0, zoom_level + 0.1)
                print(f"Zoom: {zoom_level*100:.0f}%")
            elif key == ord('-') or key == ord('_'):
                zoom_level = max(0.5, zoom_level - 0.1)
                print(f"Zoom: {zoom_level*100:.0f}%")
            elif key == ord('l'):
                show_landmarks = not show_landmarks
                print(f"Landmarks: {'ON' if show_landmarks else 'OFF'}")
            elif key == ord('h'):
                show_ui = not show_ui
                print(f"UI: {'ON' if show_ui else 'OFF'}")
            elif key == ord('f'):
                # Toggle fullscreen
                prop = cv2.getWindowProperty('LSR - Demo Mode', cv2.WND_PROP_FULLSCREEN)
                if prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty('LSR - Demo Mode', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty('LSR - Demo Mode', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
    parser.add_argument('--camera', type=str, default="0",
                        help='Index camera (0,1,2...) sau URL IP camera (http://IP:PORT/video)')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Prag confidenta (default: 0.6)')
    parser.add_argument('--demo', action='store_true',
                        help='Mod demo -- doar vizualizare landmarks, fara model')
    args = parser.parse_args()

    # Convertește camera la int dacă este un număr
    camera = args.camera
    if camera.isdigit():
        camera = int(camera)

    if args.demo:
        run_demo_mode(camera)
    else:
        run_realtime(args.model, camera, args.threshold)


if __name__ == "__main__":
    main()
