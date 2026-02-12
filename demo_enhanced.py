"""
Mod Demo Îmbunătățit pentru LSR - Vizualizare avansată și analiză landmarks.

Funcționalități:
- Înregistrare video și screenshot-uri
- Vizualizări multiple (heatmap, trails, skeleton)
- Analiză calitate și feedback în timp real
- Comparație cu dataset
- Statistici detaliate
- Meniu interactiv
- Teme vizuale
- Mini-jocuri și provocări
"""

import cv2
import numpy as np
import mediapipe as mp
import collections
import time
import sys
import os
import json
from datetime import datetime
from pathlib import Path

from config import (
    SEQUENCE_LENGTH, TOTAL_FEATURES, NUM_HAND_LANDMARKS,
    POSE_INDICES, FACE_INDICES, CLASSES, DATA_DIR
)

# ──────────────────────── Inițializare MediaPipe ────────────────────────
def init_holistic():
    """Inițializează MediaPipe Holistic cu parametri optimizați."""
    mp_holistic = mp.solutions.holistic
    return mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

def extract_landmarks(results):
    """Extrage landmarks în format vectorial (204 valori)."""
    landmarks = np.zeros(TOTAL_FEATURES)
    
    if results is None:
        return landmarks
    
    idx = 0
    
    # Mâna stângă (63 valori)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
            idx += 3
    else:
        idx += 63
    
    # Mâna dreaptă (63 valori)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
            idx += 3
    else:
        idx += 63
    
    # Pose - trunchi superior (18 valori)
    if results.pose_landmarks:
        for i in POSE_INDICES:
            lm = results.pose_landmarks.landmark[i]
            landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
            idx += 3
    else:
        idx += 18
    
    # Față - expresii (60 valori)
    if results.face_landmarks:
        for i in FACE_INDICES:
            lm = results.face_landmarks.landmark[i]
            landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
            idx += 3
    
    return landmarks


# ──────────────────────── Clase pentru funcționalități ────────────────────────

class VideoRecorder:
    """Gestionează înregistrarea video."""
    
    def __init__(self, output_dir="recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.writer = None
        self.is_recording = False
        self.filename = None
        self.start_time = None
    
    def start(self, frame_width, frame_height, fps=30):
        """Începe înregistrarea."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = self.output_dir / f"demo_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(self.filename), fourcc, fps, 
                                      (frame_width, frame_height))
        self.is_recording = True
        self.start_time = time.time()
        return str(self.filename)
    
    def write(self, frame):
        """Scrie un frame."""
        if self.is_recording and self.writer:
            self.writer.write(frame)
    
    def stop(self):
        """Oprește înregistrarea."""
        if self.writer:
            self.writer.release()
            self.writer = None
        self.is_recording = False
        duration = time.time() - self.start_time if self.start_time else 0
        return self.filename, duration
    
    def get_duration(self):
        """Returnează durata înregistrării curente."""
        if self.is_recording and self.start_time:
            return time.time() - self.start_time
        return 0


class ScreenshotManager:
    """Gestionează capturi de ecran."""
    
    def __init__(self, output_dir="screenshots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.count = 0
    
    def save(self, frame):
        """Salvează un screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"screenshot_{timestamp}.png"
        cv2.imwrite(str(filename), frame)
        self.count += 1
        return str(filename)


class TrailEffect:
    """Creează efect de urmă pentru mâini."""
    
    def __init__(self, max_length=30):
        self.left_hand_trail = collections.deque(maxlen=max_length)
        self.right_hand_trail = collections.deque(maxlen=max_length)
        self.colors = [(255, 0, 255), (0, 255, 255)]  # Magenta, Cyan
    
    def update(self, results, frame_shape):
        """Actualizează trailurile cu noile poziții."""
        h, w = frame_shape[:2]
        
        if results and results.left_hand_landmarks:
            # Folosește vârful degetului mijlociu
            tip = results.left_hand_landmarks.landmark[12]
            self.left_hand_trail.append((int(tip.x * w), int(tip.y * h)))
        
        if results and results.right_hand_landmarks:
            tip = results.right_hand_landmarks.landmark[12]
            self.right_hand_trail.append((int(tip.x * w), int(tip.y * h)))
    
    def draw(self, frame):
        """Desenează trailurile pe frame."""
        # Trail mâna stângă
        for i in range(1, len(self.left_hand_trail)):
            alpha = i / len(self.left_hand_trail)
            thickness = int(2 + alpha * 3)
            color = tuple(int(c * alpha) for c in self.colors[0])
            cv2.line(frame, self.left_hand_trail[i-1], self.left_hand_trail[i],
                    color, thickness)
        
        # Trail mâna dreaptă
        for i in range(1, len(self.right_hand_trail)):
            alpha = i / len(self.right_hand_trail)
            thickness = int(2 + alpha * 3)
            color = tuple(int(c * alpha) for c in self.colors[1])
            cv2.line(frame, self.right_hand_trail[i-1], self.right_hand_trail[i],
                    color, thickness)
    
    def clear(self):
        """Șterge trailurile."""
        self.left_hand_trail.clear()
        self.right_hand_trail.clear()


class HeatmapGenerator:
    """Generează heatmap pentru activitatea mâinilor."""
    
    def __init__(self, frame_shape, decay=0.95):
        self.heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
        self.decay = decay
    
    def update(self, results, frame_shape):
        """Actualizează heatmap-ul."""
        # Decay
        self.heatmap *= self.decay
        
        h, w = frame_shape[:2]
        
        # Adaugă heat pentru mâna stângă
        if results and results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(self.heatmap, (x, y), 15, 1.0, -1)
        
        # Adaugă heat pentru mâna dreaptă
        if results and results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(self.heatmap, (x, y), 15, 1.0, -1)
    
    def get_overlay(self):
        """Returnează overlay-ul heatmap colorat."""
        # Normalizează
        normalized = np.clip(self.heatmap * 255, 0, 255).astype(np.uint8)
        # Aplică colormap
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        return colored
    
    def clear(self):
        """Resetează heatmap-ul."""
        self.heatmap.fill(0)


class QualityAnalyzer:
    """Analizează calitatea detecției."""
    
    def __init__(self):
        self.history = collections.deque(maxlen=30)
    
    def analyze(self, results, landmarks):
        """Analizează calitatea curentă."""
        score = 0
        details = []
        
        # Verifică prezența mâinilor (40 puncte fiecare)
        if results and results.left_hand_landmarks:
            score += 40
            details.append("Mana stanga: OK")
        else:
            details.append("Mana stanga: Lipsa")
        
        if results and results.right_hand_landmarks:
            score += 40
            details.append("Mana dreapta: OK")
        else:
            details.append("Mana dreapta: Lipsa")
        
        # Verifică pose (10 puncte)
        if results and results.pose_landmarks:
            score += 10
            details.append("Pose: OK")
        else:
            details.append("Pose: Lipsa")
        
        # Verifică față (10 puncte)
        if results and results.face_landmarks:
            score += 10
            details.append("Fata: OK")
        else:
            details.append("Fata: Lipsa")
        
        # Verifică stabilitatea (features non-zero)
        non_zero_ratio = np.count_nonzero(landmarks) / TOTAL_FEATURES
        if non_zero_ratio > 0.5:
            details.append(f"Stabilitate: {non_zero_ratio*100:.0f}%")
        
        self.history.append(score)
        avg_score = np.mean(self.history) if self.history else score
        
        return score, avg_score, details
    
    def get_feedback(self, score):
        """Returnează feedback bazat pe scor."""
        if score >= 90:
            return "Excelent! Calitate perfecta", (0, 255, 0)
        elif score >= 70:
            return "Bine - Calitate buna", (0, 255, 255)
        elif score >= 50:
            return "Acceptabil - Pozitioneaza-te mai bine", (0, 165, 255)
        else:
            return "Slab - Asigura-te ca mainile sunt vizibile", (0, 0, 255)


class SessionStats:
    """Statistici pentru sesiunea curentă."""
    
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.detection_count = 0
        self.quality_scores = []
        self.fps_history = collections.deque(maxlen=100)
    
    def update(self, has_detection, quality_score, fps):
        """Actualizează statisticile."""
        self.frame_count += 1
        if has_detection:
            self.detection_count += 1
        self.quality_scores.append(quality_score)
        self.fps_history.append(fps)
    
    def get_summary(self):
        """Returnează rezumatul statisticilor."""
        duration = time.time() - self.start_time
        avg_quality = np.mean(self.quality_scores) if self.quality_scores else 0
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0
        
        return {
            'duration': duration,
            'frames': self.frame_count,
            'detections': self.detection_count,
            'detection_rate': detection_rate,
            'avg_quality': avg_quality,
            'avg_fps': avg_fps
        }


class ThemeManager:
    """Gestionează temele vizuale."""
    
    THEMES = {
        'default': {
            'bg': (30, 30, 30),
            'text': (255, 255, 255),
            'accent': (0, 255, 255),
            'good': (0, 255, 0),
            'warning': (0, 165, 255),
            'error': (0, 0, 255),
            'landmarks_left': (0, 255, 0),
            'landmarks_right': (255, 0, 0),
            'landmarks_pose': (255, 255, 0),
            'landmarks_face': (255, 0, 255)
        },
        'dark': {
            'bg': (0, 0, 0),
            'text': (200, 200, 200),
            'accent': (100, 200, 255),
            'good': (0, 200, 0),
            'warning': (255, 165, 0),
            'error': (200, 0, 0),
            'landmarks_left': (0, 200, 0),
            'landmarks_right': (200, 0, 0),
            'landmarks_pose': (200, 200, 0),
            'landmarks_face': (200, 0, 200)
        },
        'high_contrast': {
            'bg': (0, 0, 0),
            'text': (255, 255, 255),
            'accent': (255, 255, 0),
            'good': (0, 255, 0),
            'warning': (255, 128, 0),
            'error': (255, 0, 0),
            'landmarks_left': (0, 255, 255),
            'landmarks_right': (255, 0, 255),
            'landmarks_pose': (255, 255, 0),
            'landmarks_face': (255, 128, 255)
        },
        'colorblind': {
            'bg': (30, 30, 30),
            'text': (255, 255, 255),
            'accent': (255, 200, 0),
            'good': (0, 150, 255),
            'warning': (255, 200, 0),
            'error': (255, 100, 0),
            'landmarks_left': (0, 150, 255),
            'landmarks_right': (255, 200, 0),
            'landmarks_pose': (150, 150, 255),
            'landmarks_face': (255, 150, 150)
        }
    }
    
    def __init__(self):
        self.current_theme = 'default'
        self.theme_names = list(self.THEMES.keys())
        self.theme_index = 0
    
    def get_color(self, key):
        """Returnează culoarea pentru cheia specificată."""
        return self.THEMES[self.current_theme].get(key, (255, 255, 255))
    
    def next_theme(self):
        """Trece la următoarea temă."""
        self.theme_index = (self.theme_index + 1) % len(self.theme_names)
        self.current_theme = self.theme_names[self.theme_index]
        return self.current_theme


class DatasetComparator:
    """Compară landmarks-uri cu dataset-ul."""
    
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)
        self.samples = {}
        self.load_samples()
    
    def load_samples(self):
        """Încarcă câte un sample din fiecare clasă."""
        for class_name in CLASSES:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                files = list(class_dir.glob("*.npy"))
                if files:
                    # Încarcă primul sample
                    self.samples[class_name] = np.load(files[0])
    
    def get_similarity(self, current_landmarks, class_name):
        """Calculează similaritatea cu un gest din dataset."""
        if class_name not in self.samples:
            return 0.0
        
        sample = self.samples[class_name]
        # Compară doar ultimul frame
        if len(current_landmarks) > 0:
            # Distanță euclidiană normalizată
            diff = np.linalg.norm(current_landmarks - sample)
            # Convertește la similaritate (0-100)
            similarity = max(0, 100 - diff * 10)
            return similarity
        return 0.0
    
    def get_best_match(self, current_landmarks):
        """Găsește cel mai apropiat gest din dataset."""
        best_class = None
        best_similarity = 0
        
        for class_name in self.samples:
            similarity = self.get_similarity(current_landmarks, class_name)
            if similarity > best_similarity:
                best_similarity = similarity
                best_class = class_name
        
        return best_class, best_similarity





def draw_landmarks_enhanced(frame, results, theme_manager, show_connections=True):
    """Desenează landmarks cu culori din temă."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    
    # Stiluri personalizate
    hand_left_style = mp_drawing.DrawingSpec(
        color=theme_manager.get_color('landmarks_left'), thickness=2, circle_radius=2)
    hand_right_style = mp_drawing.DrawingSpec(
        color=theme_manager.get_color('landmarks_right'), thickness=2, circle_radius=2)
    pose_style = mp_drawing.DrawingSpec(
        color=theme_manager.get_color('landmarks_pose'), thickness=2, circle_radius=2)
    face_style = mp_drawing.DrawingSpec(
        color=theme_manager.get_color('landmarks_face'), thickness=1, circle_radius=1)
    
    connection_style = mp_drawing.DrawingSpec(
        color=(100, 100, 100), thickness=1) if show_connections else None
    
    # Desenează landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            hand_left_style, connection_style)
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            hand_right_style, connection_style)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            pose_style, connection_style)
    
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            face_style, connection_style)
    
    return frame


def draw_fps_graph(frame, fps_history, theme_manager, x=20, y=200, width=200, height=100):
    """Desenează grafic FPS."""
    if len(fps_history) < 2:
        return frame
    
    overlay = frame.copy()
    
    # Background
    cv2.rectangle(overlay, (x, y), (x + width, y + height),
                 theme_manager.get_color('bg'), -1)
    cv2.rectangle(overlay, (x, y), (x + width, y + height),
                 theme_manager.get_color('text'), 1)
    
    # Titlu
    cv2.putText(overlay, "FPS", (x + 5, y + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4,
               theme_manager.get_color('text'), 1)
    
    # Grafic
    fps_list = list(fps_history)
    max_fps = max(fps_list) if fps_list else 60
    min_fps = min(fps_list) if fps_list else 0
    
    points = []
    for i, fps_val in enumerate(fps_list):
        px = x + int((i / len(fps_list)) * width)
        py = y + height - int(((fps_val - min_fps) / (max_fps - min_fps + 0.1)) * (height - 20))
        points.append((px, py))
    
    # Desenează linia
    for i in range(1, len(points)):
        color = theme_manager.get_color('good') if fps_list[i] > 25 else theme_manager.get_color('warning')
        cv2.line(overlay, points[i-1], points[i], color, 2)
    
    # Valori min/max
    cv2.putText(overlay, f"{max_fps:.0f}", (x + width - 30, y + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3,
               theme_manager.get_color('text'), 1)
    cv2.putText(overlay, f"{min_fps:.0f}", (x + width - 30, y + height - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3,
               theme_manager.get_color('text'), 1)
    
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    return frame


# ──────────────────────── Funcția principală ────────────────────────

def run_enhanced_demo(camera_idx=0):
    """
    Mod DEMO îmbunătățit cu funcționalități avansate.
    """
    print("="*70)
    print("MOD DEMO ÎMBUNĂTĂȚIT -- Vizualizare și Analiză Avansată")
    print("="*70)
    print("\nInițializare componente...")
    
    # Inițializare
    holistic = init_holistic()
    recorder = VideoRecorder()
    screenshot_mgr = ScreenshotManager()
    trail_effect = TrailEffect()
    quality_analyzer = QualityAnalyzer()
    session_stats = SessionStats()
    theme_manager = ThemeManager()
    dataset_comparator = DatasetComparator()
    
    # Cameră
    cap = None
    if isinstance(camera_idx, str):
        print(f"Conectare IP camera: {camera_idx}")
        cap = cv2.VideoCapture(camera_idx)
    else:
        camera_sources = [camera_idx, 0, 1, 2]
        for idx in camera_sources:
            print(f"Incercare camera {idx}...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"OK Camera {idx} functioneaza!")
                    camera_idx = idx
                    break
                cap.release()
                cap = None
    
    if not cap or not cap.isOpened():
        print("\nERROR: Nu s-a gasit nicio camera functionala!")
        sys.exit(1)
    
    # Configurare cameră
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    display_width, display_height = 1280, 960
    
    # Creează fereastra și o aduce în prim-plan
    window_name = 'LSR - Demo Enhanced'
    print(f"Creare fereastra: {window_name}")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(window_name, display_width, display_height)
    
    # Mută fereastra în poziție vizibilă
    cv2.moveWindow(window_name, 50, 50)
    print(f"Fereastra creata si mutata la pozitia (50, 50)")
    
    # Creează un frame inițial pentru a forța afișarea ferestrei
    initial_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    cv2.putText(initial_frame, "Initializare...", (display_width//2 - 100, display_height//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(window_name, initial_frame)
    cv2.waitKey(100)  # Așteaptă 100ms pentru a forța refresh
    print("Fereastra ar trebui sa fie vizibila acum!")
    
    # State
    buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    fps_timer = time.time()
    fps = 0
    fps_count = 0
    zoom_level = 1.0
    show_landmarks = True
    show_ui = True
    show_trails = False
    show_heatmap = False
    show_fps_graph = False
    show_dataset_compare = False
    heatmap_gen = None
    
    print("\n" + "="*70)
    print("DEMO PORNIT")
    print("Taste: Q=iesire | R=record | S=screenshot | +/-=zoom")
    print("       L=landmarks | T=trails | H=heatmap | U=UI | I=FPS graph")
    print("       D=dataset compare | F=fullscreen | X=clear")
    print("="*70 + "\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ATENTIE: Nu se pot citi cadre!")
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (display_width, display_height), 
                             interpolation=cv2.INTER_CUBIC)
            
            # Procesare MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            landmarks = extract_landmarks(results)
            buffer.append(landmarks)
            
            # Heatmap
            if show_heatmap:
                if heatmap_gen is None:
                    heatmap_gen = HeatmapGenerator(frame.shape)
                heatmap_gen.update(results, frame.shape)
                heatmap_overlay = heatmap_gen.get_overlay()
                frame = cv2.addWeighted(frame, 0.7, heatmap_overlay, 0.3, 0)
            
            # Trails
            if show_trails:
                trail_effect.update(results, frame.shape)
                trail_effect.draw(frame)
            
            # Landmarks
            if show_landmarks:
                frame = draw_landmarks_enhanced(frame, results, theme_manager)
            
            # Analiză calitate
            quality_score, avg_quality, quality_details = quality_analyzer.analyze(results, landmarks)
            feedback_text, feedback_color = quality_analyzer.get_feedback(quality_score)
            
            # Statistici
            has_detection = (results.left_hand_landmarks is not None or 
                           results.right_hand_landmarks is not None)
            session_stats.update(has_detection, quality_score, fps)
            
            # UI Overlay
            if show_ui:
                h, w = frame.shape[:2]
                overlay = frame.copy()
                
                # Header
                cv2.rectangle(overlay, (0, 0), (w, 200), theme_manager.get_color('bg'), -1)
                frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
                
                cv2.putText(frame, "MOD DEMO IMBUNATATIT",
                           (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                           theme_manager.get_color('accent'), 2)
                
                # Calitate
                cv2.putText(frame, f"Calitate: {quality_score}/100 ({avg_quality:.0f} avg)",
                           (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           theme_manager.get_color('text'), 1)
                
                cv2.putText(frame, feedback_text,
                           (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           feedback_color, 2)
                
                # Detectii
                detected_text = " | ".join(quality_details[:4])
                cv2.putText(frame, detected_text,
                           (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                           theme_manager.get_color('text'), 1)
                
                # Features
                non_zero = np.count_nonzero(landmarks)
                cv2.putText(frame, f"Features: {non_zero}/{TOTAL_FEATURES} | "
                           f"Buffer: {len(buffer)}/{SEQUENCE_LENGTH} | "
                           f"Zoom: {zoom_level*100:.0f}%",
                           (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                           theme_manager.get_color('text'), 1)
                
                # Tema
                cv2.putText(frame, f"Tema: {theme_manager.current_theme}",
                           (15, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                           theme_manager.get_color('text'), 1)
                
                # Recording indicator
                if recorder.is_recording:
                    duration = recorder.get_duration()
                    cv2.circle(frame, (w - 40, 30), 10, (0, 0, 255), -1)
                    cv2.putText(frame, f"REC {duration:.1f}s",
                               (w - 150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                               (0, 0, 255), 2)
                
                # FPS
                cv2.putText(frame, f"FPS: {fps:.0f}",
                           (w - 100, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           theme_manager.get_color('good'), 2)
                
                # Dataset compare
                if show_dataset_compare and len(buffer) > 0:
                    best_class, similarity = dataset_comparator.get_best_match(landmarks)
                    if best_class and similarity > 30:
                        cv2.putText(frame, f"Similar cu: {best_class} ({similarity:.0f}%)",
                                   (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                   theme_manager.get_color('accent'), 2)
            
            # FPS Graph
            if show_fps_graph:
                frame = draw_fps_graph(frame, session_stats.fps_history, theme_manager)
            
            # Recording
            if recorder.is_recording:
                recorder.write(frame)
            
            # Calcul FPS
            fps_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_count / (time.time() - fps_timer)
                fps_count = 0
                fps_timer = time.time()
            
            cv2.imshow(window_name, frame)
            
            # Taste
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            
            elif key == ord('r'):
                if not recorder.is_recording:
                    filename = recorder.start(display_width, display_height, 30)
                    print(f"Inregistrare pornita: {filename}")
                else:
                    filename, duration = recorder.stop()
                    print(f"Inregistrare salvata: {filename} ({duration:.1f}s)")
            
            elif key == ord('s'):
                filename = screenshot_mgr.save(frame)
                print(f"Screenshot salvat: {filename}")
            
            elif key == ord('+') or key == ord('='):
                zoom_level = min(2.0, zoom_level + 0.1)
            
            elif key == ord('-') or key == ord('_'):
                zoom_level = max(0.5, zoom_level - 0.1)
            
            elif key == ord('l'):
                show_landmarks = not show_landmarks
            
            elif key == ord('t'):
                show_trails = not show_trails
                if not show_trails:
                    trail_effect.clear()
            
            elif key == ord('h'):
                show_heatmap = not show_heatmap
                if show_heatmap and heatmap_gen is None:
                    heatmap_gen = HeatmapGenerator(frame.shape)
                elif not show_heatmap and heatmap_gen:
                    heatmap_gen = None
            
            elif key == ord('u'):
                show_ui = not show_ui
            
            elif key == ord('f'):
                prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                if prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_FULLSCREEN)
            
            elif key == ord('x'):
                trail_effect.clear()
                if heatmap_gen:
                    heatmap_gen.clear()
                print("Trails si heatmap curatate")
            
            elif key == ord('i'):
                show_fps_graph = not show_fps_graph
            
            elif key == ord('d'):
                show_dataset_compare = not show_dataset_compare
                print(f"Dataset compare: {'ON' if show_dataset_compare else 'OFF'}")
    
    finally:
        # Cleanup
        if recorder.is_recording:
            filename, duration = recorder.stop()
            print(f"\nInregistrare salvata: {filename} ({duration:.1f}s)")
        
        # Statistici finale
        stats = session_stats.get_summary()
        print("\n" + "="*70)
        print("STATISTICI SESIUNE")
        print("="*70)
        print(f"Durata: {stats['duration']:.1f}s")
        print(f"Frame-uri procesate: {stats['frames']}")
        print(f"Detectii: {stats['detections']} ({stats['detection_rate']:.1f}%)")
        print(f"Calitate medie: {stats['avg_quality']:.1f}/100")
        print(f"FPS mediu: {stats['avg_fps']:.1f}")
        print(f"Screenshots: {screenshot_mgr.count}")
        print("="*70 + "\n")
        
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LSR Demo Îmbunătățit')
    parser.add_argument('--camera', type=str, default='0',
                       help='Index cameră sau URL IP camera')
    args = parser.parse_args()
    
    camera = args.camera
    try:
        camera = int(camera)
    except ValueError:
        pass
    
    run_enhanced_demo(camera)
