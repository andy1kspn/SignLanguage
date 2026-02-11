"""
Script de colectare a datelor pentru Limbajul Semnelor Românesc (LSR).

Funcționare:
  1. Selectezi un semn din lista de clase (config.CLASSES)
  2. Numărătoare inversă de 3 secunde (pregătire)
  3. Se înregistrează 30 de cadre (~1 secundă) de coordonate
  4. Coordonatele se salvează ca fișier .npy (NU video brut)
     Format: (30, 204) — secvență × features

Avantajele salvării coordonatelor vs. video brut:
  - Fișiere ~50KB vs. ~5MB per secvență
  - Antrenare de ~100× mai rapidă (nu mai trebuie extracție)
  - Confidențialitate: nu se salvează imagini ale persoanei

Utilizare:
  python collect_data.py --sign buna --samples 50
  python collect_data.py --sign multumesc --samples 30
  python collect_data.py --all  # colectează pentru toate semnele
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np

from config import (
    DATA_DIR, CLASSES, CLASS_TO_IDX, SEQUENCE_LENGTH,
    SAMPLES_PER_CLASS, COLLECTION_COUNTDOWN, TOTAL_FEATURES
)
from utils import init_holistic, extract_landmarks, draw_landmarks_on_frame


def collect_samples_for_sign(sign_name, num_samples, holistic, cap):
    """
    Colectează num_samples secvențe de coordonate pentru un semn dat.

    Procesul per secvență:
      1. Afișează indicații pe ecran
      2. Numărătoare inversă de COLLECTION_COUNTDOWN secunde
      3. Capturează SEQUENCE_LENGTH cadre, extrage landmarks
      4. Salvează ca .npy

    Args:
        sign_name: numele semnului (ex: "buna")
        num_samples: câte secvențe de colectat
        holistic: instanță MediaPipe Holistic
        cap: instanță cv2.VideoCapture
    """
    sign_dir = os.path.join(DATA_DIR, sign_name)
    os.makedirs(sign_dir, exist_ok=True)

    # Numără fișierele existente pentru a continua numerotarea
    existing = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
    print(f"\n{'='*50}")
    print(f"  Semn: {sign_name.upper()}")
    print(f"  Secvențe existente: {existing}")
    print(f"  De colectat: {num_samples}")
    print(f"{'='*50}")

    for sample_idx in range(existing, existing + num_samples):
        # ── Faza 1: Așteptare (pauză între secvențe) ──
        wait_for_keypress(cap, holistic, sign_name, sample_idx, num_samples + existing)

        # ── Faza 2: Numărătoare inversă ──
        countdown(cap, holistic, sign_name)

        # ── Faza 3: Capturare secvență ──
        sequence = record_sequence(cap, holistic, sign_name, sample_idx)

        if sequence is not None:
            # Salvează secvența ca .npy
            filepath = os.path.join(sign_dir, f"{sign_name}_{sample_idx:04d}.npy")
            np.save(filepath, sequence)
            print(f"  Salvat: {filepath} — shape {sequence.shape}")
        else:
            print(f"  EROARE: Secvența {sample_idx} nu a putut fi capturată.")


def wait_for_keypress(cap, holistic, sign_name, current, total):
    """Afișează instrucțiuni și așteaptă apăsarea SPACE pentru a continua."""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)  # oglindire pentru naturalețe

        # Procesare MediaPipe (pentru feedback vizual)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        frame = draw_landmarks_on_frame(frame, results)

        # Interfață text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.putText(frame, f'Semn: {sign_name.upper()} ({current + 1}/{total})',
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, 'Pregateste semnul, apoi apasa SPACE',
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Colectare LSR', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            break
        elif key == ord('q') or key == 27:  # Q sau ESC
            print("\nColectare întreruptă de utilizator.")
            sys.exit(0)


def countdown(cap, holistic, sign_name):
    """Numărătoare inversă vizuală înainte de înregistrare."""
    start_time = time.time()
    while time.time() - start_time < COLLECTION_COUNTDOWN:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        frame = draw_landmarks_on_frame(frame, results)

        remaining = COLLECTION_COUNTDOWN - int(time.time() - start_time)
        # Cerc mare cu numărătoare
        cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2),
                   80, (0, 0, 255), 3)
        cv2.putText(frame, str(remaining),
                    (frame.shape[1] // 2 - 20, frame.shape[0] // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4)
        cv2.putText(frame, f'Pregateste: {sign_name.upper()}',
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('Colectare LSR', frame)
        cv2.waitKey(1)


def record_sequence(cap, holistic, sign_name, sample_idx):
    """
    Înregistrează o secvență de SEQUENCE_LENGTH cadre.

    Returns:
        np.ndarray: shape (SEQUENCE_LENGTH, TOTAL_FEATURES) sau None
    """
    sequence = []

    for frame_num in range(SEQUENCE_LENGTH):
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)

        # Extracție landmarks
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        keypoints = extract_landmarks(results)
        sequence.append(keypoints)

        # Feedback vizual
        frame = draw_landmarks_on_frame(frame, results)

        # Bară de progres
        progress = int((frame_num + 1) / SEQUENCE_LENGTH * frame.shape[1])
        cv2.rectangle(frame, (0, frame.shape[0] - 10),
                      (progress, frame.shape[0]), (0, 255, 0), -1)
        cv2.putText(frame, f'INREGISTRARE... {frame_num + 1}/{SEQUENCE_LENGTH}',
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Colectare LSR', frame)
        cv2.waitKey(1)

    return np.array(sequence)


def main():
    parser = argparse.ArgumentParser(
        description="Colectare date pentru Limbajul Semnelor Romanesc (LSR)"
    )
    parser.add_argument('--sign', type=str, default=None,
                        help=f'Numele semnului de colectat. Opțiuni: {CLASSES}')
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_CLASS,
                        help=f'Număr de secvențe per semn (default: {SAMPLES_PER_CLASS})')
    parser.add_argument('--all', action='store_true',
                        help='Colectează pentru toate semnele din CLASSES')
    parser.add_argument('--camera', type=int, default=0,
                        help='Index cameră (default: 0)')
    args = parser.parse_args()

    if not args.all and args.sign is None:
        print("Semnele disponibile:")
        for i, cls in enumerate(CLASSES):
            sign_dir = os.path.join(DATA_DIR, cls)
            count = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")]) \
                if os.path.exists(sign_dir) else 0
            print(f"  [{i:2d}] {cls:20s} — {count} secvențe colectate")
        print("\nUtilizare: python collect_data.py --sign buna --samples 50")
        print("      sau: python collect_data.py --all")
        return

    # Inițializare cameră
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("EROARE: Nu se poate deschide camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    holistic = init_holistic()

    try:
        if args.all:
            signs = CLASSES
        else:
            if args.sign not in CLASSES:
                print(f"EROARE: Semnul '{args.sign}' nu e în lista de clase.")
                print(f"Opțiuni valide: {CLASSES}")
                return
            signs = [args.sign]

        for sign in signs:
            collect_samples_for_sign(sign, args.samples, holistic, cap)

        print("\n✓ Colectare completă!")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()


if __name__ == "__main__":
    main()
