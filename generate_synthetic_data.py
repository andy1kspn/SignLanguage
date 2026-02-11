"""
Generare date sintetice pentru testarea pipeline-ului de antrenare.

ATENȚIE: Aceste date sunt SINTETICE — nu reprezintă semne reale.
Scopul este de a valida că întregul pipeline funcționează end-to-end.
Pentru rezultate reale, colectează date cu: python collect_data.py

Strategia: Fiecare clasă primește un „pattern" unic de mișcare,
astfel încât modelul are ceva de învățat (nu e random pur).
"""

import os
import numpy as np
from config import (
    DATA_DIR, CLASSES, SEQUENCE_LENGTH, TOTAL_FEATURES,
    NUM_HAND_LANDMARKS, COORDS_PER_LANDMARK
)


def generate_sign_pattern(class_idx, num_classes):
    """
    Generează un pattern de mișcare unic per clasă.

    Fiecare clasă are:
      - O frecvență de oscilație diferită (simulează ritmuri diferite)
      - Un offset spațial diferit (simulează poziții diferite ale mâinii)
      - Amplitudini diferite pe axele x/y/z
    """
    rng = np.random.RandomState(class_idx * 42)

    # Parametri unici per clasă
    freq = 0.5 + class_idx * 0.3  # frecvență oscilație
    phase = class_idx * np.pi / num_classes
    base_pos = 0.3 + 0.4 * (class_idx / num_classes)  # poziție de bază
    amplitude = 0.05 + 0.03 * (class_idx % 5)

    sequence = np.zeros((SEQUENCE_LENGTH, TOTAL_FEATURES), dtype=np.float32)

    for t in range(SEQUENCE_LENGTH):
        time_ratio = t / SEQUENCE_LENGTH

        # Mâna dreaptă — mișcare principală
        for lm in range(NUM_HAND_LANDMARKS):
            idx = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK + lm * 3  # offset mâna dreaptă
            lm_offset = lm / NUM_HAND_LANDMARKS * 0.1
            sequence[t, idx] = base_pos + amplitude * np.sin(
                2 * np.pi * freq * time_ratio + phase + lm_offset)
            sequence[t, idx + 1] = 0.5 + amplitude * np.cos(
                2 * np.pi * freq * time_ratio + phase * 0.7 + lm_offset)
            sequence[t, idx + 2] = 0.01 * np.sin(
                np.pi * time_ratio + class_idx)

        # Mâna stângă — mișcare complementară (doar pentru unele clase)
        if class_idx % 3 != 0:
            for lm in range(NUM_HAND_LANDMARKS):
                idx = lm * 3
                lm_offset = lm / NUM_HAND_LANDMARKS * 0.1
                sequence[t, idx] = (1 - base_pos) + amplitude * 0.5 * np.sin(
                    2 * np.pi * freq * 0.5 * time_ratio - phase + lm_offset)
                sequence[t, idx + 1] = 0.5 + amplitude * 0.5 * np.cos(
                    2 * np.pi * freq * 0.5 * time_ratio - phase * 0.5)
                sequence[t, idx + 2] = 0.005

        # Pose — umeri și coate (relativ stabili)
        pose_start = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK * 2  # 126
        shoulders_x = [0.35, 0.65, 0.3, 0.7, 0.25, 0.75]
        shoulders_y = [0.4, 0.4, 0.55, 0.55, 0.65, 0.65]
        for p in range(6):
            idx = pose_start + p * 3
            sequence[t, idx] = shoulders_x[p] + 0.005 * np.sin(time_ratio * np.pi)
            sequence[t, idx + 1] = shoulders_y[p]
            sequence[t, idx + 2] = 0.0

        # Față — buze și sprâncene
        face_start = pose_start + 6 * 3  # 144
        for f in range(20):
            idx = face_start + f * 3
            if f < 10:  # buze
                angle = f / 10 * 2 * np.pi
                lip_open = 0.01 * (1 + np.sin(2 * np.pi * freq * 0.3 * time_ratio))
                sequence[t, idx] = 0.5 + 0.03 * np.cos(angle)
                sequence[t, idx + 1] = 0.35 + 0.02 * np.sin(angle) + lip_open
                sequence[t, idx + 2] = 0.0
            else:  # sprâncene
                brow_idx = f - 10
                side = 0 if brow_idx < 5 else 1
                local_idx = brow_idx % 5
                brow_raise = 0.005 * np.sin(2 * np.pi * freq * 0.2 * time_ratio + class_idx)
                x_base = 0.4 if side == 0 else 0.6
                sequence[t, idx] = x_base + local_idx * 0.02
                sequence[t, idx + 1] = 0.25 + brow_raise
                sequence[t, idx + 2] = 0.0

    return sequence


def generate_dataset(samples_per_class=60):
    """Generează dataset-ul complet cu variații per secvență."""
    print(f"Generare date sintetice: {samples_per_class} secvente x {len(CLASSES)} clase\n")

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for sample in range(samples_per_class):
            # Pattern de bază + variație aleatorie per secvență
            base = generate_sign_pattern(class_idx, len(CLASSES))

            # Adaugă variație: zgomot, ușoară translație, variație temporală
            noise = np.random.normal(0, 0.008, base.shape).astype(np.float32)
            translation = np.random.uniform(-0.03, 0.03, (1, TOTAL_FEATURES)).astype(np.float32)
            mask = (base != 0).astype(np.float32)
            sequence = base + noise * mask + translation * mask

            filepath = os.path.join(class_dir, f"{class_name}_{sample:04d}.npy")
            np.save(filepath, sequence)

        print(f"  {class_name:20s} : {samples_per_class} secvente generate")

    total = samples_per_class * len(CLASSES)
    print(f"\nTotal: {total} secvente generate in {DATA_DIR}")


if __name__ == "__main__":
    generate_dataset(samples_per_class=60)
