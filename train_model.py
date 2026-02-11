"""
Script de antrenare pentru modelul de recunoastere LSR.

Pipeline complet:
  1. Incarca secventele .npy din data/
  2. Normalizeaza secventele (invarianta la pozitie/scala)
  3. Aplica augmentare pentru a multiplica datasetul
  4. Optional: include date de transfer learning (WLASL)
  5. Split train/val
  6. Antreneaza cu early stopping + reduce LR on plateau
  7. Salveaza modelul antrenat

Utilizare:
  python train_model.py --arch lstm_attention --epochs 100
  python train_model.py --arch transformer --augment 10
  python train_model.py --arch lstm_attention --transfer_dir ./wlasl_data
"""

import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)

from config import (
    DATA_DIR, MODEL_DIR, MODEL_PATH, CLASSES, CLASS_TO_IDX,
    NUM_CLASSES, SEQUENCE_LENGTH, TOTAL_FEATURES,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE, VALIDATION_SPLIT
)
from utils import normalize_sequence, pad_or_truncate_sequence
from augmentation import generate_augmented_dataset, map_wlasl_to_lsr, prepare_transfer_data
from model import get_model


def load_dataset():
    """
    Incarca toate secventele .npy din DATA_DIR.

    Structura asteptata:
      data/
        buna/
          buna_0000.npy  -- shape (30, 204)
          buna_0001.npy
          ...
        multumesc/
          multumesc_0000.npy
          ...

    Returns:
        X: np.ndarray shape (N, 30, 204)
        y: np.ndarray shape (N,) -- indici de clasa
        counts: dict {clasa: numar_secvente}
    """
    X = []
    y = []
    counts = {}

    for class_name in CLASSES:
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            counts[class_name] = 0
            continue

        class_idx = CLASS_TO_IDX[class_name]
        npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        counts[class_name] = len(npy_files)

        for npy_file in npy_files:
            filepath = os.path.join(class_dir, npy_file)
            sequence = np.load(filepath)

            # Validare dimensiuni
            if sequence.ndim != 2 or sequence.shape[1] != TOTAL_FEATURES:
                print(f"  SKIP: {filepath} -- shape incorect: {sequence.shape}")
                continue

            # Pad/truncate la lungimea corecta
            sequence = pad_or_truncate_sequence(sequence, SEQUENCE_LENGTH)
            X.append(sequence)
            y.append(class_idx)

    if not X:
        return None, None, counts

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), counts


def print_dataset_stats(counts, total):
    """Afiseaza statistici despre dataset."""
    print(f"\n{'='*50}")
    print(f"  STATISTICI DATASET")
    print(f"{'='*50}")
    for cls, cnt in counts.items():
        bar = '#' * min(cnt, 40)
        print(f"  {cls:20s} : {cnt:4d} {bar}")
    print(f"{'-'*50}")
    print(f"  {'TOTAL':20s} : {total:4d} secvente")
    print(f"{'='*50}\n")


def train(args):
    """Pipeline principal de antrenare."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # -- 1. Incarca datele --
    print("Incarcare date...")
    X, y, counts = load_dataset()

    if X is None:
        print("\nERROR: Nu s-au gasit date de antrenare!")
        print(f"Directorul asteptat: {DATA_DIR}")
        print(f"Ruleaza mai intai: python collect_data.py --sign buna --samples 50")
        return

    print_dataset_stats(counts, len(X))

    # -- 2. Transfer Learning (optional) --
    if args.transfer_dir and os.path.exists(args.transfer_dir):
        print("Incarca date transfer learning (WLASL)...")
        mapping = map_wlasl_to_lsr()
        X_transfer, y_transfer = prepare_transfer_data(args.transfer_dir, mapping)
        if X_transfer is not None:
            print(f"  Date transfer: {len(X_transfer)} secvente adaugate")
            X = np.concatenate([X_transfer, X], axis=0)
            y = np.concatenate([y_transfer, y], axis=0)

    # -- 3. Normalizare --
    print("Normalizare secvente...")
    X = np.array([normalize_sequence(seq) for seq in X])

    # -- 4. Augmentare --
    if args.augment > 1:
        print(f"Augmentare date (x{args.augment})...")
        original_count = len(X)
        X, y = generate_augmented_dataset(X, y, augment_factor=args.augment)
        print(f"  {original_count} -> {len(X)} secvente")

    # -- 5. Split train/val --
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} secvente")
    print(f"Val:   {len(X_val)} secvente")

    # -- 6. Class weights (pentru dataset dezechilibrat) --
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nPonderi clase: { {CLASSES[i]: f'{w:.2f}' for i, w in class_weight_dict.items()} }")

    # -- 7. Construire model --
    print(f"\nArhitectura: {args.arch}")
    model = get_model(args.arch)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # -- 8. Callbacks --
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
    ]

    # TensorBoard (optional)
    if args.tensorboard:
        log_dir = os.path.join(MODEL_DIR, "logs")
        callbacks.append(TensorBoard(log_dir=log_dir))
        print(f"TensorBoard logs: {log_dir}")

    # -- 9. Antrenare --
    print(f"\nIncepere antrenare ({args.epochs} epoci max)...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # -- 10. Evaluare finala --
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n{'='*50}")
    print(f"  REZULTATE FINALE")
    print(f"{'='*50}")
    print(f"  Val Loss:     {val_loss:.4f}")
    print(f"  Val Accuracy: {val_acc:.4f} ({val_acc * 100:.1f}%)")
    print(f"  Model salvat: {MODEL_PATH}")
    print(f"{'='*50}")

    # Matrice de confuzie
    if args.confusion_matrix:
        print_confusion_matrix(model, X_val, y_val)


def print_confusion_matrix(model, X_val, y_val):
    """Afiseaza matricea de confuzie in terminal."""
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    print(f"\n{'='*50}")
    print("  RAPORT DE CLASIFICARE")
    print(f"{'='*50}")

    # Doar clasele prezente in validare
    present_classes = sorted(set(y_val))
    target_names = [CLASSES[i] for i in present_classes]

    print(classification_report(y_val, y_pred, labels=present_classes,
                                target_names=target_names))

    cm = confusion_matrix(y_val, y_pred, labels=present_classes)
    print("Matrice de confuzie:")
    header = "          " + " ".join(f"{name[:6]:>6s}" for name in target_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{val:6d}" for val in row)
        print(f"{target_names[i]:>10s} {row_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antrenare model LSR")
    parser.add_argument('--arch', type=str, default='lstm_attention',
                        choices=['lstm_attention', 'transformer', 'gcn_lstm'],
                        help='Arhitectura modelului')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Numar maxim de epoci (default: {EPOCHS})')
    parser.add_argument('--augment', type=int, default=5,
                        help='Factor de augmentare (default: 5)')
    parser.add_argument('--transfer_dir', type=str, default=None,
                        help='Director cu date WLASL procesate (optional)')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Activeaza logging TensorBoard')
    parser.add_argument('--confusion_matrix', action='store_true', default=True,
                        help='Afiseaza matricea de confuzie')
    args = parser.parse_args()
    train(args)
