"""
Arhitecturi de model pentru recunoașterea Limbajului Semnelor Românesc.

Trei variante implementate, în ordine crescătoare a complexității:

1. LSTM cu Atenție (recomandat pentru start)
   - 2 straturi LSTM bidirecționale
   - Mecanism de atenție temporală (care cadre sunt cele mai importante)
   - ~500K parametri, antrenare rapidă pe CPU

2. Transformer Spatio-Temporal
   - Self-attention pe dimensiunea temporală
   - Positional encoding pentru ordine temporală
   - ~1M parametri, performanță superioară cu date suficiente

3. GCN + LSTM (Graph Convolutional Network)
   - Exploatează structura de graf a scheletului
   - Conexiunile anatomice devin muchii în graf
   - ~800K parametri, cel mai bun la capturarea relațiilor spațiale

Input comun: (batch_size, 30, 204) — secvență de 30 cadre × 204 features
Output comun: (batch_size, num_classes) — probabilități per clasă
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from config import SEQUENCE_LENGTH, TOTAL_FEATURES, NUM_CLASSES


# ═══════════════════════════════════════════════════════════════
#  1. LSTM CU ATENȚIE (Varianta recomandată)
# ═══════════════════════════════════════════════════════════════

class TemporalAttention(layers.Layer):
    """
    Mecanism de atenție temporală.

    Intuiția: Nu toate cadrele dintr-o secvență sunt la fel de importante.
    Momentul cheie al unui semn (ex: punctul de contact al mâinii)
    ar trebui să primească o pondere mai mare.

    Calcul:
      attention_weights = softmax(V · tanh(W · h + b))
      context = sum(attention_weights * h)
    """

    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.W = self.add_weight(
            name="attention_W", shape=(feature_dim, self.units),
            initializer="glorot_uniform"
        )
        self.b = self.add_weight(
            name="attention_b", shape=(self.units,),
            initializer="zeros"
        )
        self.V = self.add_weight(
            name="attention_V", shape=(self.units, 1),
            initializer="glorot_uniform"
        )

    def call(self, inputs):
        # inputs shape: (batch, seq_len, features)
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)  # (batch, seq, units)
        score = tf.matmul(score, self.V)                          # (batch, seq, 1)
        attention_weights = tf.nn.softmax(score, axis=1)          # (batch, seq, 1)
        context = tf.reduce_sum(inputs * attention_weights, axis=1)  # (batch, features)
        return context

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_lstm_attention_model(num_classes=NUM_CLASSES,
                                seq_length=SEQUENCE_LENGTH,
                                num_features=TOTAL_FEATURES):
    """
    Construiește modelul LSTM bidirecțional cu atenție.

    Arhitectură:
      Input (30, 204)
        → LayerNorm
        → BiLSTM(256) → Dropout(0.3)
        → BiLSTM(128) → Dropout(0.3)
        → TemporalAttention(128)
        → Dense(128, ReLU) → Dropout(0.4)
        → Dense(64, ReLU)
        → Dense(num_classes, Softmax)
    """
    inputs = layers.Input(shape=(seq_length, num_features), name="input_sequence")

    # Normalizare — stabilizează antrenarea
    x = layers.LayerNormalization(name="input_norm")(inputs)

    # Primul strat LSTM bidirecțional
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_1"
    )(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)

    # Al doilea strat LSTM bidirecțional
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_2"
    )(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)

    # Atenție temporală — ponderează cadrele importante
    x = TemporalAttention(128, name="temporal_attention")(x)

    # Clasificator
    x = layers.Dense(128, activation='relu', name="fc_1")(x)
    x = layers.Dropout(0.4, name="dropout_3")(x)
    x = layers.Dense(64, activation='relu', name="fc_2")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LSR_LSTM_Attention")
    return model


# ═══════════════════════════════════════════════════════════════
#  2. TRANSFORMER SPATIO-TEMPORAL
# ═══════════════════════════════════════════════════════════════

class PositionalEncoding(layers.Layer):
    """
    Positional encoding sinusoidal (Vaswani et al., 2017).

    Necesitate: Transformer-ul nu are noțiune inerentă de ordine temporală.
    PE adaugă un semnal unic pentru fiecare poziție din secvență.

      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, max_len=SEQUENCE_LENGTH, d_model=128, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape):
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) *
                          -(np.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pe[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


def build_transformer_model(num_classes=NUM_CLASSES,
                            seq_length=SEQUENCE_LENGTH,
                            num_features=TOTAL_FEATURES,
                            d_model=128,
                            num_heads=8,
                            ff_dim=256,
                            num_transformer_blocks=3):
    """
    Construiește modelul Transformer pentru clasificare de secvențe.

    Arhitectură:
      Input (30, 204)
        → Dense(d_model) — proiecție în spațiul Transformer
        → PositionalEncoding
        → N × TransformerBlock:
            → MultiHeadAttention(num_heads)
            → FeedForward(ff_dim)
            → LayerNorm + Residual
        → GlobalAveragePooling1D
        → Dense(128, ReLU) → Dropout(0.4)
        → Dense(num_classes, Softmax)
    """
    inputs = layers.Input(shape=(seq_length, num_features), name="input_sequence")

    # Proiecție liniară în spațiul Transformer
    x = layers.Dense(d_model, name="input_projection")(inputs)
    x = PositionalEncoding(seq_length, d_model, name="pos_encoding")(x)

    # Blocuri Transformer
    for i in range(num_transformer_blocks):
        # Multi-Head Self-Attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads,
            name=f"mha_{i}"
        )(x, x)
        attn_output = layers.Dropout(0.2, name=f"attn_dropout_{i}")(attn_output)
        x = layers.LayerNormalization(name=f"norm1_{i}")(x + attn_output)

        # Feed-Forward Network
        ff_output = layers.Dense(ff_dim, activation='relu', name=f"ff1_{i}")(x)
        ff_output = layers.Dense(d_model, name=f"ff2_{i}")(ff_output)
        ff_output = layers.Dropout(0.2, name=f"ff_dropout_{i}")(ff_output)
        x = layers.LayerNormalization(name=f"norm2_{i}")(x + ff_output)

    # Agregare temporală
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Clasificator
    x = layers.Dense(128, activation='relu', name="fc_1")(x)
    x = layers.Dropout(0.4, name="dropout_final")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LSR_Transformer")
    return model


# ═══════════════════════════════════════════════════════════════
#  3. GCN + LSTM (Graph Convolutional Network)
# ═══════════════════════════════════════════════════════════════

class GraphConvLayer(layers.Layer):
    """
    Strat de convoluție pe graf (GCN) adaptat pentru schelet uman.

    Intuiția: Landmark-urile nu sunt independente — mâna se conectează
    la încheietură, care se conectează la cot, etc. GCN-ul exploatează
    aceste conexiuni anatomice.

    Formula: H' = σ(Â · H · W)
      unde Â = D^(-1/2) · (A + I) · D^(-1/2) — matricea de adiacență normalizată
    """

    def __init__(self, output_dim, adjacency_matrix, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.A = adjacency_matrix  # matrice de adiacență pre-calculată

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.W = self.add_weight(
            name="gcn_W", shape=(feature_dim, self.output_dim),
            initializer="glorot_uniform"
        )
        self.b = self.add_weight(
            name="gcn_b", shape=(self.output_dim,),
            initializer="zeros"
        )

        # Normalizare matrice adiacență: Â = D^(-1/2)(A+I)D^(-1/2)
        A_hat = self.A + np.eye(self.A.shape[0])
        D = np.diag(np.sum(A_hat, axis=1) ** (-0.5))
        D = np.nan_to_num(D, nan=0.0)
        A_norm = D @ A_hat @ D
        self.A_norm = tf.constant(A_norm, dtype=tf.float32)

    def call(self, inputs):
        # inputs: (batch, num_nodes, features)
        support = tf.matmul(inputs, self.W) + self.b       # (batch, nodes, output_dim)
        output = tf.matmul(self.A_norm, support)            # (batch, nodes, output_dim)
        return tf.nn.relu(output)

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        return config


def build_skeleton_adjacency():
    """
    Construiește matricea de adiacență pentru scheletul de landmarks.

    Structura grafului (68 noduri):
      - Noduri 0-20: mâna stângă (21 landmarks)
      - Noduri 21-41: mâna dreaptă (21 landmarks)
      - Noduri 42-47: pose — umeri, coate, încheieturi (6 landmarks)
      - Noduri 48-67: față — buze + sprâncene (20 landmarks)

    Conexiunile sunt bazate pe anatomia umană:
      - Degete: vârf → articulație → bază → centru palmă
      - Braț: mână → încheietură → cot → umăr
      - Față: contur buze, arc sprâncene
    """
    num_nodes = 21 + 21 + 6 + 20  # 68
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Conexiuni mână (identice pentru stânga și dreapta)
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),         # deget mare
        (0, 5), (5, 6), (6, 7), (7, 8),         # index
        (0, 9), (9, 10), (10, 11), (11, 12),    # mijlociu
        (0, 13), (13, 14), (14, 15), (15, 16),  # inelar
        (0, 17), (17, 18), (18, 19), (19, 20),  # mic
        (5, 9), (9, 13), (13, 17),               # conexiuni palmă
    ]

    # Mâna stângă (offset 0)
    for (i, j) in hand_connections:
        A[i, j] = A[j, i] = 1

    # Mâna dreaptă (offset 21)
    for (i, j) in hand_connections:
        A[i + 21, j + 21] = A[j + 21, i + 21] = 1

    # Pose: umăr_stâng(42) — umăr_drept(43) — cot_stâng(44) — etc.
    pose_connections = [
        (42, 43),  # umăr stâng — umăr drept
        (42, 44),  # umăr stâng — cot stâng
        (43, 45),  # umăr drept — cot drept
        (44, 46),  # cot stâng — încheietură stângă
        (45, 47),  # cot drept — încheietură dreaptă
    ]
    for (i, j) in pose_connections:
        A[i, j] = A[j, i] = 1

    # Conexiune mâini → încheieturi
    A[0, 46] = A[46, 0] = 1     # centru palmă stângă → încheietură stângă
    A[21, 47] = A[47, 21] = 1   # centru palmă dreaptă → încheietură dreaptă

    # Față: conexiuni buze (contur)
    face_offset = 48
    lip_ring = list(range(10))  # primele 10 landmarks sunt buze
    for i in range(len(lip_ring)):
        j = (i + 1) % len(lip_ring)
        A[face_offset + lip_ring[i], face_offset + lip_ring[j]] = 1
        A[face_offset + lip_ring[j], face_offset + lip_ring[i]] = 1

    # Sprâncene (arc)
    for brow_start in [10, 15]:  # stânga, dreapta (5 puncte fiecare)
        for k in range(4):
            A[face_offset + brow_start + k,
              face_offset + brow_start + k + 1] = 1
            A[face_offset + brow_start + k + 1,
              face_offset + brow_start + k] = 1

    return A


def build_gcn_lstm_model(num_classes=NUM_CLASSES,
                         seq_length=SEQUENCE_LENGTH):
    """
    Construiește modelul GCN + LSTM.

    Arhitectură:
      Input (30, 204) → Reshape (30, 68, 3) — 68 noduri × 3 coordonate
        → Per cadru:
            → GCN(64) → GCN(32) — extracție features spațiale din graf
            → Flatten → vector per cadru
        → LSTM(128) pe secvența de vectori
        → Dense(64, ReLU) → Dropout(0.4)
        → Dense(num_classes, Softmax)
    """
    num_nodes = 68  # 21 + 21 + 6 + 20
    A = build_skeleton_adjacency()

    inputs = layers.Input(shape=(seq_length, num_nodes * 3), name="input_sequence")

    # Reshape: (batch, seq, 204) → (batch, seq, 68, 3)
    x = layers.Reshape((seq_length, num_nodes, 3), name="reshape_to_graph")(inputs)

    # Aplicăm GCN per cadru folosind TimeDistributed
    gcn1 = GraphConvLayer(64, A, name="gcn_1")
    gcn2 = GraphConvLayer(32, A, name="gcn_2")

    # Process fiecare cadru prin GCN
    x = layers.TimeDistributed(gcn1, name="td_gcn_1")(x)
    x = layers.TimeDistributed(gcn2, name="td_gcn_2")(x)

    # Flatten nodurile: (batch, seq, 68, 32) → (batch, seq, 68*32)
    x = layers.TimeDistributed(layers.Flatten(), name="td_flatten")(x)

    # Dependențe temporale cu LSTM
    x = layers.LSTM(128, return_sequences=False, name="temporal_lstm")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    x = layers.Dense(64, activation='relu', name="fc_1")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LSR_GCN_LSTM")
    return model


# ═══════════════════════════════════════════════════════════════
#  FACTORY — Selectare model
# ═══════════════════════════════════════════════════════════════

def get_model(architecture='lstm_attention'):
    """
    Factory function pentru crearea modelului.

    Args:
        architecture: 'lstm_attention' | 'transformer' | 'gcn_lstm'
    Returns:
        keras.Model compilat
    """
    if architecture == 'lstm_attention':
        model = build_lstm_attention_model()
    elif architecture == 'transformer':
        model = build_transformer_model()
    elif architecture == 'gcn_lstm':
        model = build_gcn_lstm_model()
    else:
        raise ValueError(f"Arhitectură necunoscută: {architecture}")

    return model


if __name__ == "__main__":
    # Testare rapidă a tuturor arhitecturilor
    for arch in ['lstm_attention', 'transformer', 'gcn_lstm']:
        print(f"\n{'='*60}")
        print(f"  {arch.upper()}")
        print(f"{'='*60}")
        model = get_model(arch)
        model.summary()

        # Test cu date sintetice
        dummy_input = np.random.randn(2, SEQUENCE_LENGTH, TOTAL_FEATURES).astype(np.float32)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")  # (2, NUM_CLASSES)
