"""
dl_models.py
------------
Track B derin öğrenme modelleri (Keras). Hepsi HI ∈ [0,1] (sigmoid) tahmin eder.
  - build_gru(F)      : öznitelik dizisi (W, F) → GRU
  - build_tcn(F)      : öznitelik dizisi (W, F) → dilated TCN
  - build_raw_tcn()   : ham sinyal (2560, 2) → strided + dilated TCN

Hepsi küçük + güçlü regularize (6 eğri → overfit riski). GaussianNoise = augmentasyon.
TensorFlow gerekir (Colab/GPU).
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

from config import SEQ_LEN, RAW_WINDOW


def _compile(m, lr=1e-3):
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return m


def build_gru(F: int, W: int = SEQ_LEN) -> Model:
    inp = layers.Input((W, F))
    x = layers.GaussianNoise(0.1)(inp)
    x = layers.GRU(48, return_sequences=True, dropout=0.2,
                   kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.GRU(24, dropout=0.2, kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return _compile(Model(inp, out, name="GRU"))


def _tcn_block(x, f, d, kernel=3, drop=0.15, causal=True, l2=1e-3):
    pad = "causal" if causal else "same"
    sc = layers.Conv1D(f, 1, padding="same")(x)
    c = layers.Conv1D(f, kernel, padding=pad, dilation_rate=d,
                      kernel_regularizer=regularizers.l2(l2))(x)
    c = layers.BatchNormalization()(c); c = layers.Activation("relu")(c)
    c = layers.SpatialDropout1D(drop)(c)
    c = layers.Conv1D(f, kernel, padding=pad, dilation_rate=d,
                      kernel_regularizer=regularizers.l2(l2))(c)
    c = layers.BatchNormalization()(c); c = layers.Activation("relu")(c)
    c = layers.SpatialDropout1D(drop)(c)
    return layers.Activation("relu")(layers.add([sc, c]))


def build_tcn(F: int, W: int = SEQ_LEN) -> Model:
    inp = layers.Input((W, F))
    x = layers.GaussianNoise(0.1)(inp)
    # padding='same' — doğrulanmış Colab TCN ile birebir (causal değil)
    x = _tcn_block(x, 32, 1, causal=False); x = _tcn_block(x, 32, 2, causal=False)
    x = _tcn_block(x, 32, 4, causal=False)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return _compile(Model(inp, out, name="TCN"))


def build_raw_tcn() -> Model:
    """Ham sinyal (2560,2): strided conv ile downsample (2560→80) + dilated TCN."""
    inp = layers.Input(RAW_WINDOW)
    x = layers.Conv1D(16, 16, strides=4, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 8, strides=4, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    # pencere-içi: causal gerekmez (same)
    x = _tcn_block(x, 32, 1, causal=False, drop=0.1, l2=1e-4)
    x = _tcn_block(x, 32, 2, causal=False, drop=0.1, l2=1e-4)
    x = _tcn_block(x, 32, 4, causal=False, drop=0.1, l2=1e-4)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return _compile(Model(inp, out, name="raw_TCN"))


BUILDERS = {"GRU": build_gru, "TCN": build_tcn}   # öznitelik-dizisi modelleri
