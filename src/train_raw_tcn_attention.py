"""
train_raw_tcn_attention.py  (v6 — Sıfırdan Basit ve Kararlı Varsayılan Sürüm)
-------------------------------------------------------------------------
Karmaşık tüm yapılar (curriculum learning, condition embedding vb.) kaldırıldı.
Ham titreşim verileri (.npz) üzerinde çalışan ilk varsayılan SA-TCN modeline geri dönüldü.

Çalıştırma:
    python src/train_raw_tcn_attention.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model

# ── Dizin Yapılandırması ───────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent          # .../femto_rul/src
_PROJECT    = _HERE.parent                             # .../femto_rul
MODELS_DIR  = _PROJECT / "experiments" / "models"
RESULTS_DIR = _PROJECT / "experiments" / "results"
NPY_DIR     = _PROJECT / "experiments" / "numpy_data"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "satcn_v6_default.keras"

# ── Sabitler ──────────────────────────────────────────────────────────────────
BATCH_SIZE          = 64
EPOCHS              = 80
EARLY_STOP_PATIENCE = 15
LR_PATIENCE         = 7
RUL_CAP             = 125.0  # Dakika

# Hafif ve dengeli ceza
PENALTY = 2.0   
DELTA   = 0.05  

ACTUAL_RUL_SECONDS = {
    "Bearing1_3": 5730, "Bearing1_4": 339,  "Bearing1_5": 1610,
    "Bearing1_6": 1460, "Bearing1_7": 7570,
    "Bearing2_3": 7530, "Bearing2_4": 1390, "Bearing2_5": 3090,
    "Bearing2_6": 1290, "Bearing2_7": 580,
    "Bearing3_3": 820,
}

# ── 1. Veri Yükleme ───────────────────────────────────────────────────────────

def baseline_standardize(X: np.ndarray, num_baseline: int = 50) -> np.ndarray:
    """
    Her rulmanın ilk 50 penceresindeki std'ye göre kanal bazında normalize eder.
    """
    std_h = max(np.std(X[:min(num_baseline, len(X)), :, 0]), 1e-5)
    std_v = max(np.std(X[:min(num_baseline, len(X)), :, 1]), 1e-5)
    X_norm = X.copy()
    X_norm[:, :, 0] /= std_h
    X_norm[:, :, 1] /= std_v
    return X_norm


def load_bearing_data(bname: str, is_test: bool = False,
                      monotone_labels: bool = False) -> tuple:
    """
    NPZ dosyasını yükler ve baseline normalize eder.
    Döner: (X_norm, y)
    """
    prefix = "test" if is_test else "train"
    path   = NPY_DIR / f"{prefix}_{bname}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Dosya bulunamadı: {path}\n"
            f"Lütfen {NPY_DIR} klasöründe verinin olduğunu kontrol edin."
        )

    data = np.load(path)
    X    = data["X"].astype(np.float32)

    if is_test:
        actual_rul_s = ACTUAL_RUL_SECONDS[bname]
        y = np.clip(
            ((len(X) - 1 - np.arange(len(X))) * 10.0 + actual_rul_s) / (RUL_CAP * 60.0),
            0.0, 1.0
        ).astype(np.float32)
    else:
        y = data["y"].astype(np.float32)

    if monotone_labels:
        y = np.minimum.accumulate(y)

    X_norm = baseline_standardize(X)
    return X_norm, y


def load_all_data_per_bearing(bearings: list, monotone_labels: bool = False):
    """
    Her bearing ayrı normalize edilir ve birleştirilir.
    """
    X_list, y_list = [], []
    for b in bearings:
        X, y = load_bearing_data(b, is_test=False, monotone_labels=monotone_labels)
        X_list.append(X)
        y_list.append(y)
        print(f"    {b}: {X.shape} | y ∈ [{y.min():.3f}, {y.max():.3f}]")
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


# ── 2. Model Mimarisi ─────────────────────────────────────────────────────────

def tcn_block(x, filters: int, kernel_size: int, dilation_rate: int,
              dropout_rate: float = 0.2):
    shortcut = layers.Conv1D(filters, 1, padding="same")(x)

    c = layers.Conv1D(
        filters, kernel_size, padding="same", dilation_rate=dilation_rate,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    c = layers.BatchNormalization()(c)
    c = layers.Activation("relu")(c)
    c = layers.SpatialDropout1D(dropout_rate)(c)

    c = layers.Conv1D(
        filters, kernel_size, padding="same", dilation_rate=dilation_rate,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(c)
    c = layers.BatchNormalization()(c)
    c = layers.Activation("relu")(c)
    c = layers.SpatialDropout1D(dropout_rate)(c)

    out = layers.add([shortcut, c])
    return layers.Activation("relu")(out)


def asymmetric_huber_loss(delta: float = DELTA, over_pred_penalty: float = PENALTY):
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        err    = y_pred - y_true
        abs_e  = tf.abs(err)
        huber  = tf.where(
            abs_e <= delta,
            0.5 * tf.square(err),
            delta * (abs_e - 0.5 * delta),
        )
        weight = tf.where(
            err > 0.0,
            tf.fill(tf.shape(err), float(over_pred_penalty)),
            tf.ones_like(err),
        )
        return tf.reduce_mean(weight * huber)
    _loss.__name__ = "asym_huber"
    return _loss


def build_model(input_shape: tuple = (2560, 2), lr: float = 1e-3) -> Model:
    inp_signal = layers.Input(shape=input_shape, name="raw_signal")

    # ── Sinyal yolu: Stride Conv (1/2 küçültme) + 4×TCN + Self-Attention ──
    x = layers.Conv1D(
        32, kernel_size=8, strides=2, padding="same", activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="stride_conv"
    )(inp_signal)
    x = layers.BatchNormalization()(x)

    x = tcn_block(x, filters=32, kernel_size=3, dilation_rate=1)
    x = tcn_block(x, filters=32, kernel_size=3, dilation_rate=2)
    x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=4)
    x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=8)

    attn = layers.MultiHeadAttention(
        num_heads=4, key_dim=16, dropout=0.1, name="self_attention"
    )(query=x, value=x, key=x)
    x = layers.add([x, attn])
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)  # (batch, 64)

    z = layers.Dense(64, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(32, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(z)
    z = layers.Dropout(0.1)(z)
    out = layers.Dense(1, activation="sigmoid", name="hi_output")(z)

    model = Model(inputs=inp_signal, outputs=out, name="SA_TCN_v6_Default")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=asymmetric_huber_loss(delta=DELTA, over_pred_penalty=PENALTY),
        metrics=["mae"],
    )
    return model


# ── 3. PHM Metrikleri ─────────────────────────────────────────────────────────

def phm_percent_error(act: float, pred: float) -> float:
    return 100.0 * (act - pred) / (act + 1e-10)


def phm_accuracy(percent_error: float) -> float:
    ln05 = np.log(0.5)
    er   = percent_error
    if er <= 0:
        return float(np.exp(-ln05 * (er / 5.0)))
    else:
        return float(np.exp(ln05 * (er / 20.0)))


def phm_penalty_score(actual_rul_s: float, predicted_rul_s: float) -> float:
    di = predicted_rul_s - actual_rul_s
    if di >= 0:
        return float(np.exp(di / 650.0) - 1.0)
    else:
        return float(np.exp(-di / 1000.0) - 1.0)


# ── 4. Ana Akış ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  FEMTO PHM — SA-TCN v6 (Sıfırdan Varsayılan Sürüm)")
    print("=" * 70)

    # 4.1 Veri yükle
    print("\n[1/4] Veriler yükleniyor...")
    train_bearings = ["Bearing1_1", "Bearing2_1", "Bearing3_1"]
    val_bearings   = ["Bearing1_2", "Bearing2_2", "Bearing3_2"]

    X_train, y_train = load_all_data_per_bearing(train_bearings, monotone_labels=True)
    X_val, y_val     = load_all_data_per_bearing(val_bearings, monotone_labels=True)

    print(f"\n  Eğitim   : X={X_train.shape} | y={y_train.shape}")
    print(f"  Doğrulama: X={X_val.shape}   | y={y_val.shape}")

    # 4.2 Model oluştur & eğit
    print(f"\n[2/4] Model eğitiliyor (epoch={EPOCHS}, penalty={PENALTY}, delta={DELTA})...")
    model = build_model(lr=1e-3)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH), monitor="val_loss",
            save_best_only=True, mode="min", verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=LR_PATIENCE, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    # Eğitim eğrisi
    fig_hist, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["mae"],     label="Train MAE")
    axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_title("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig_hist.tight_layout()
    fig_hist.savefig(RESULTS_DIR / "satcn_v6_training_history.png", dpi=150)
    plt.close(fig_hist)

    # 4.3 Test
    print("\n[3/4] Test rulmanları değerlendiriliyor...")
    test_bearings = [
        "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
        "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
        "Bearing3_3",
    ]

    if MODEL_PATH.exists():
        model.load_weights(str(MODEL_PATH))
        print(f"  En iyi model yüklendi: {MODEL_PATH.name}")

    results = []
    n_cols  = 3
    n_rows  = int(np.ceil(len(test_bearings) / n_cols))
    fig     = plt.figure(figsize=(18, n_rows * 5))
    fig.suptitle(
        "SA-TCN v6 (Varsayılan Sürüm) — Test Seti HI Tahminleri",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.3)

    for b_idx, bname in enumerate(test_bearings):
        print(f"  {bname} ... ", end="", flush=True)

        X_test, y_test = load_bearing_data(bname, is_test=True)

        pred_hi      = model.predict(X_test, batch_size=64, verbose=0).flatten()
        pred_hi_mono = np.minimum.accumulate(pred_hi)
        true_hi      = y_test

        actual_rul_s     = ACTUAL_RUL_SECONDS[bname]
        pred_rul_s_mono  = float(pred_hi_mono[-1]) * (RUL_CAP * 60.0)
        rul_err_min_mono = (pred_rul_s_mono - actual_rul_s) / 60.0

        err_pct_mono  = phm_percent_error(actual_rul_s / 60.0, pred_rul_s_mono / 60.0)
        acc_mono      = phm_accuracy(err_pct_mono)
        penalty_mono  = phm_penalty_score(actual_rul_s, pred_rul_s_mono)

        status = "✅" if penalty_mono < 2 else ("🟠" if penalty_mono < 100 else "🔴")
        print(f"Ceza={penalty_mono:.2f} {status} | RUL Hata={rul_err_min_mono:+.1f} dk")

        results.append({
            "bearing"          : bname,
            "actual_rul_min"   : round(actual_rul_s / 60.0, 1),
            "pred_rul_min_mono": round(pred_rul_s_mono / 60.0, 1),
            "rul_err_min_mono" : round(rul_err_min_mono, 1),
            "accuracy_mono"    : round(acc_mono, 4),
            "penalty_mono"     : round(penalty_mono, 2),
        })

        # Grafik
        ax = fig.add_subplot(gs[b_idx // n_cols, b_idx % n_cols])
        t  = np.arange(len(true_hi)) * 10.0 / 60.0

        ax.fill_between(t, true_hi, alpha=0.10, color="#42A5F5")
        ax.plot(t, true_hi,      lw=1.5, color="#42A5F5", label="Gerçek HI")
        ax.plot(t, pred_hi,      lw=1.2, color="#EF5350", alpha=0.5,
                linestyle=":", label="Ham Tahmin")
        ax.plot(t, pred_hi_mono, lw=1.5, color="#E53935", label="Tahmin (Mono)")

        ax.set_title(
            f"{bname}\nAcc={acc_mono:.3f} | RUL err={rul_err_min_mono:+.1f}dk",
            fontsize=9, fontweight="bold"
        )
        ax.set_xlabel("Zaman (dk)", fontsize=8)
        ax.set_ylabel("HI [0,1]",   fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    plt.savefig(RESULTS_DIR / "satcn_v6_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4.4 Sonuçlar
    print("\n[4/4] Sonuçlar kaydediliyor...")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "satcn_v6_results.csv", index=False)

    mean_acc     = df["accuracy_mono"].mean()
    total_pen    = df["penalty_mono"].sum()
    phm_score    = mean_acc / (1.0 + total_pen)

    print(f"\n{'='*70}")
    print("  SA-TCN v6 — TEST SETİ SONUÇLARI")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"{'='*70}")
    print(f"  Ortalama Doğruluk (Accuracy)  : {mean_acc:.4f}")
    print(f"  Toplam Ceza Puanı (Penalty)   : {total_pen:.2f}")
    print(f"  PHM Gösterge Skoru            : {phm_score:.4f}")
    print(f"{'='*70}")
    print(f"\n  Eğitim grafiği : {RESULTS_DIR / 'satcn_v6_training_history.png'}")
    print(f"  Tahmin grafiği : {RESULTS_DIR / 'satcn_v6_predictions.png'}")
    print(f"  Sonuç CSV      : {RESULTS_DIR / 'satcn_v6_results.csv'}")
    print(f"  Model          : {MODEL_PATH}")


if __name__ == "__main__":
    main()
