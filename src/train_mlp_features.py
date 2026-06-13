"""
src/train_mlp_features.py
-------------------------
Manuel öznitelikler üzerinde çalışan Çok Katmanlı Algılayıcı (MLP) Derin Öğrenme modelini eğitir.
XGBoost ile birebir aynı veri ön işleme (RobustScaler, MI Selection, Degradation Filter) adımlarını kullanır.

Çalıştırma:
    python src/train_mlp_features.py
"""

import sys
import gc
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Proje kök dizinini ekle
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
sys.path.insert(0, str(_HERE))

from data_utils import train_val_split
from evaluation import evaluate_val, evaluate_test_bearings
from config import FEATURES_TRAIN, FEATURES_TEST

MODELS_DIR = _PROJECT / "experiments" / "models"
RESULTS_DIR = _PROJECT / "experiments" / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "mlp_features_model.keras"

# ── 1. RUL Standartlaştırma ──────────────────────────────────────────────────
def apply_standard_rul(df: pd.DataFrame, max_rul: float = 125.0) -> pd.DataFrame:
    total_life = df.groupby('bearing').apply(lambda g: (g['time_s'] + g['rul_s']).max(), include_groups=False)
    total_map = df['bearing'].map(total_life)
    df['rul_min'] = np.clip((total_map - df['time_s']) / 60.0, 0, max_rul)
    return df

# ── 2. MLP Model Mimarisi ────────────────────────────────────────────────────
def build_mlp(input_dim: int, lr: float = 1e-3) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # RUL tahmini için çıkış katmanı (doğrusal çıkış)
    outputs = tf.keras.layers.Dense(1, name="rul_output")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MLP_Feature_Regressor")
    
    # XGBoost ile uyumlu standart Huber loss kullanalım
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"]
    )
    return model

# ── 3. Test Tahmin Fonksiyonu ────────────────────────────────────────────────
def get_test_predictions(model, prep, test_df: pd.DataFrame) -> dict:
    predictions = {}
    for bname in test_df["bearing"].unique():
        bdf = test_df[test_df["bearing"] == bname].sort_values("time_s")
        X_b, _ = prep.transform(bdf)
        pred = model.predict(X_b[-1:], verbose=0)
        predictions[bname] = float(pred[0, 0])
    return predictions

# ── 4. Ana Akış ────────────────==============================================
def main():
    print("=" * 70)
    print("  FEMTO PHM — Manuel Öznitelikli MLP Eğitimi")
    print("=" * 70)
    
    # Veriyi Yükle
    print("\n[1/4] Veriler yükleniyor...")
    df_train_full = pd.read_csv(FEATURES_TRAIN)
    df_test = pd.read_csv(FEATURES_TEST)
    
    df_train_full = apply_standard_rul(df_train_full)
    df_test = apply_standard_rul(df_test)
    
    train_df, val_df = train_val_split(df_train_full)
    
    # Degradasyon Filtresi
    RUL_CAP = 125.0
    DEGRAD_THRESH = RUL_CAP * 0.98  # 122.5 dk
    train_df = train_df[train_df['rul_min'] < DEGRAD_THRESH].reset_index(drop=True)
    val_df = val_df[val_df['rul_min'] < DEGRAD_THRESH].reset_index(drop=True)
    
    print(f"  Eğitim Verisi: {len(train_df)} satır | Val Verisi: {len(val_df)} satır")
    
    # Preprocessor Yükle
    prep_path = MODELS_DIR / "preprocessor.pkl"
    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessor bulunamadı: {prep_path}. Önce train_track_a.py'yi çalıştırmalısınız.")
        
    print(f"\n[2/4] Preprocessor yükleniyor: {prep_path.name}...")
    prep = joblib.load(prep_path)
    
    # Transform
    X_train, y_train = prep.transform(train_df)
    X_val, y_val = prep.transform(val_df)
    
    input_dim = X_train.shape[1]
    print(f"  Girdi öznitelik sayısı: {input_dim}")
    
    # MLP Modeli Oluştur ve Eğit
    print("\n[3/4] MLP modeli kuruluyor ve eğitiliyor...")
    model = build_mlp(input_dim=input_dim, lr=1e-3)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH), monitor="val_loss",
            save_best_only=True, mode="min", verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        )
    ]
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=64,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )
    
    # Validation Değerlendirmesi
    y_val_pred = model.predict(X_val, verbose=0).flatten()
    metrics = evaluate_val(y_val, y_val_pred, model_name="MLP (Manuel Öznitelik)", verbose=True)
    
    # Test Değerlendirmesi
    print("\n[4/4] Test bearing tahminleri yapılıyor...")
    test_preds = get_test_predictions(model, prep, df_test)
    phm_result = evaluate_test_bearings(test_preds, unit="min")
    
    # Sonuçları Kaydet
    results_path = RESULTS_DIR / "mlp_features_results.csv"
    pd.DataFrame([{
        "model": "MLP (Manuel)",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "phm_score": phm_result["score"]
    }]).to_csv(results_path, index=False)
    
    print(f"\nSonuçlar kaydedildi: {results_path}")
    print("MLP Eğitimi tamamlandı!")

if __name__ == "__main__":
    main()
