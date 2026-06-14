"""
train_track_a.py
----------------
Track A: Random Forest ve XGBoost modellerini eğitir ve değerlendirir.
(SVR, hedef değişkenin devasa ölçeği nedeniyle iptal edildi.)

Çalıştırma:
    cd femto_rul/src
    python train_track_a.py

Çıktılar (experiments/ altında):
    models/preprocessor.pkl        → kaydedilmiş preprocessing pipeline
    models/Random_Forest.pkl       → eğitilmiş RF modeli
    models/XGBoost.pkl             → eğitilmiş XGBoost modeli
    results/track_a_results.csv    → tüm modellerin metrik tablosu
    results/track_a_comparison.png → karşılaştırma grafiği
    results/track_a_phm_score.png  → PHM score grafiği
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_utils    import train_val_split, get_feature_cols
from preprocessing import Preprocessor
from models        import RandomForestModel, XGBoostModel
from evaluation    import (
    evaluate_val, evaluate_test_bearings,
    plot_predictions, plot_model_comparison,
    plot_phm_scoring_function,
)
from config import FEATURES_TRAIN, FEATURES_TEST

_HERE       = Path(__file__).resolve().parent          # .../femto_rul/src
_PROJECT    = _HERE.parent                             # .../femto_rul
MODELS_DIR  = _PROJECT / "experiments" / "models"
RESULTS_DIR = _PROJECT / "experiments" / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_test_prediction(model, prep, test_df: pd.DataFrame) -> dict:
    """
    Her test bearing'i için son penceredeki RUL tahminini döndürür.
    Test bearing'lerin verisi kesilmiş — son pencere mevcut anı temsil eder.
    Oradan tahmin edilen RUL = kalan ömür tahmini.
    """
    predictions = {}
    for bname in test_df["bearing"].unique():
        bdf   = test_df[test_df["bearing"] == bname].sort_values("time_s")
        X_b, _ = prep.transform(bdf)
        # Son pencerenin tahmini
        pred = model.predict(X_b[-1:])
        predictions[bname] = float(pred[0])
    return predictions


def apply_standard_rul(df: pd.DataFrame, max_rul: float = 125.0) -> pd.DataFrame:
    """Standardizes piecewise linear RUL to a global cap, fixing massive MSE issues on healthy phase."""
    total_life = df.groupby('bearing').apply(lambda g: (g['time_s'] + g['rul_s']).max(), include_groups=False)
    total_map = df['bearing'].map(total_life)
    df['rul_min'] = np.clip((total_map - df['time_s']) / 60.0, 0, max_rul)
    return df

def main():
    print("\n" + "=" * 60)
    print("  FEMTO PHM — Track A Eğitimi")
    print("=" * 60)

    # ── 1. Veri Yükle ────────────────────────────────────────────────────────
    print("\n[1/4] Veri yükleniyor...")
    df_train_full = pd.read_csv(FEATURES_TRAIN)
    df_test       = pd.read_csv(FEATURES_TEST)
    
    # Standart RUL hedeflerini uygula (hepsi için max 125 dk sabit)
    df_train_full = apply_standard_rul(df_train_full)
    df_test       = apply_standard_rul(df_test)
    
    train_df, val_df = train_val_split(df_train_full)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(df_test)}")

    # ── Degradasyon Filtresi ──────────────────────────────────────────────────
    # Saglikli faz (RUL = 125 dk tavaninda) feature uzayinda sifir civarinda
    # yuzer; model bu duzlugu ogrenmeye calisirken kaybolur.
    # Sadece bozunum baslangici sonrasindaki pencereleri kullan.
    RUL_CAP = 125.0
    DEGRAD_THRESH = RUL_CAP * 0.98   # 122.5 dk altindaki pencereler = bozunum
    n_train_all = len(train_df)
    n_val_all   = len(val_df)
    train_df = train_df[train_df['rul_min'] < DEGRAD_THRESH].reset_index(drop=True)
    val_df   = val_df[val_df['rul_min'] < DEGRAD_THRESH].reset_index(drop=True)
    print(f"  [Filtre] Train: {n_train_all} -> {len(train_df)} | Val: {n_val_all} -> {len(val_df)}")
    print(f"           (Saglikli faz pencereleri ci karildi)")

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    print("\n[2/4] Preprocessing...")
    prep = Preprocessor(n_features=25, use_pca=False)
    X_train, y_train = prep.fit_transform(train_df)
    X_val,   y_val   = prep.transform(val_df)
    prep.save(MODELS_DIR / "preprocessor.pkl")

    fi_df = prep.get_feature_importance_df()
    fi_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    print(f"\n  Seçilen öznitelikler: {prep.selected_features}")

    # ── 3. Model Eğitimi ─────────────────────────────────────────────────────
    print("\n[3/4] Modeller eğitiliyor...")

    models = [
        RandomForestModel(n_estimators=300, min_samples_leaf=5),
        # Early stopping kaldirildi: filtreli-train vs filtreli/filtrelenmemis-val
        # hedef dagilimi uyumsuzlugu yuzunden XGBoost iter=3-35'de duruyordu.
        # Sabit 400 tur + dusuk lr cok daha kararli sonuc verir.
        XGBoostModel(n_estimators=400, learning_rate=0.02, max_depth=6),
    ]

    all_val_results  = []
    all_test_results = []

    for model in models:
        print(f"\n  -> {model.name}")

        # Egit — XGBoost icin early stopping YOK (sabit tur)
        model.fit(X_train, y_train)

        # Validation değerlendirme
        y_val_pred = model.predict(X_val)
        metrics    = evaluate_val(y_val, y_val_pred,
                                  model_name=model.name, verbose=True)
        all_val_results.append(metrics)

        # Val tahmin grafiği
        safe_name = model.name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_predictions(
            y_val, y_val_pred,
            model_name=model.name,
            save_path=RESULTS_DIR / f"val_pred_{safe_name}.png"
        )

        # Test bearing tahmini (son pencere)
        test_preds = get_test_prediction(model, prep, df_test)
        phm_result = evaluate_test_bearings(test_preds, unit="min")
        all_test_results.append({
            "model":     model.name,
            "phm_score": phm_result["score"],
        })

        # Modeli kaydet
        model.save(MODELS_DIR / f"{safe_name}.pkl")

    # ── 4. Karşılaştırma ─────────────────────────────────────────────────────
    print("\n[4/4] Sonuçlar kaydediliyor...")

    val_df_res  = pd.DataFrame(all_val_results)
    test_df_res = pd.DataFrame(all_test_results)
    results_df  = val_df_res.merge(test_df_res, on="model")
    results_df.to_csv(RESULTS_DIR / "track_a_results.csv", index=False)

    print("\n" + "=" * 60)
    print("  TRACK A — SONUÇLAR")
    print("=" * 60)
    print(results_df.to_string(index=False))

    plot_model_comparison(
        results_df,
        save_path=RESULTS_DIR / "track_a_comparison.png"
    )

    # PHM scoring fonksiyon grafiği (tez için)
    plot_phm_scoring_function(
        save_path=RESULTS_DIR / "phm_scoring_function.png"
    )

    print(f"\n  Tüm çıktılar: {RESULTS_DIR.resolve()}")
    print("  Track A tamamlandı!")


if __name__ == "__main__":
    main()