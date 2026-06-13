"""
sensitivity_analysis_thresh.py
------------------------------
Runs a threshold sensitivity analysis for DEGRAD_THRESH on XGBoost.
Leaves all core modules untouched.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_utils import train_val_split
from preprocessing import Preprocessor
from models import XGBoostModel
from evaluation import evaluate_val, evaluate_test_bearings
from config import FEATURES_TRAIN, FEATURES_TEST

def get_test_prediction(model, prep, test_df: pd.DataFrame) -> dict:
    predictions = {}
    for bname in test_df["bearing"].unique():
        bdf   = test_df[test_df["bearing"] == bname].sort_values("time_s")
        X_b, _ = prep.transform(bdf)
        pred = model.predict(X_b[-1:])
        predictions[bname] = float(pred[0])
    return predictions

def apply_standard_rul(df: pd.DataFrame, max_rul: float = 125.0) -> pd.DataFrame:
    total_life = df.groupby('bearing').apply(lambda g: (g['time_s'] + g['rul_s']).max())
    total_map = df['bearing'].map(total_life)
    df['rul_min'] = np.clip((total_map - df['time_s']) / 60.0, 0, max_rul)
    return df

def main():
    print("="*60)
    print("  DEGRAD_THRESH SENSITIVITY ANALYSIS (XGBoost)")
    print("="*60)

    # 1. Veri Yükle
    df_train_full = pd.read_csv(FEATURES_TRAIN)
    df_test       = pd.read_csv(FEATURES_TEST)
    
    df_train_full = apply_standard_rul(df_train_full)
    df_test       = apply_standard_rul(df_test)
    
    train_df_baseline, val_df_baseline = train_val_split(df_train_full)

    RUL_CAP = 125.0
    threshold_multipliers = [1.00, 0.99, 0.98, 0.95]
    
    results = []

    for mult in threshold_multipliers:
        thresh = RUL_CAP * mult
        print(f"\n[Test] DEGRAD_THRESH = {thresh:.2f} dk ({mult*100:.0f}%)")
        
        # Filtreleme
        train_df = train_df_baseline[train_df_baseline['rul_min'] < thresh].reset_index(drop=True)
        val_df   = val_df_baseline[val_df_baseline['rul_min'] < thresh].reset_index(drop=True)
        
        # Preprocessing
        prep = Preprocessor(n_features=30, use_pca=False)
        X_train, y_train = prep.fit_transform(train_df)
        X_val,   y_val   = prep.transform(val_df)
        
        # XGBoost Modeli
        model = XGBoostModel(n_estimators=400, learning_rate=0.02, max_depth=6)
        model.fit(X_train, y_train)
        
        # Validation Değerlendirme
        y_val_pred = model.predict(X_val)
        val_metrics = evaluate_val(y_val, y_val_pred, model_name=model.name, verbose=False)
        
        # Test (PHM Skoru) Değerlendirmesi
        test_preds = get_test_prediction(model, prep, df_test)
        phm_result = evaluate_test_bearings(test_preds, unit="min")
        
        results.append({
            "Ratio": f"{mult*100:.0f}%",
            "Threshold_min": round(thresh, 2),
            "Train_size": len(train_df),
            "R2": round(val_metrics["r2"], 4),
            "PHM_Score": round(phm_result["score"], 4)
        })
        
    res_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("  DUYARLILIK ANALIZI SONUCLARI")
    print("="*60)
    print(res_df.to_string(index=False))
    
    # Kaydet
    out_path = Path("../experiments/results/sensitivity_analysis.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"\n  Sonuclar kaydedildi: {out_path.resolve()}")

if __name__ == "__main__":
    main()
