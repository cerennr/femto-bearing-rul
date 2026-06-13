"""
src/evaluate_raw_tcn.py
-----------------------
Colab'da eğitilip Drive'a (dolayısıyla yerel klasöre) senkronize olan
satcn_v6_default.keras modelini yerelde test eder.
Bellek dostudur; her rulmanı tek tek işleyip RAM'i temizler.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import tensorflow as tf
import gc

# ── Dizin Yapılandırması ───────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent          # .../femto_rul/src
_PROJECT    = _HERE.parent                             # .../femto_rul
MODELS_DIR  = _PROJECT / "experiments" / "models"
RESULTS_DIR = _PROJECT / "experiments" / "results"
NPY_DIR     = _PROJECT / "experiments" / "numpy_data"

MODEL_PATH = MODELS_DIR / "satcn_v6_default.keras"
RUL_CAP    = 125.0  # Dakika

ACTUAL_RUL_SECONDS = {
    "Bearing1_3": 5730, "Bearing1_4": 339,  "Bearing1_5": 1610,
    "Bearing1_6": 1460, "Bearing1_7": 7570,
    "Bearing2_3": 7530, "Bearing2_4": 1390, "Bearing2_5": 3090,
    "Bearing2_6": 1290, "Bearing2_7": 580,
    "Bearing3_3": 820,
}

# ── PHM Metrikleri ────────────────────────────────────────────────────────────

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

def baseline_standardize(X: np.ndarray, num_baseline: int = 50) -> np.ndarray:
    std_h = max(np.std(X[:min(num_baseline, len(X)), :, 0]), 1e-5)
    std_v = max(np.std(X[:min(num_baseline, len(X)), :, 1]), 1e-5)
    X_norm = X.copy()
    X_norm[:, :, 0] /= std_h
    X_norm[:, :, 1] /= std_v
    return X_norm

def load_bearing_data(bname: str) -> tuple:
    path = NPY_DIR / f"test_{bname}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    data = np.load(path)
    X = data["X"].astype(np.float32)
    actual_rul_s = ACTUAL_RUL_SECONDS[bname]
    y = np.clip(
        ((len(X) - 1 - np.arange(len(X))) * 10.0 + actual_rul_s) / (RUL_CAP * 60.0),
        0.0, 1.0
    ).astype(np.float32)
    return baseline_standardize(X), y

# ── Ana Değerlendirme ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  FEMTO PHM — SA-TCN v6 Yerel Test Değerlendirmesi")
    print("=" * 70)
    
    if not MODEL_PATH.exists():
        print(f"HATA: Model dosyası bulunamadı! {MODEL_PATH}")
        print("Lütfen Colab'daki modelin bu yerel klasöre senkronize olduğunu kontrol edin.")
        return
        
    print(f"Model yükleniyor: {MODEL_PATH.name}...")
    # Safe mode ile lambda/custom engellerini aşıyoruz
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False, safe_mode=False)
    
    test_bearings = list(ACTUAL_RUL_SECONDS.keys())
    results = []
    
    n_cols = 3
    n_rows = int(np.ceil(len(test_bearings) / n_cols))
    fig = plt.figure(figsize=(18, n_rows * 5))
    fig.suptitle("SA-TCN v6 (Default) — Yerel Test HI Tahminleri", fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.3)
    
    for b_idx, bname in enumerate(test_bearings):
        print(f"  {bname} test ediliyor... ", end="", flush=True)
        
        # Veriyi yükle
        X_test, y_test = load_bearing_data(bname)
        
        # Tahminleri parça parça alıp RAM'i koru
        pred_hi = model.predict(X_test, batch_size=256, verbose=0).flatten()
        
        # Post-processing (Gürültüyü engellemek için 30 rolling + monoton)
        pred_hi_smooth = pd.Series(pred_hi).rolling(window=30, min_periods=1).mean().values
        pred_hi_mono = np.minimum.accumulate(pred_hi_smooth)
        
        # Son RUL Tahminleri
        actual_rul_s = ACTUAL_RUL_SECONDS[bname]
        pred_rul_s_raw = float(pred_hi[-1]) * (RUL_CAP * 60.0)
        pred_rul_s_mono = float(pred_hi_mono[-1]) * (RUL_CAP * 60.0)
        
        # Hatalar ve Doğruluk (Monoton versiyon PHM standardıdır)
        err_pct_raw = phm_percent_error(actual_rul_s / 60.0, pred_rul_s_raw / 60.0)
        err_pct_mono = phm_percent_error(actual_rul_s / 60.0, pred_rul_s_mono / 60.0)
        
        acc_raw = phm_accuracy(err_pct_raw)
        acc_mono = phm_accuracy(err_pct_mono)
        
        pen_raw = phm_penalty_score(actual_rul_s, pred_rul_s_raw)
        pen_mono = phm_penalty_score(actual_rul_s, pred_rul_s_mono)
        
        status = "✅" if pen_mono < 2.0 else ("🟠" if pen_mono < 100.0 else "🔴")
        print(f"Raw Acc: {acc_raw:.3f} | Mono Acc: {acc_mono:.3f} | Ceza: {pen_mono:.2f} {status}")
        
        results.append({
            "bearing": bname,
            "actual_min": round(actual_rul_s / 60.0, 1),
            "pred_raw_min": round(pred_rul_s_raw / 60.0, 1),
            "pred_mono_min": round(pred_rul_s_mono / 60.0, 1),
            "acc_raw": round(acc_raw, 4),
            "acc_mono": round(acc_mono, 4),
            "pen_raw": round(pen_raw, 2),
            "pen_mono": round(pen_mono, 2)
        })
        
        # Grafik Çizim
        ax = fig.add_subplot(gs[b_idx // n_cols, b_idx % n_cols])
        t = np.arange(len(y_test)) * 10.0 / 60.0
        
        ax.fill_between(t, y_test, alpha=0.10, color="#42A5F5")
        ax.plot(t, y_test, lw=1.5, color="#42A5F5", label="Gerçek HI")
        ax.plot(t, pred_hi, lw=1.0, color="#EF5350", alpha=0.3, linestyle=":", label="Ham Tahmin")
        ax.plot(t, pred_hi_mono, lw=1.5, color="#E53935", label="Yumuşatılmış Mono")
        
        ax.set_title(f"{bname}\nAcc (Mono)={acc_mono:.3f} | Ceza={pen_mono:.1f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Zaman (dk)", fontsize=8)
        ax.set_ylabel("HI", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        
        # Bellek boşalt
        del X_test, y_test, pred_hi, pred_hi_smooth, pred_hi_mono
        gc.collect()
        
    plt.savefig(RESULTS_DIR / "satcn_v6_local_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # ── Sonuçları Kaydet ve Yazdır ─────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "satcn_v6_local_results.csv", index=False)
    
    mean_acc_raw = df["acc_raw"].mean()
    mean_acc_mono = df["acc_mono"].mean()
    total_pen_raw = df["pen_raw"].sum()
    total_pen_mono = df["pen_mono"].sum()
    
    print("\n" + "="*80)
    print("  SA-TCN v6 DEFAULT — YEREL TEST SONUÇLARI KARŞILAŞTIRMASI")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    print(f"  RAW  Ortalama Doğruluk (Accuracy) : {mean_acc_raw:.4f} | Toplam Ceza: {total_pen_raw:.2f}")
    print(f"  MONO Ortalama Doğruluk (Accuracy) : {mean_acc_mono:.4f} | Toplam Ceza: {total_pen_mono:.2f}")
    print("="*80)
    print(f"  Grafik kaydedildi : {RESULTS_DIR / 'satcn_v6_local_predictions.png'}")
    print(f"  Sonuç CSV         : {RESULTS_DIR / 'satcn_v6_local_results.csv'}")

if __name__ == "__main__":
    main()
