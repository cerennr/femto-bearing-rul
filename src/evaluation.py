"""
evaluation.py
-------------
Model değerlendirme metrikleri ve görselleştirme.

Metrikler:
    - RMSE  : Root Mean Squared Error
    - MAE   : Mean Absolute Error
    - R²    : Determinasyon katsayısı
    - PHM Score : IEEE PHM 2012 Challenge orijinal scoring fonksiyonu

PHM Score mantığı:
    Erken tahmin (pred < actual → %Er > 0) → hafif ceza
    Geç tahmin  (pred > actual → %Er < 0) → ağır ceza
    Skor 0-1 arası, yüksek = iyi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Temel Metrikler
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# PHM 2012 Challenge Score
# ─────────────────────────────────────────────────────────────────────────────

def phm_percent_error(act: float, pred: float) -> float:
    """
    %Er = 100 × (ActRUL - PredRUL) / ActRUL

    Pozitif → erken tahmin (pred < actual)
    Negatif → geç tahmin  (pred > actual)  ← daha ağır cezalandırılır
    """
    return 100.0 * (act - pred) / (act + 1e-10)


def phm_accuracy(percent_error: float) -> float:
    """
    Ai = exp(-ln(0.5) × (Er/5))   if Er ≤ 0  (geç tahmin)
    Ai = exp(+ln(0.5) × (Er/20))  if Er > 0  (erken tahmin)

    Maksimum = 1.0 (mükemmel tahmin)
    Er = -10 → Ai = 0.25  (geç, ağır ceza)
    Er = +20 → Ai = 0.50  (erken, hafif ceza)
    """
    ln05 = np.log(0.5)
    er   = percent_error
    if er <= 0:
        return float(np.exp(-ln05 * (er / 5.0)))
    else:
        return float(np.exp(ln05 * (er / 20.0)))


def phm_score(
    act_ruls:  list | np.ndarray,
    pred_ruls: list | np.ndarray,
) -> dict:
    """
    IEEE PHM 2012 Challenge final skorunu hesaplar.

    Parameters
    ----------
    act_ruls  : gerçek RUL değerleri (dakika)
    pred_ruls : tahmin edilen RUL değerleri (dakika)

    Returns
    -------
    dict:
        score          : final score (0-1)
        percent_errors : her bearing için %Er
        accuracies     : her bearing için Ai
    """
    act  = np.array(act_ruls,  dtype=float)
    pred = np.array(pred_ruls, dtype=float)

    pct_errors = [phm_percent_error(a, p) for a, p in zip(act, pred)]
    accuracies = [phm_accuracy(er) for er in pct_errors]
    score      = float(np.mean(accuracies))

    return {
        "score":          score,
        "percent_errors": pct_errors,
        "accuracies":     accuracies,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation Değerlendirmesi (tüm pencereler)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_val(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    model_name: str  = "Model",
    unit:       str  = "dk",
    verbose:    bool = True,
) -> dict:
    """
    Validation seti üzerinde RMSE, MAE, R² hesaplar.
    (Pencere bazlı — her pencere bir tahmin)
    """
    metrics = {
        "model": model_name,
        "rmse":  rmse(y_true, y_pred),
        "mae":   mae(y_true, y_pred),
        "r2":    r2(y_true, y_pred),
    }
    if verbose:
        print(f"\n  {'-'*40}")
        print(f"  Model : {model_name}")
        print(f"  RMSE  : {metrics['rmse']:.3f} {unit}")
        print(f"  MAE   : {metrics['mae']:.3f} {unit}")
        print(f"  R²    : {metrics['r2']:.4f}")
        print(f"  {'-'*40}")
    return metrics


# -----------------------------------------------------------------------------
# Test Seti PHM Score (bearing bazlı)
# -----------------------------------------------------------------------------

def evaluate_test_bearings(
    predictions: dict,
    unit:        str  = "min",
    verbose:     bool = True,
) -> dict:
    """
    Test bearing'leri için PHM 2012 Challenge skorunu hesaplar.

    Parameters
    ----------
    predictions : {bearing_name: predicted_rul_value}  (dakika cinsinden)
    unit        : "min" veya "s"

    Returns
    -------
    dict: PHM score + detaylar
    """
    from config import ACTUAL_RUL_SECONDS

    names     = list(predictions.keys())
    pred_ruls = [predictions[n] for n in names]

    if unit == "min":
        act_ruls = [ACTUAL_RUL_SECONDS[n] / 60.0 for n in names]
    else:
        act_ruls = [ACTUAL_RUL_SECONDS[n] for n in names]

    result = phm_score(act_ruls, pred_ruls)

    if verbose:
        print(f"\n  {'='*58}")
        print(f"  PHM 2012 Challenge Score : {result['score']:.4f}")
        print(f"  {'-'*58}")
        print(f"  {'Bearing':<14} {'Gerçek':>8} {'Tahmin':>8} "
              f"{'%Er':>7} {'Ai':>7}")
        print(f"  {'-'*58}")
        for i, name in enumerate(names):
            print(f"  {name:<14} {act_ruls[i]:>8.1f} {pred_ruls[i]:>8.1f} "
                  f"{result['percent_errors'][i]:>6.1f}% "
                  f"{result['accuracies'][i]:>7.4f}")
        print(f"  {'='*58}")

    result["names"]     = names
    result["act_ruls"]  = act_ruls
    result["pred_ruls"] = pred_ruls
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Görselleştirme
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    model_name: str = "Model",
    unit:       str = "dakika",
    save_path:  str | Path = None,
):
    """Gerçek vs tahmin RUL grafiği (zaman serisi + scatter)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Zaman serisi
    ax = axes[0]
    ax.plot(y_true, color="#2196F3", lw=1.5, label="Gerçek RUL", alpha=0.9)
    ax.plot(y_pred, color="#F44336", lw=1.5, label="Tahmin RUL", alpha=0.8)
    ax.set_xlabel("Pencere indeksi")
    ax.set_ylabel(f"RUL ({unit})")
    ax.set_title(f"{model_name} — Zaman Serisi")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Scatter
    ax = axes[1]
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    ax.scatter(y_true, y_pred, alpha=0.25, s=6, color="#9C27B0")
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.5, label="Mükemmel")
    ax.set_xlabel(f"Gerçek RUL ({unit})")
    ax.set_ylabel(f"Tahmin RUL ({unit})")
    ax.set_title(f"{model_name} — Scatter")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show()


def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path:  str | Path = None,
):
    """
    Birden fazla modelin metrik karşılaştırması.

    Parameters
    ----------
    results_df : sütunlar: model, rmse, mae, r2, phm_score (opsiyonel)
    """
    metrics = ["rmse", "mae", "r2"]
    if "phm_score" in results_df.columns:
        metrics.append("phm_score")

    n      = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_df)))

    for ax, metric in zip(axes, metrics):
        bars = ax.bar(
            results_df["model"], results_df[metric],
            color=colors, edgecolor="white"
        )
        ax.set_title(metric.upper(), fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.25, axis="y")

        for bar, val in zip(bars, results_df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    plt.suptitle("Track A — Model Karşılaştırması",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show()


def plot_phm_scoring_function(save_path: str | Path = None):
    """PHM 2012 scoring fonksiyonunu görselleştirir (tez için)."""
    er_vals = np.linspace(-50, 50, 500)
    ai_vals = [phm_accuracy(er) for er in er_vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(er_vals, ai_vals, color="#E91E63", lw=2.5)
    ax.axvline(0,   color="gray", ls="--", lw=1)
    ax.axhline(0.5, color="gray", ls=":",  lw=1, alpha=0.7)
    ax.fill_between(er_vals, ai_vals, where=(np.array(er_vals) <= 0),
                    alpha=0.08, color="red",  label="Geç tahmin (ağır ceza)")
    ax.fill_between(er_vals, ai_vals, where=(np.array(er_vals) > 0),
                    alpha=0.08, color="blue", label="Erken tahmin (hafif ceza)")
    ax.annotate("Er=-10 → Ai=0.25", xy=(-10, 0.25),
                xytext=(-40, 0.15), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red"), color="red")
    ax.annotate("Er=+20 → Ai=0.50", xy=(20, 0.5),
                xytext=(28, 0.65), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="blue"), color="blue")
    ax.set_xlabel("%Er (Yüzde hata)", fontsize=11)
    ax.set_ylabel("Ai (Doğruluk skoru)", fontsize=11)
    ax.set_title("PHM 2012 Scoring Fonksiyonu", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show()
