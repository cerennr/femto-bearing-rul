"""
hi_labeling.py
--------------
Track B (ve dürüst Track A) için ortak HI metodolojisi.

HI (Health Indicator / bozulma fraksiyonu):
    HI = clip((time_s - t_star_s) / cap_s, 0, 1)     sağlıklı=0, arıza=1
    (yataktan bağımsız, iyi-kurulu hedef — mutlak RUL'un aksine)

HI → RUL okuma (fraksiyon yöntemi, sızıntısız):
    RUL = (time_last - t_star) * (1 - HI_last) / HI_last
    - t_star (CUSUM-FPT) test'te nedensel olarak bilinir → sızıntı yok
    - cap_s KULLANILMAZ (test'te bilinmez)
    - HI trajektorisi EMA ile yumuşatılır, HI^gamma ile kalibre edilir (gamma LOBO ile seçilir)

Tüm fonksiyonlar tek pencere/yatak bazında çalışır; orkestrasyon train_track_*.py'de.
"""

import numpy as np
import pandas as pd

from config import RUL_CAP_MIN, HI_EMA_SPAN, GAMMA_GRID, LOBO_TRUNC_FRACS
from evaluation import phm_accuracy, phm_percent_error


# ── HI etiketi ────────────────────────────────────────────────────────────────

def add_hi_label(df: pd.DataFrame) -> pd.DataFrame:
    """df'e 'hi' sütunu ekler: clip((time_s - t_star_s) / cap_s, 0, 1)."""
    df = df.copy()
    df["hi"] = np.clip((df["time_s"] - df["t_star_s"]) / df["cap_s"], 0.0, 1.0)
    return df


def actual_rul_min(group: pd.DataFrame) -> float:
    """Bir yatağın son penceresindeki gerçek RUL (dk, RUL_CAP_MIN'e kırpılı)."""
    total_life_s = (group["time_s"] + group["rul_s"]).max()
    last_time_s = group["time_s"].iloc[-1]
    return float(np.clip((total_life_s - last_time_s) / 60.0, 0, RUL_CAP_MIN))


# ── HI → RUL okuma (fraksiyon) ────────────────────────────────────────────────

def smooth_hi(hi: np.ndarray, gamma: float = 1.0, span: int = HI_EMA_SPAN) -> np.ndarray:
    """HI^gamma kalibrasyonu + EMA yumuşatma (monoton → sıralamayı/korelasyonu korur)."""
    h = np.clip(hi, 1e-6, 1.0) ** gamma
    return pd.Series(h).ewm(span=span, adjust=False).mean().values


def rul_from_hi(hi: np.ndarray, times_s: np.ndarray, t_star_s: float,
                gamma: float = 1.0, cap: float = RUL_CAP_MIN) -> float:
    """Fraksiyon yöntemi: bir yatağın HI trajektorisinden son-pencere RUL (dk)."""
    hi_s = smooth_hi(hi, gamma)
    hl = max(hi_s[-1], 1e-3)
    deg_elapsed_min = (times_s[-1] - t_star_s) / 60.0
    return float(np.clip(deg_elapsed_min * (1.0 - hl) / hl, 0.0, cap))


# ── Metrikler ─────────────────────────────────────────────────────────────────

def phm_from_pairs(acts, preds) -> float:
    """Ortalama PHM accuracy (dk cinsinden gerçek/tahmin listeleri)."""
    return float(np.mean([phm_accuracy(phm_percent_error(a, p)) for a, p in zip(acts, preds)]))

def endpoint_corr(acts, preds) -> float:
    """Yataklar-arası: son-pencere tahmin RUL ile gerçek RUL korelasyonu."""
    acts, preds = np.asarray(acts, float), np.asarray(preds, float)
    if np.std(acts) < 1e-9 or np.std(preds) < 1e-9:
        return np.nan
    return float(np.corrcoef(acts, preds)[0, 1])

def traj_corr(hi_true: np.ndarray, hi_pred: np.ndarray) -> float:
    """Yatak-içi: tahmin HI trajektorisi ile gerçek HI şekli korelasyonu."""
    if np.std(hi_true) < 1e-6:
        return np.nan   # sabit-HI yatak (erken kesik, hiç bozulmamış)
    return float(np.corrcoef(hi_true, hi_pred)[0, 1])


# ── LOBO ile gamma seçimi ─────────────────────────────────────────────────────

def lobo_gamma_records(hi_pred: np.ndarray, times_s: np.ndarray, rul_s: np.ndarray,
                       t_star_s: float, gammas=GAMMA_GRID,
                       trunc_fracs=LOBO_TRUNC_FRACS) -> dict:
    """
    Held-out bir EĞİTİM yatağı için (run-to-failure), bozulma fazında simüle-truncation
    yaparak her gamma için PHM accuracy listesi döndürür. {gamma: [acc, ...]}.
    Truncation SADECE FPT sonrası (test senaryosuyla aynı) → leak-free.
    """
    n = len(hi_pred)
    fpt_idx = int(np.searchsorted(times_s, t_star_s))
    total_life_s = (times_s + rul_s).max()
    out = {g: [] for g in gammas}
    for fr in trunc_fracs:
        u = fpt_idx + int((n - 1 - fpt_idx) * fr)
        if u <= fpt_idx:
            continue
        true_rul = float(np.clip((total_life_s - times_s[u]) / 60.0, 0, RUL_CAP_MIN))
        for g in gammas:
            pred = rul_from_hi(hi_pred[:u + 1], times_s[:u + 1], t_star_s, g)
            out[g].append(phm_accuracy(phm_percent_error(true_rul, pred)))
    return out


def pick_best_gamma(all_records: list, gammas=GAMMA_GRID) -> float:
    """LOBO foldlarından toplanan kayıtların ortalamasını maksimize eden gamma."""
    agg = {g: [] for g in gammas}
    for rec in all_records:
        for g in gammas:
            agg[g].extend(rec.get(g, []))
    return max(gammas, key=lambda g: np.mean(agg[g]) if agg[g] else -1)
