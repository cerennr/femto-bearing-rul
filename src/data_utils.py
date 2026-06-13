"""
data_utils.py
-------------
FEMTO RUL tahmini için veri işleme ve özellik çıkarımı yapar:
  1. load_bearing()       → Ham CSV'leri okur, pencerelere böler
  2. compute_rul_cusum()  → CUSUM ile bozulma başlangıcını bulur 
  3. extract_features()   → Her pencereden ~30 sayısal özet çıkarır
  4. build_dataset()      → Tüm bearing'leri işler, features.csv olarak kaydeder
"""

import glob
import numpy as np
import pandas as pd
import pywt
from pathlib import Path
from scipy import stats
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm

from config import (
    LEARN_DIR, TEST_DIR,
    LEARNING_BEARINGS, TEST_BEARINGS,
    ACTUAL_RUL_SECONDS, VAL_BEARINGS,
    SAMPLING_FREQ, WINDOW_SIZE, WINDOW_STEP,
    FEATURES_TRAIN, FEATURES_TEST,
    PROCESSED_DIR,
)

# ── CUSUM ve Füzyon Parametreleri ─────────────────────────────────────────────
CUSUM_K_MULT    = 0.5
HEALTHY_FRAC    = 0.15
BURN_IN_WINDOWS = 20

# Füzyon ağırlıkları (Condition 2 override'ı compute_rul_cusum içinde)
RMS_WEIGHT  = 0.70
KURT_WEIGHT = 0.30

# [YENİ] Condition 3 minimum cap garantisi
# Bearing3_1 için CUSUM çok geç alarm verdi → cap sadece 3.2 dk
# Model bu bearing'den degradasyon öğrenemez, minimum 20 dk garanti ediyoruz
MIN_CAP_COND3 = 20.0 * 60.0   # saniye cinsinden (20 dakika)


# BOLUM 1 — VERI OKUMA

def load_bearing(bearing_dir: Path) -> dict:
    """
    Tek bir bearing klasöründeki tüm acc_*.csv dosyalarını okur.
    Zaman: dosya sırası x WINDOW_STEP 
    """
    acc_files = sorted(glob.glob(str(bearing_dir / "acc_*.csv")))
    if not acc_files:
        raise FileNotFoundError(f"Hic acc_*.csv bulunamadi: {bearing_dir}")

    windows, times = [], []
    for idx, fpath in enumerate(acc_files):
        df = pd.read_csv(fpath, header=None,
                         names=["hour","minute","second","usecond","horiz","vert"])
        windows.append(df[["horiz","vert"]].values)
        times.append(idx * float(WINDOW_STEP))

    return {
        "name":    bearing_dir.name,
        "windows": np.array(windows, dtype=np.float32),
        "times":   np.array(times,   dtype=np.float32),
    }


def load_dataset(split: str = "learning") -> dict:
    """Tüm bearing'leri yükler."""
    base_dir = LEARN_DIR if split == "learning" else TEST_DIR
    bearings = LEARNING_BEARINGS if split == "learning" else TEST_BEARINGS
    dataset  = {}
    for bname in bearings:
        bdir = base_dir / bname
        if not bdir.exists():
            print(f"  [ATLA] {bname} bulunamadi.")
            continue
        dataset[bname] = load_bearing(bdir)
    return dataset



# BOLUM 2 — CUSUM TABANLI RUL ETIKETLEME

def _run_cusum(hi: np.ndarray,
               mean_h: float,
               std_h: float,
               start_alarm_idx: int,
               h_mult: float) -> tuple:
    """
    Tek tarafli CUSUM algoritmasi.
    S(t) = max(0,  S(t-1) + x(t) - mean_h - k)
    Alarm: S(t) > h  ve  t > start_alarm_idx (burn-in sonrası)
    """
    k = CUSUM_K_MULT * std_h
    h = h_mult * std_h
    n = len(hi)
    S = np.zeros(n)
    t_star = -1
 
    for t in range(1, n):
        S[t] = max(0.0, S[t-1] + hi[t] - mean_h - k)
        if t_star == -1 and S[t] > h and t > start_alarm_idx:
            t_star = t
 
    return S, t_star
 
 
def compute_rul_cusum(bearing_data: dict, split: str = "learning") -> dict:
    """
    Bir bearing icin CUSUM tabanli RUL etiketleri hesaplar.
 
    Koşula özel kararlar:
      Condition 1: RMS(%70) + Kurtosis(%30), smooth=100, h_mult=10
      Condition 2: Sadece RMS(%100), smooth=150, h_mult=18
                   → Kurtosis ghost hump nedeniyle devre dışı
      Condition 3: RMS(%70) + Kurtosis(%30), smooth=20, h_mult=5
                   → cap >= MIN_CAP_COND3 (20 dk) garantisi
    """
    name      = bearing_data["name"]
    times     = bearing_data["times"]
    windows   = bearing_data["windows"]
    n         = len(times)
    condition = int(name[7])
 
    # ── Koşula özel parametreler ──────────────────────────────────────────────
    # smooth_w : yumusatma penceresi | h_mult : CUSUM esik carpani
    # kurt_w   : kurtosis agirligi (Condition 2'de 0 → ghost hump)
    COND_PARAMS = {
        1: (100, 10.0, KURT_WEIGHT),
        2: (150, 18.0, 0.0),
        3: (20,   5.0, KURT_WEIGHT),
    }
    smooth_w, h_mult, kurt_w = COND_PARAMS[condition]
 
    # ── 1. Bileşik Sağlık Göstergesi ─────────────────────────────────────────
    horiz_data = windows[:, :, 0]
    rms_raw    = np.sqrt(np.mean(horiz_data ** 2, axis=1))
    rms_norm   = (rms_raw - rms_raw.min()) / (np.ptp(rms_raw) + 1e-10)
 
    if kurt_w > 0:
        kurt_raw  = stats.kurtosis(horiz_data, axis=1)
        kurt_norm = (kurt_raw - kurt_raw.min()) / (np.ptp(kurt_raw) + 1e-10)
        composite = RMS_WEIGHT * rms_norm + kurt_w * kurt_norm
    else:
        composite = rms_norm
 
    # Yumuşatma
    hi = (pd.Series(composite)
            .rolling(window=smooth_w, min_periods=1, center=False)
            .mean()
            .values)
 
    # ── 2. Sağlıklı Baseline ──────────────────────────────────────────────────
    n_healthy     = max(int(n * HEALTHY_FRAC), 50)
    baseline_data = hi[BURN_IN_WINDOWS:n_healthy]
    mean_h        = float(np.mean(baseline_data))
    std_h         = float(max(np.std(baseline_data), 1e-6))
 
    # ── 3. CUSUM ──────────────────────────────────────────────────────────────
    cusum_vals, t_star_idx = _run_cusum(
        hi, mean_h, std_h,
        start_alarm_idx=n_healthy,
        h_mult=h_mult
    )
 
    if t_star_idx == -1:
        t_star_idx = int(n * 0.95)
        print(f"  [{name}] CUSUM alarm vermedi → fallback t*={t_star_idx}. pencere")
 
    t_star_s = float(times[t_star_idx])
 
    # ── 4. Toplam Ömür ────────────────────────────────────────────────────────
    obs_end = float(times[-1]) + WINDOW_STEP
    if split == "learning":
        total_life_s = obs_end
    else:
        actual_rul = ACTUAL_RUL_SECONDS.get(name)
        if actual_rul is None:
            raise ValueError(f"{name} icin gercek RUL bilinmiyor.")
        total_life_s = obs_end + actual_rul
 
    # ── 5. Piecewise RUL ──────────────────────────────────────────────────────
    cap_s = total_life_s - t_star_s
 
    # Bearing3_1 cap=3.2 dk → modele öğretecek veri yok
    # En az MIN_CAP_COND3 (20 dk) olmasını garanti ediyoruz
    if condition == 3 and cap_s < MIN_CAP_COND3:
        print(f"  [{name}] Cond3 min cap: {cap_s/60:.1f} → {MIN_CAP_COND3/60:.1f} dk")
        cap_s      = MIN_CAP_COND3
        t_star_s   = total_life_s - cap_s
        t_star_idx = max(0, min(int(t_star_s / WINDOW_STEP), n - 1))
 
    cap_s = max(cap_s, 1.0)
 
    rul_s = np.where(
        times < t_star_s,
        cap_s,
        total_life_s - times
    )
    rul_s = np.clip(rul_s, 0.0, cap_s)
 
    return {
        "rul_s":        rul_s,
        "rul_min":      rul_s / 60.0,
        "t_star_s":     t_star_s,
        "cap_s":        cap_s,
        "total_life_s": total_life_s,
        "hi":           hi,
        "cusum_vals":   cusum_vals,
        "condition":    condition,
        "kurt_w":       kurt_w,
    }

def compute_rul_3sigma(bearing_data: dict, split: str = "learning", sigma_mult: float = 3.0) -> dict:
    """
    3-Sigma (Alarm Bound) kuralini kullanarak bozulma baslangicini (t_star) hesaplar.
    Ortalama + 3 * Standart Sapma sinirini asan ilk nokta bozulma anidir.
    """
    name      = bearing_data["name"]
    times     = bearing_data["times"]
    windows   = bearing_data["windows"]
    n         = len(times)
    condition = int(name[7])
 
    COND_PARAMS = {
        1: (100, KURT_WEIGHT),
        2: (150, 0.0),
        3: (20,  KURT_WEIGHT),
    }
    smooth_w, kurt_w = COND_PARAMS[condition]
 
    horiz_data = windows[:, :, 0]
    rms_raw    = np.sqrt(np.mean(horiz_data ** 2, axis=1))
    rms_norm   = (rms_raw - rms_raw.min()) / (np.ptp(rms_raw) + 1e-10)
 
    if kurt_w > 0:
        kurt_raw  = stats.kurtosis(horiz_data, axis=1)
        kurt_norm = (kurt_raw - kurt_raw.min()) / (np.ptp(kurt_raw) + 1e-10)
        composite = RMS_WEIGHT * rms_norm + kurt_w * kurt_norm
    else:
        composite = rms_norm
 
    hi = (pd.Series(composite)
            .rolling(window=smooth_w, min_periods=1, center=False)
            .mean()
            .values)
 
    n_healthy     = max(int(n * HEALTHY_FRAC), 50)
    baseline_data = hi[BURN_IN_WINDOWS:n_healthy]
    mean_h        = float(np.mean(baseline_data))
    std_h         = float(max(np.std(baseline_data), 1e-6))
 
    # ── 3-Sigma Alarm Kurali ──
    limit = mean_h + sigma_mult * std_h
    t_star_idx = -1
    
    for t in range(n_healthy, n):
        if hi[t] > limit:
            t_star_idx = t
            break

    if t_star_idx == -1:
        t_star_idx = int(n * 0.95)
        print(f"  [{name}] 3-Sigma alarm vermedi (limit={limit:.3f}) -> fallback t*={t_star_idx}. pencere")
 
    t_star_s = float(times[t_star_idx])
 
    obs_end = float(times[-1]) + WINDOW_STEP
    if split == "learning":
        total_life_s = obs_end
    else:
        actual_rul = ACTUAL_RUL_SECONDS.get(name)
        total_life_s = obs_end + actual_rul
 
    cap_s = total_life_s - t_star_s
 
    if condition == 3 and cap_s < MIN_CAP_COND3:
        cap_s      = MIN_CAP_COND3
        t_star_s   = total_life_s - cap_s
        t_star_idx = max(0, min(int(t_star_s / WINDOW_STEP), n - 1))
 
    cap_s = max(cap_s, 1.0)
 
    rul_s = np.where(
        times < t_star_s,
        cap_s,
        total_life_s - times
    )
    rul_s = np.clip(rul_s, 0.0, cap_s)
 
    return {
        "rul_s":        rul_s,
        "rul_min":      rul_s / 60.0,
        "t_star_s":     t_star_s,
        "cap_s":        cap_s,
        "total_life_s": total_life_s,
        "hi":           hi,
        "cusum_vals":   np.zeros_like(hi), # CUSUM donusu beklenmedigi icin dummy
        "condition":    condition,
        "kurt_w":       kurt_w,
    }

# BOLUM 3 — FEATURE EXTRACTION

_FREQS = rfftfreq(WINDOW_SIZE, d=1.0 / SAMPLING_FREQ)


def _dwt_denoise(sig: np.ndarray, wavelet: str = 'db4', level: int = 1) -> np.ndarray:
    """Donoho-Johnstone universal threshold ile wavelet denoising."""
    coeffs  = pywt.wavedec(sig, wavelet, mode='per')
    sigma   = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(sig)))
    coeffs[1:] = (pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:])
    return pywt.waverec(coeffs, wavelet, mode='per')


def _time_features(sig: np.ndarray) -> dict:
    rms      = np.sqrt(np.mean(sig ** 2))
    peak     = np.max(np.abs(sig))
    mean_abs = np.mean(np.abs(sig))
    return {
        "rms":          rms,
        "std":          np.std(sig),
        "peak":         peak,
        "peak2peak":    np.max(sig) - np.min(sig),
        "kurtosis":     float(stats.kurtosis(sig)),
        "skewness":     float(stats.skew(sig)),
        "crest_factor": peak / (rms + 1e-10),
        "shape_factor": rms  / (mean_abs + 1e-10),
    }


def _freq_features(sig: np.ndarray) -> dict:
    mag   = np.abs(rfft(sig)) / WINDOW_SIZE
    total = np.sum(mag ** 2) + 1e-10

    centroid = np.sum(_FREQS * mag) / (np.sum(mag) + 1e-10)
    spread   = np.sqrt(np.sum((_FREQS - centroid)**2 * mag) / (np.sum(mag) + 1e-10))
    prob     = np.clip(mag**2 / total, 1e-12, None)
    entropy  = float(-np.sum(prob * np.log(prob)))

    bands = [(0, 1000), (1000, 5000), (5000, SAMPLING_FREQ / 2)]
    band_e = {}
    for i, (lo, hi_f) in enumerate(bands):
        mask = (_FREQS >= lo) & (_FREQS < hi_f)
        band_e[f"band_energy_{i}"] = float(np.sum(mag[mask]**2) / total)

    return {
        "spectral_centroid": float(centroid),
        "spectral_spread":   float(spread),
        "spectral_entropy":  entropy,
        **band_e,
    }


def _stft_features(sig: np.ndarray) -> dict:
    """
    Tek eksen sinyalinden STFT tabanlı zaman-frekans öznitelikleri.

    STFT parametreleri:
      nperseg = 256  → frekans çözünürlüğü ~100 Hz/bin
      noverlap = 128 → %50 overlap, yeterli zaman çözünürlüğü
      2560 sample / (256-128) hop = ~19 zaman dilimi

    Her frekans bandı için çıkarılan öznitelikler:
      mean_energy   : bandın ortalama enerjisi (genel güç seviyesi)
      std_energy    : bant enerjisinin zamana göre değişimi (kararlılık)
      max_energy    : en yüksek anlık enerji (darbe şiddeti)
      temporal_ratio: (ikinci_yarı_ort) / (ilk_yarı_ort + eps)
                      > 1 → enerji zamanla artıyor (bozunum işareti)
                      < 1 → enerji azalıyor
    Toplam: 4 öznitelik × 3 bant = 12 öznitelik/eksen
    """
    from scipy.signal import stft as scipy_stft

    nperseg  = 256
    noverlap = 128

    _, _, Zxx = scipy_stft(sig, fs=SAMPLING_FREQ,
                           nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx) ** 2          # (freq_bins, time_frames)
    freqs = np.linspace(0, SAMPLING_FREQ / 2, power.shape[0])

    # 3 frekans bandı (FFT ile aynı sınırlar — karşılaştırılabilirlik için)
    bands = [(0, 1000), (1000, 5000), (5000, SAMPLING_FREQ / 2)]
    feats = {}

    for i, (lo, hi_f) in enumerate(bands):
        mask       = (freqs >= lo) & (freqs < hi_f)
        band_power = power[mask, :]           # (band_bins, time_frames)

        if band_power.size == 0:
            # Boş bant — sıfır doldur
            feats[f"stft_band{i}_mean"]  = 0.0
            feats[f"stft_band{i}_std"]   = 0.0
            feats[f"stft_band{i}_max"]   = 0.0
            feats[f"stft_band{i}_tratio"]= 1.0
            continue

        # Zaman boyunca toplam bant enerjisi (her frame için)
        band_energy_over_time = band_power.sum(axis=0)   # (time_frames,)
        total                 = band_energy_over_time.sum() + 1e-10

        mean_e = float(band_energy_over_time.mean())
        std_e  = float(band_energy_over_time.std())
        max_e  = float(band_energy_over_time.max())

        # Temporal ratio: ikinci yarı ortalaması / ilk yarı ortalaması
        n_frames  = len(band_energy_over_time)
        half      = max(n_frames // 2, 1)
        first_h   = band_energy_over_time[:half].mean()
        second_h  = band_energy_over_time[half:].mean()
        t_ratio   = float(second_h / (first_h + 1e-10))

        feats[f"stft_band{i}_mean"]   = mean_e
        feats[f"stft_band{i}_std"]    = std_e
        feats[f"stft_band{i}_max"]    = max_e
        feats[f"stft_band{i}_tratio"] = t_ratio

    return feats


def extract_features(window: np.ndarray) -> dict:
    """
    Tek bir (2560, 2) penceresinden tüm öznitelikleri çıkarır.

    Pipeline:
      1. Wavelet DWT denoising (her iki eksen)
      2. Zaman alanı öznitelikleri  (8 × 2 eksen = 16)
      3. Frekans alanı öznitelikleri (6 × 2 eksen = 12)
      4. STFT zaman-frekans öznitelikleri (12 × 2 eksen = 24)
      5. Çapraz öznitelikler (2)
      Toplam: 54 öznitelik

    h_ = yatay eksen, v_ = dikey eksen, cross_ = çapraz
    """
    horiz = _dwt_denoise(window[:, 0].astype(np.float64))
    vert  = _dwt_denoise(window[:, 1].astype(np.float64))

    feats = {}

    # Zaman alanı
    for k, v in _time_features(horiz).items():
        feats[f"h_{k}"] = v
    for k, v in _time_features(vert).items():
        feats[f"v_{k}"] = v

    # Frekans alanı (FFT)
    for k, v in _freq_features(horiz).items():
        feats[f"h_{k}"] = v
    for k, v in _freq_features(vert).items():
        feats[f"v_{k}"] = v

    # Zaman-frekans (STFT)
    for k, v in _stft_features(horiz).items():
        feats[f"h_{k}"] = v
    for k, v in _stft_features(vert).items():
        feats[f"v_{k}"] = v

    # Çapraz
    feats["cross_corr"]      = float(np.corrcoef(horiz, vert)[0, 1])
    feats["cross_rms_ratio"] = feats["h_rms"] / (feats["v_rms"] + 1e-10)

    return feats


# BOLUM 4 — ANA FONKSİYON

def build_dataset(split: str = "learning", method: str = "cusum", save: bool = True) -> pd.DataFrame:
    """
    Tüm bearing'leri okur, CUSUM veya 3-Sigma ile RUL etiketler,
    wavelet denoising + öznitelik çıkarımı yapar, CSV kaydeder.
    """
    base_dir = LEARN_DIR if split == "learning" else TEST_DIR
    bearings = LEARNING_BEARINGS if split == "learning" else TEST_BEARINGS
    
    if method == "cusum":
        out_path = FEATURES_TRAIN if split == "learning" else FEATURES_TEST
    else:
        out_path = PROCESSED_DIR / f"features_{'train' if split=='learning' else 'test'}_3sigma.csv"

    all_rows = []

    for bname in bearings:
        bdir = base_dir / bname
        if not bdir.exists():
            print(f"  [ATLA] {bname} bulunamadi.")
            continue

        data = load_bearing(bdir)
        if method == "cusum":
            rul_result = compute_rul_cusum(data, split=split)
        else:
            rul_result = compute_rul_3sigma(data, split=split)

        rul_s    = rul_result["rul_s"]
        rul_min  = rul_result["rul_min"]
        t_star_s = rul_result["t_star_s"]
        cap_s    = rul_result["cap_s"]
        total_s  = rul_result["total_life_s"]
        cond     = int(bname[7])

        print(f"  {bname:12s} | t*={t_star_s/60:5.1f} dk "
              f"| cap={cap_s/60:5.1f} dk "
              f"| total={total_s/60:5.1f} dk"
              f"{'  [kurtosis=OFF]' if rul_result['kurt_w']==0 else ''}"
              f"{'  [min_cap garantisi]' if cond==3 and cap_s==MIN_CAP_COND3 else ''}")

        for i, (window, t, rs, rm) in enumerate(
            tqdm(zip(data["windows"], data["times"], rul_s, rul_min),
                 total=len(data["times"]), desc=f"  {bname}", leave=False)
        ):
            row = extract_features(window)
            row["bearing"]    = bname
            row["window_idx"] = i
            row["time_s"]     = float(t)
            row["rul_s"]      = float(rs)
            row["rul_min"]    = float(rm)
            row["t_star_s"]   = t_star_s
            row["cap_s"]      = cap_s
            row["condition"]  = cond
            row["split"]      = split
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n  Kaydedildi: {out_path}")
        print(f"  Sekil: {df.shape[0]} satir x {df.shape[1]} sutun")

    return df

# YARDIMCI FONKSİYONLAR

def get_feature_cols(df: pd.DataFrame) -> list:
    """Meta sütunlar dışındaki öznitelik sütunlarını döndürür."""
    meta = {"bearing", "window_idx", "time_s", "rul_s", "rul_min",
            "t_star_s", "cap_s", "condition", "split"}
    return [c for c in df.columns if c not in meta]


def train_val_split(df: pd.DataFrame) -> tuple:
    """Bearing bazlı train/validation ayrımı. Data leakage olmaz."""
    is_val   = df["bearing"].isin(VAL_BEARINGS)
    train_df = df[~is_val].reset_index(drop=True)
    val_df   = df[is_val].reset_index(drop=True)
    print(f"  Train: {len(train_df)} pencere -> {sorted(train_df['bearing'].unique())}")
    print(f"  Val  : {len(val_df)} pencere -> {sorted(val_df['bearing'].unique())}")
    return train_df, val_df

# HIZLI TEST

if __name__ == "__main__":
    print("=== Learning Set ===")
    df_train = build_dataset(split="learning")

    print("\n=== Test Set ===")
    df_test = build_dataset(split="test")

    print("\n=== Train/Val Split ===")
    train_df, val_df = train_val_split(df_train)

    print(f"\nOzellik sayisi : {len(get_feature_cols(df_train))}")
    print("\nPer-bearing ozet:")
    print(df_train.groupby("bearing")[["rul_min","cap_s","t_star_s"]].agg({
        "rul_min":  ["min","max"],
        "cap_s":    "first",
        "t_star_s": "first",
    }).round(1))