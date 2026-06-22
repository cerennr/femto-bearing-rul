"""
sequence.py
-----------
Track B girdileri:
  (1) Öznitelik dizileri: Preprocessor çıktısı (seçili+ölçekli 25 öznitelik) →
      yatak-içi nedensel (N, W, F) pencereler  (GRU/TCN için)
  (2) Ham sinyal pencereleri: .npz → baseline_standardize → (N, 2560, 2),
      HI etiketi CSV'den pencere sırasıyla hizalanır  (raw-TCN için)
"""
import numpy as np
import pandas as pd

from config import NPY_DIR, SEQ_LEN, WINDOW_SIZE


# ── (1) Öznitelik dizileri ────────────────────────────────────────────────────

def make_feature_sequences(feat_matrix: np.ndarray, W: int = SEQ_LEN) -> np.ndarray:
    """(n, F) ölçekli öznitelik matrisi → (n, W, F) nedensel pencereler (pad=ilk değer)."""
    n = len(feat_matrix)
    idx = [[max(0, i - j) for j in range(W - 1, -1, -1)] for i in range(n)]
    return feat_matrix[np.array(idx)]

def build_feature_dataset(prep, df: pd.DataFrame, W: int = SEQ_LEN):
    """
    Eğitilmiş Preprocessor ile df'i dizilere çevirir.
    Döndürür: X (toplam, W, F), y_hi (toplam,), groups (yatak adları, sıralı liste)
    """
    Xs, ys, groups = [], [], []
    for b, g in df.groupby("bearing"):
        g = g.sort_values("time_s")
        Xf, _ = prep.transform(g.copy())            # (n, F) ölçekli
        Xs.append(make_feature_sequences(Xf, W))
        ys.append(g["hi"].values.astype(np.float32))
        groups.append((b, len(g)))
    return np.concatenate(Xs), np.concatenate(ys), groups


# ── (2) Ham sinyal pencereleri ────────────────────────────────────────────────

def baseline_standardize(X: np.ndarray, n: int = 50) -> np.ndarray:
    """Her kanalı ilk n pencerenin std'ine böler (bozulma genliğini korur)."""
    Xn = X.copy().astype(np.float32)
    for ch in (0, 1):
        s = max(np.std(X[:min(n, len(X)), :, ch]), 1e-5)
        Xn[:, :, ch] /= s
    return Xn

def load_raw_bearing(bname: str, is_test: bool, meta_df: pd.DataFrame):
    """
    .npz ham X yükler, baseline_standardize uygular, HI + zaman + t_star CSV'den hizalar.
    meta_df: features_train/test birleşik (bearing, window_idx, time_s, rul_s, t_star_s, cap_s, hi).
    Döndürür: X (n,2560,2), hi_true (n,), times_s (n,), t_star_s, total_life_s
    """
    prefix = "test" if is_test else "train"
    d = np.load(NPY_DIR / f"{prefix}_{bname}.npz")
    X = d["X"].astype(np.float32)
    m = meta_df[meta_df.bearing == bname].sort_values("window_idx")
    assert len(m) == len(X), f"{bname}: csv {len(m)} != npz {len(X)}"
    hi = np.clip((m["time_s"].values - m["t_star_s"].values) / m["cap_s"].values, 0, 1).astype(np.float32)
    total_life_s = (m["time_s"] + m["rul_s"]).max()
    return (baseline_standardize(X), hi, m["time_s"].values.astype(np.float32),
            float(m["t_star_s"].iloc[0]), float(total_life_s))
