"""
preprocessing.py
----------------
Track A için öznitelik seçimi ve ölçekleme pipeline'ı.

Adımlar (sırayla):
  0. Trend Özellikleri   → Modellerin zamanı görebilmesi için son 5 pencerenin ivmesini ekler.
  1. Variance threshold  → std < eşik olan öznitelikleri at
  2. Correlation filter  → r > 0.95 çiftlerden MI düşük olanı at
  3. Mutual Information  → RUL ile MI skoru hesapla, top-K seç
  4. RobustScaler        → train'de fit, val/test'e transform
  5. PCA (opsiyonel)     → 95% variance

KRİTİK: Tüm fit işlemleri SADECE train seti üzerinde yapılır.
Val ve test setine yalnızca transform uygulanır.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression


# ── Sabitler ──────────────────────────────────────────────────────────────────
STD_THRESHOLD  = 0.01   # Bu std'nin altındaki öznitelikler sabit sayılır → at
CORR_THRESHOLD = 0.95   # Bu korelasyonun üstündeki çiftler redundant → birini at
N_FEATURES     = 25     # MI'ya göre seçilecek öznitelik sayısı
TARGET_COL     = "rul_min"

# Meta sütunlar — bunlar hiçbir zaman öznitelik değil
# NOT: time_s ve condition ARTIK öznitelik olarak kullanılıyor
#   time_s   → normalize edilmiş biçimde (time_s_norm) ekleniyor
#   condition → sayısal operasyon koşulu (1/2/3)
# KRİTİK: deg_progress = (time_s - t_star_s) / cap_s olarak hesaplanır.
#   t_star_s ve cap_s, CUSUM ile tüm ömür bittikten sonra bilinen değerlerdir.
#   Bu yüzden deg_progress modele VERİLMEZ — hedef sızıntısı (target leakage) yapar.
META_COLS = {
    "bearing", "window_idx", "time_s", "rul_s", "rul_min",
    "t_star_s", "cap_s", "split", "deg_progress"
}


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in META_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# Adım 0.1 — Smoothing & Baseline (YENİ)
# ─────────────────────────────────────────────────────────────────────────────

def smooth_and_baseline_correct(
    df: pd.DataFrame,
    span: int = 20,
    baseline_p: int = 20,
    normalize_per_bearing: bool = False,
) -> pd.DataFrame:
    """
    1. Her bearing için EMA smoothing uygular (gürültüyü bastırır)
    2. Her bearing'in sağlıklı başlangıç baseline'ını çıkarır (bozunumu izole eder)
    3. time_s_norm özniteliği ekler: geçen süre / bearing'in gözlem ufku [0-1]
       Bu öznitelik modelin 'ne kadar zamandır çalışıyor' sorusunu yanıtlamasını sağlar.
    4. [opsiyonel] Per-bearing min-max normalization: her özniteliği kendi
       gözlem aralığına göre [0,1]'e sıkıştırır. Farklı ömürde/şiddette
       bearing'ler arasındaki mutlak genlik farkını ortadan kaldırır.
       Örnek: Bearing1_1 h_std∈[−0.04, 2.98] → [0,1]
              Bearing1_2 h_std∈[−0.04, 0.30] → [0,1]  (farklı şiddet, aynı skala)
    """
    df_out = df.copy()
    df_out = df_out.sort_values(['bearing', 'time_s'])

    feat_cols = [c for c in get_feature_cols(df_out)
                 if c not in ('condition', 'time_s_norm')]
    if not feat_cols:
        return df_out

    # 1. EMA Smoothing
    smoothed = df_out.groupby('bearing')[feat_cols].transform(
        lambda x: x.ewm(span=span, min_periods=1, adjust=False).mean()
    )
    df_out[feat_cols] = smoothed

    # 2. Baseline Subtraction
    def subtract_baseline(group):
        baseline = group.iloc[:baseline_p].mean()
        return group - baseline

    corrected = df_out.groupby('bearing')[feat_cols].transform(subtract_baseline)
    df_out[feat_cols] = corrected

    # 3. Sızıntısız Zaman İlerleme İndeksi
    #    time_progress = time_s / max(time_s bu bearing'de)  ∈ [0, 1]
    #    Sadece GÖZLEMLENMİŞ geçen süreyi kullanır.
    #    Gerçek dünyada: rulman ne kadar süredir çalışıyor?
    #    t_star_s veya cap_s KULLANILMAZ → sızıntı yok.
    if 'time_s_norm' in df_out.columns:
        df_out.drop(columns=['time_s_norm'], inplace=True, errors='ignore')
    if 'deg_progress' in df_out.columns:
        df_out.drop(columns=['deg_progress'], inplace=True, errors='ignore')
    max_time = df_out.groupby('bearing')['time_s'].transform('max')
    df_out['time_progress'] = df_out['time_s'] / (max_time + 1e-6)

    # 4. Per-bearing min-max normalization (opsiyonel — LSTM için kritik)
    if normalize_per_bearing:
        def minmax_col(col):
            fmin = col.min()
            fmax = col.max()
            rng  = max(fmax - fmin, 1e-8)
            return (col - fmin) / rng

        normalized = df_out.groupby('bearing')[feat_cols].transform(minmax_col)
        df_out[feat_cols] = normalized

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Adım 0.2 — Trend (İvme) Özellikleri Ekleme (YENİ)
# ─────────────────────────────────────────────────────────────────────────────

def add_trend_features(df: pd.DataFrame, periods: int = 5) -> pd.DataFrame:
    """
    Rulmanların titreşim değişim hızını (türevini) hesaplayarak 
    zaman serisi bağlamını (temporal context) klasik modellere kazandırır.
    """
    # Verinin orjinalini bozmamak için kopyasını al
    df_out = df.copy()
    
    # Rulmanları ve zamanı garanti altına alarak sırala
    df_out = df_out.sort_values(['bearing', 'time_s'])
    
    trend_candidates = [
        'h_std', 'h_rms', 'h_kurtosis', 'h_peak', 
        'h_stft_band1_mean', 'h_stft_band0_mean', 
        'v_std', 'v_rms'
    ]
    
    added_count = 0
    for col in trend_candidates:
        if col in df_out.columns:
            new_col = f"{col}_trend"
            # periods=5 -> Yaklaşık son 50 saniyelik süreçteki fark
            df_out[new_col] = df_out.groupby('bearing')[col].diff(periods=periods).fillna(0.0)
            added_count += 1
            
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Adım 1 — Variance Threshold
# ─────────────────────────────────────────────────────────────────────────────

def variance_filter(df: pd.DataFrame,
                    feature_cols: list,
                    threshold: float = STD_THRESHOLD) -> list:
    stds      = df[feature_cols].std()
    keep_mask = stds >= threshold
    removed   = stds[~keep_mask].index.tolist()
    kept      = stds[keep_mask].index.tolist()

    if removed:
        print(f"  [Variance] {len(removed)} öznitelik atıldı "
              f"(std < {threshold}): {removed[:5]}{'...' if len(removed)>5 else ''}")
    print(f"  [Variance] {len(kept)} öznitelik kaldı.")
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Adım 2 — Correlation Filter
# ─────────────────────────────────────────────────────────────────────────────

def correlation_filter(df: pd.DataFrame,
                       feature_cols: list,
                       threshold: float = CORR_THRESHOLD,
                       mi_scores: pd.Series = None) -> list:
    corr_matrix = df[feature_cols].corr().abs()
    upper       = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper.columns:
        high_corr_partners = upper.index[upper[col] > threshold].tolist()
        for partner in high_corr_partners:
            if partner in to_drop or col in to_drop:
                continue
            if mi_scores is not None:
                drop_col = col if mi_scores.get(col, 0) < mi_scores.get(partner, 0) else partner
            else:
                drop_col = col 
            to_drop.add(drop_col)

    kept = [c for c in feature_cols if c not in to_drop]
    if to_drop:
        print(f"  [Correlation] {len(to_drop)} öznitelik atıldı "
              f"(r > {threshold}): {list(to_drop)[:5]}{'...' if len(to_drop)>5 else ''}")
    print(f"  [Correlation] {len(kept)} öznitelik kaldı.")
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Adım 3 — Mutual Information Sıralaması
# ─────────────────────────────────────────────────────────────────────────────

def compute_mi_scores(df: pd.DataFrame,
                      feature_cols: list,
                      target_col: str = TARGET_COL,
                      random_state: int = 42) -> pd.Series:
    X = df[feature_cols].fillna(0).values
    y = df[target_col].values

    scores = mutual_info_regression(X, y, random_state=random_state)
    mi     = pd.Series(scores, index=feature_cols).sort_values(ascending=False)
    return mi


def select_top_features(mi_scores: pd.Series,
                        feature_cols: list,
                        n: int = N_FEATURES) -> list:
    available = [c for c in mi_scores.index if c in feature_cols]
    selected  = available[:n]
    print(f"  [MI Selection] Top {n} öznitelik seçildi.")
    print(f"    En yüksek 5: {selected[:5]}")
    print(f"    En düşük  5: {selected[-5:]}")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Ana Preprocessor Sınıfı
# ─────────────────────────────────────────────────────────────────────────────

class Preprocessor:
    def __init__(
        self,
        n_features:            int   = N_FEATURES,
        std_threshold:         float = STD_THRESHOLD,
        corr_threshold:        float = CORR_THRESHOLD,
        use_pca:               bool  = False,
        pca_variance:          float = 0.95,
        target_col:            str   = TARGET_COL,
        add_trends:            bool  = True,
        normalize_per_bearing: bool  = False,  # LSTM için: bearing bazlı [0,1] normalizasyonu
    ):
        self.n_features            = n_features
        self.std_threshold         = std_threshold
        self.corr_threshold        = corr_threshold
        self.use_pca               = use_pca
        self.pca_variance          = pca_variance
        self.target_col            = target_col
        self.add_trends            = add_trends
        self.normalize_per_bearing = normalize_per_bearing

        self.selected_features: list      = []
        self.mi_scores:         pd.Series = None
        self.scaler:            RobustScaler = RobustScaler()
        self.pca:               PCA | None   = None

    def fit_transform(
        self, train_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        print("\n=== Preprocessor Fit ===")
        
        # YENİ: Smoothing and Baseline Correction
        print("  [Smoothing & Baseline] EMA (span=20) ve referans baseline sıfırlama...")
        train_df = smooth_and_baseline_correct(
            train_df, span=20, baseline_p=20,
            normalize_per_bearing=self.normalize_per_bearing
        )

        # 0. Trend Özelliklerini Ekle
        if self.add_trends:
            print("  [Trend] Değişim hızı (ivme) özellikleri ekleniyor...")
            train_df = add_trend_features(train_df)
            
        all_feats = get_feature_cols(train_df)
        print(f"  Başlangıç öznitelik sayısı: {len(all_feats)}")

        # 1. Variance filter
        feats = variance_filter(train_df, all_feats, self.std_threshold)

        # 2. MI skorları (correlation filter'dan önce hesapla)
        print("  MI skorları hesaplanıyor...")
        self.mi_scores = compute_mi_scores(train_df, feats, self.target_col)

        # 3. Correlation filter (MI rehberliğinde)
        feats = correlation_filter(train_df, feats,
                                   self.corr_threshold, self.mi_scores)

        # 4. Top-N seçimi
        self.selected_features = select_top_features(
            self.mi_scores, feats, self.n_features
        )

        # 5. RobustScaler fit
        X = train_df[self.selected_features].fillna(0).values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # 6. PCA (opsiyonel)
        if self.use_pca:
            self.pca = PCA(n_components=self.pca_variance, random_state=42)
            X_scaled = self.pca.fit_transform(X_scaled)
            print(f"  [PCA] {self.pca.n_components_} bileşen "
                  f"({self.pca_variance*100:.0f}% varyans)")

        y = train_df[self.target_col].values.astype(np.float32)
        print(f"\n  Çıktı: X={X_scaled.shape}, y={y.shape}")
        return X_scaled, y

    def transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.selected_features:
            raise RuntimeError("Önce fit_transform() çağrılmalı.")

        # YENİ: Smoothing and Baseline Correction
        df = smooth_and_baseline_correct(
            df, span=20, baseline_p=20,
            normalize_per_bearing=self.normalize_per_bearing
        )

        # 0. Trend Özelliklerini Ekle (Eğitimde yapıldıysa testte de yapılmalı)
        if self.add_trends:
            df = add_trend_features(df)

        missing = [c for c in self.selected_features if c not in df.columns]
        if missing:
            print(f"  [UYARI] Eksik sütunlar sıfırla dolduruldu: {missing}")

        X = (df.reindex(columns=self.selected_features, fill_value=0.0)
               .fillna(0).values.astype(np.float32))
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)

        if self.use_pca and self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)

        if self.target_col in df.columns:
            y = df[self.target_col].values.astype(np.float32)
        else:
            y = np.full(len(df), np.nan, dtype=np.float32)

        return X_scaled, y

    def get_feature_importance_df(self) -> pd.DataFrame:
        if self.mi_scores is None:
            raise RuntimeError("Önce fit_transform() çağrılmalı.")
        df = pd.DataFrame({
            "feature":  self.selected_features,
            "mi_score": [self.mi_scores.get(f, 0) for f in self.selected_features],
        }).sort_values("mi_score", ascending=False).reset_index(drop=True)
        return df

    def save(self, path: str | Path):
        joblib.dump(self, path)
        print(f"  Preprocessor kaydedildi: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        return joblib.load(path)