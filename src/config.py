from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Veri Yolları 
DATA_ROOT = BASE_DIR / "data" / "raw"
LEARN_DIR = DATA_ROOT / "Learning_set"
TEST_DIR  = DATA_ROOT / "Test_set"

# İşlenmiş veri çıktıları
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
FEATURES_TRAIN = PROCESSED_DIR / "features_train.csv"
FEATURES_TEST  = PROCESSED_DIR / "features_test.csv"

# Ham sinyal (.npz) — Track B raw-TCN için
NPY_DIR = BASE_DIR / "data" / "numpy_data"

# Deney çıktıları
MODELS_DIR  = BASE_DIR / "experiments" / "models"
RESULTS_DIR = BASE_DIR / "experiments" / "results"

# Bearing Grupları 
LEARNING_BEARINGS = [
    "Bearing1_1", "Bearing1_2",
    "Bearing2_1", "Bearing2_2",
    "Bearing3_1", "Bearing3_2",
]

TEST_BEARINGS = [
    "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
    "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
    "Bearing3_3",
]

# Validation için her koşuldan 1 bearing ayrılır
VAL_BEARINGS = ["Bearing1_2", "Bearing2_2", "Bearing3_2"]

# Sinyal Parametreleri 
SAMPLING_FREQ = 25600   # Hz  (vibrasyon örnekleme frekansı, 25.6 kHz)
WINDOW_SIZE   = 2560    # sample (her penceredeki örnek sayısı)
WINDOW_STEP   = 10      # saniye (pencereler arası süre)

# RUL Parametreleri
RUL_CAP_S     = 7500    # saniye — piecewise lineer tavan (data_utils etiketleme)
RUL_CAP_MIN   = 125.0   # dakika — modelleme/değerlendirme tavanı
RUL_CAP       = RUL_CAP_S   # geriye dönük uyumluluk (eski kod saniye bekliyor)
FAILURE_THRESH = 20     # g — bu genliği aşınca arıza kabul edilir

# Gerçek RUL Değerleri (Dokümandaki Table 3'ten alınmıştır) 
ACTUAL_RUL_SECONDS = {
    "Bearing1_3": 5730,
    "Bearing1_4": 339,
    "Bearing1_5": 1610,
    "Bearing1_6": 1460,
    "Bearing1_7": 7570,
    "Bearing2_3": 7530,
    "Bearing2_4": 1390,
    "Bearing2_5": 3090,
    "Bearing2_6": 1290,
    "Bearing2_7": 580,
    "Bearing3_3": 820,
}

# Çalışma Koşulları
OPERATING_CONDITIONS = {
    1: {"rpm": 1800, "load_n": 4000},
    2: {"rpm": 1650, "load_n": 4200},
    3: {"rpm": 1500, "load_n": 5000},
}

# ── DÜRÜSTLÜK FLAG'LERİ ───────────────────────────────────────────────────────
# time_progress = time_s/max(time_s): test'te daima 1.0 (dejenere). PHM'i şişirir
# ama yatak-içi korelasyonu negatife çevirir → dürüst pipeline'da KAPALI.
USE_TIME_PROGRESS = False
# Sağlıklı-faz filtresi (rul_min < 122.5): tezin 3.6.1 ablasyonu için flag.
# Dürüst varsayılan = kapalı (filtre uzun-ömürlü yatakların tahminini bozuyor).
USE_HEALTHY_FILTER = False
DEGRAD_THRESH_FRAC = 0.98   # filtre açıksa: rul_min < RUL_CAP_MIN * 0.98

# ── ÖZNİTELİK MÜHENDİSLİĞİ (preprocessing) ────────────────────────────────────
EMA_SPAN       = 20     # öznitelik EMA yumuşatma penceresi
BASELINE_P     = 20     # baseline çıkarma için ilk N pencere
TREND_PERIODS  = 5      # trend (diff) periyodu
N_FEATURES     = 25     # MI ile seçilecek öznitelik sayısı
STD_THRESHOLD  = 0.01
CORR_THRESHOLD = 0.95

# ── HI HEDEFİ + HI→RUL OKUMA (Track B) ────────────────────────────────────────
# HI = clip((time_s - t_star_s) / cap_s, 0, 1)  → bozulma fraksiyonu (sağlıklı 0, arıza 1)
# RUL okuma (fraksiyon): RUL = (time-t_star) * (1-HI)/HI, EMA + gamma kalibrasyon
HI_EMA_SPAN    = 20     # tahmin HI trajektorisi yumuşatma
GAMMA_GRID     = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]   # LOBO ile seçilecek HI^gamma
LOBO_TRUNC_FRACS = [0.3, 0.5, 0.7, 0.85, 0.95]          # LOBO simüle-truncation noktaları

# ── DİZİ / DERİN ÖĞRENME ──────────────────────────────────────────────────────
SEQ_LEN  = 24          # W: dizi penceresi (feature-seq ve LOBO için)
# 15 seed ensemble → tek-seed varyansını (±0.04 PHM) bastırır, tekrarlanabilir tek skor.
SEEDS    = [0, 1, 2, 3, 5, 7, 11, 13, 17, 21, 33, 42, 77, 101, 202]
RAW_SEEDS = SEEDS[:5]  # raw-TCN ağır → daha az seed
EPOCHS   = 120
BATCH_SIZE = 128
RAW_WINDOW = (WINDOW_SIZE, 2)  # ham sinyal girdi şekli (2560, 2)

# Meta sütunlar — modele asla öznitelik olarak verilmez
# 'hi' ve 'rul_min' HEDEF; 'deg_progress'/'time_s_norm' sızıntı; 't_star_s'/'cap_s' gelecek bilgisi
META_COLS = {
    "bearing", "window_idx", "time_s", "rul_s", "rul_min", "hi",
    "t_star_s", "cap_s", "condition", "split", "deg_progress", "time_s_norm",
}