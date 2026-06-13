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

# RUL Parametreleri (eski)
RUL_CAP       = 7500    # saniye (125 dakika) — piecewise lineer tavan
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