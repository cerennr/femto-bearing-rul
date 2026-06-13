# FEMTO RUL Tahmini

Rulman ömrü (Remaining Useful Life - RUL) tahmini için FEMTO-ST veri setini kullanan makine öğrenmesi pipeline'ı.

## Yöntem

Ham titreşim sinyallerinden DWT, FFT ve STFT ile öznitelik çıkarımı yapılır. CUSUM algoritması ile bozulma başlangıç noktası tespit edilerek piecewise linear RUL etiketi üretilir. Tüm ön işleme adımları (variance filter, mutual information, RobustScaler) **sadece eğitim seti** üzerinde öğrenilir.

## Proje Yapısı

```
femto_rul/
├── data/
│   ├── raw/              # Ham FEMTO verisi (Git'e dahil değil)
│   └── processed/        # İşlenmiş öznitelik CSV'leri (Git'e dahil değil)
├── experiments/
│   ├── models/           # Eğitilmiş modeller (Git'e dahil değil)
│   ├── numpy_data/       # Ara numpy dosyaları (Git'e dahil değil)
│   └── results/          # Metrik ve grafikler
├── notebooks/
├── src/
│   ├── config.py
│   ├── data_utils.py     # Veri yükleme, CUSUM, öznitelik çıkarımı
│   ├── preprocessing.py  # Öznitelik seçimi ve ölçekleme
│   ├── models.py         # RF ve XGBoost modelleri
│   ├── train_track_a.py  # Ana eğitim scripti
│   └── evaluation.py     # Metrik ve görselleştirme
└── requirements.txt
```

## Kurulum

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Veri Hazırlığı

Ham FEMTO verisi data/raw/ altına yerleştirilmeli:

```bash
cd src
python data_utils.py   # features_train.csv ve features_test.csv üretir
```

## Eğitim

```bash
cd src
python train_track_a.py
```

## Veri Sızıntısı Notları

- t_star_s, cap_s ve deg_progress META_COLS içinde tanımlıdır, modele verilmez.
- Train/Val ayrımı rulman (bearing) bazında yapılır.
- Scaler ve feature selector yalnızca train verisiyle fit edilir.

## Modeller

| Model | n_estimators | learning_rate | max_depth |
|-------|-------------|---------------|-----------|
| Random Forest | 300 | - | - |
| XGBoost | 400 | 0.02 | 6 |

## Gereksinimler

Python 3.10+ - bkz. requirements.txt
