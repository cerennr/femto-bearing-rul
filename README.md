# FEMTO RUL — Rulman Kalan Faydalı Ömür Tahmini

IEEE PHM 2012 (FEMTO-PRONOSTIA) titreşim veri setinde rulman **Kalan Faydalı Ömür (RUL)** tahmini.
Geleneksel makine öğrenmesi (Track A) ile derin öğrenme (Track B) yöntemleri, **11 test rulman** üzerinden karşılaştırılır.

## Yaklaşım

- **Hedef = Sağlık Göstergesi (HI):** `HI = clip((t − t*) / cap, 0, 1)`
  (sağlıklı = 0, arıza = 1). t* (bozulma başlangıcı) CUSUM ile bulunur. 
- **HI → RUL:** fraksiyon yöntemi + LOBO (leave one bearing out)  ile seçilen γ kalibrasyonu yapıldı.
- **Değerlendirme:** PHM 2012 skoru + MAE/RMSE + korelasyon (endpoint_r, traj_r) + **LOBO** (6-fold).

## Sonuçlar (11 test rulmanı)

| Model | Girdi | PHM | MAE (dk) | endpoint_r | LOBO traj_r |
|---|---|---:|---:|---:|---:|
| Random Forest | tek-pencere öznitelik | 0.152 | 40.5 | +0.16 | — |
| XGBoost | tek-pencere öznitelik | 0.186 | 44.5 | −0.07 | — |
| **GRU** | öznitelik dizisi (24 pencere) | **0.230** | **~31** | **+0.59** | **0.61** |
| TCN | öznitelik dizisi (24 pencere) | 0.185 | ~52 | +0.50 | 0.30 |
| raw-TCN | ham sinyal (2560×2) | 0.131 | 54.3 | −0.05 | 0.60 |

> DL skorları 5-seed ensemble. GRU PHM seed varyansı: **0.200 ± 0.029**. 

## Temel Bulgular

1. **Dizi-DL > tek-pencere ML:** GRU, MAE (31 vs 44) ve rulman sıralaması (endpoint_r +0.59 vs
   −0.07) açısından geleneksel ML'i istikrarlı biçimde geçer.
2. **Otomatik özellik çıkarımı, manuel özellik çıkarımı sonuçlarını yakalayamaz:** ham sinyalden öğrenen raw-TCN, bozulma *şeklini*
   genelliyor (LOBO traj_r 0.60) ama mutlak RUL'u veremiyor (endpoint_r ≈ 0) — el-yapımı öznitelik
   mühendisliğinin altında kalıyor.
3. **Fiziksel limit (truncation):** 4 rulman (1_6, 2_4, 2_6, 2_7) kesim anında ~%80 bozuk olmasına
   rağmen titreşimleri sağlıklı görünüyor (ani arıza). Bu rulmanlar prensipte tahmin edilemez —
   hiçbir model çözemez.
4. **PHM metriği kararsız:** asimetrik + tavanda doygun yapısı ve yalnızca 11 test rulmanı nedeniyle
   seed/çalıştırma arası ciddi oynamaktadır. Bu nedenle karşılaştırma
   **MAE + endpoint_r + LOBO traj_r** üzerinden yapıldı; PHM ikincil raporlandı.

## Yapı

```
src/
  config.py         sabitler, yollar, dürüstlük flag'leri
  data_utils.py     CUSUM + 54 öznitelik çıkarımı → CSV
  preprocessing.py  öznitelik seçimi (MI top-25) + ölçekleme
  hi_labeling.py    HI hedefi, HI→RUL, LOBO-gamma, metrikler
  sequence.py       dizi ve ham pencere hazırlama
  models.py         Random Forest, XGBoost
  dl_models.py      GRU, TCN, raw-TCN
  evaluation.py     metrikler ve grafikler
  train_track_a.py  Track A (RF + XGBoost) 
  train_track_b.py  Track B (GRU + TCN + raw-TCN)
data/{raw, numpy_data, processed}
experiments/{models, results}
notebooks/          görselleştirme 
```

## Çalıştırma

Projeyi çalıştırmak için aşağıdaki adımları sırasıyla izleyin:

```bash
# 1. Repoyu klonlayın ve dizine girin
git clone https://github.com/cerennr/femto-bearing-rul.git
cd femto-bearing-rul

# 2. Python sanal ortamı (virtual environment) oluşturun ve aktifleştirin
python -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate

# 3. Gerekli kütüphaneleri kurun
pip install -r requirements.txt

# 4. Veri Seti Hazırlığı (Kaggle)
Özellik çıkarımı ve model eğitimi yapabilmek için ham veri setine ve numpy dizilerine ihtiyacınız vardır. Veri boyutu büyük olduğu için GitHub'da yer almamaktadır.

Projenin veri setini [Kaggle Linki Buraya Gelecek] adresinden indirin.
İndirdiğiniz arşiv dosyasının içindeki raw/ ve numpy_data/ klasörlerini, projenin kök dizinindeki data/ klasörünün içine yerleştirin.
bash


# 5. Öznitelik Çıkarımı (Feature Extraction)
# Ham verilerden (data/raw/) 54 farklı özelliği çıkarıp processed klasörüne CSV olarak kaydeder.
python src/data_utils.py
# 6. Modellerin Eğitimi ve Değerlendirilmesi
python src/train_track_a.py    # ML modelleri 
python src/train_track_b.py    # DL modelleri 

## Veri

IEEE PHM 2012 Prognostic Challenge — FEMTO-ST / PRONOSTIA hızlandırılmış rulman bozulma testi.
6 eğitim (sonuna kadar arızalı) + 11 test (kesik) rulmanı, 3 çalışma koşulu, 25.6 kHz titreşim.
