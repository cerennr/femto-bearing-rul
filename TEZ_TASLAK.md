# TEZ TASLAĞI (Yeniden Yapılandırma)

> Bu dosya tezin **iskeletidir**. Her bölümde: **[DURUM]** korunuyor / değişiyor / yeni-içerik, ardından
> *ne yapılacak* ve *kullanılacak sayılar*. İçeriği kullanıcı dolduracak; bu taslak haritadır.
>
> **2 temel karar (sabitlendi):**
> 1. **Dürüst reframe** — manşet GRU 0.261 (DL) + XGB 0.186 (ML). Filtreli 0.2748 BIRAKILDI; filtre artık
>    "PHM'i şişirir ama korelasyonu öldürür" ablasyonu.
> 2. **SA-TCN çıkarıldı** — DL bölümü yalnızca GRU + TCN + raw-TCN.

---

## BÖLÜM 1 — GİRİŞ  `[KORUNUYOR + küçük ekleme]`

- **1.1 Amaç ve Önem** — korunuyor.
- **1.2 Problemin Tanımı** — korunuyor. *Ekle:* tek anlık pencereden mutlak RUL kestiriminin
  ill-posed olduğu, bunun yerine HI (bozulma durumu) tahmin → RUL'a çevirme yaklaşımının gerekçesi.
- **1.3 Literatür Özeti** — korunuyor.
- **YENİ → 1.4 Araştırma Soruları** (veya 1.2 sonuna):
  - **RQ1:** Öznitelik dizisi üzerinde DL (GRU/TCN), tek-pencere geleneksel ML'i geçer mi?
  - **RQ2:** Ham sinyalden otomatik öğrenme (raw-TCN), manuel öznitelik mühendisliğini yakalar mı?

## BÖLÜM 2 — KURAMSAL ÇERÇEVE  `[KORUNUYOR]`

- 2.1 Bakım yöntemleri (2.1.1–2.1.7, RUL kavramı) — korunuyor.
- 2.2 RUL yaklaşımları (veri/fizik/hibrit) — korunuyor.
- 2.3 ML temelleri — korunuyor.
- 2.4 Regresyon — korunuyor.
- 2.5 ML algoritmaları (RF, XGBoost, LightGBM, SVR) — korunuyor.
- 2.6 Derin öğrenme ve zaman serisi (ANN, DL farkları, RNN, LSTM) — korunuyor.
  - *Not:* SA-TCN/öz-dikkat anlatımı buradan da temizlenecek (artık kullanılmıyor).
    Yerine **GRU/TCN** kuramı eklenebilir (kullanılan modeller bunlar).

---

## BÖLÜM 3 — YÖNTEM

### 3.1 FEMTO-PRONOSTIA Veri Seti  `[KORUNUYOR]`
6 eğitim + 11 test yatağı, 3 koşul, ham `(2560,2)` pencere (25.6 kHz, 0.1 s / 10 s).

### 3.2 Veri Ön İşleme
- **3.2.1 CUSUM-FPT** `[KORUNUYOR]` — koşula özel eşik, t_star tespiti.
- **3.2.2 Wavelet Denoising (db4)** `[KORUNUYOR]`
- **3.2.3 Öznitelik Çıkarımı (54)** `[KORUNUYOR]` — zaman16 + frekans12 + STFT24 + çapraz2.
- **3.2.4 Yumuşatma + Baseline** `[KORUNUYOR]` — EMA(span20).
- **3.2.5 Trend (İvme) Öznitelikleri** `[KORUNUYOR]` — diff5.
- **3.2.6 Hedef Değişken** `[DEĞİŞİYOR — kritik]`
  - ESKİ: piecewise-lineer RUL (RUL_CAP=125 dk, ömrün %50-73'ü tavanda → etiket doygunluğu).
  - YENİ: **HI (bozulma fraksiyonu)** = `clip((time_s − t_star_s)/cap_s, 0, 1)`; sağlıklı=0, arıza=1.
  - *Ekle:* HI'nin neden tercih edildiği (etiket doygunluğu sorununu çözer, yatay karşılaştırma sağlar).
  - *Ekle:* **sızıntı uyarısı** — `deg_progress`, `time_progress`, `cap` test'te bilinmez/dejenere;
    META_COLS'a alınıp dışlandı (bu, dürüstlük metodolojisinin parçası, kısaca anlatılmalı).

### 3.3 → **YENİ: 3.3 HI'dan RUL'a Çevrim (Sızıntısız Okuma)**  `[YENİ]`
- `RUL = (time_last − t_star)·(1 − HI_last)/HI_last` (fraksiyon yöntemi).
- t_star test'te nedensel bilinir; **cap KULLANILMAZ**.
- HI trajektorisi EMA ile yumuşatılır, `HI^γ` ile kalibre edilir.
- **γ, LOBO (Leave-One-Bearing-Out) ile seçilir** — test'e bakılmaz (test-tuning yok).
  *Bu, eski tezdeki "tek son pencere okuması"nın yerini alır.*

### 3.4 Özellik Seçimi  `[KORUNUYOR]`
- 3.4.1 Varyans + korelasyon → **MI top-25**. RobustScaler.

### 3.5 Performans Metrikleri  `[DEĞİŞİYOR — genişliyor]`
- **3.5.1 Geleneksel Metrikler** `[KORUNUYOR]` — RMSE, MAE, R².
- **3.5.2 IEEE PHM 2012 Skoru** `[KORUNUYOR]` — asimetrik ceza.
- **YENİ → 3.5.3 Korelasyon ve Genelleme Metrikleri**:
  - `endpoint_r` (yataklar-arası RUL sıralaması), `traj_r` (yatak-içi HI şekli),
    **LOBO traj_r** (görülmemiş yatağa genelleme).
- **YENİ → 3.5.4 "PHM Tek Başına Neden Yanıltıcı"**:
  - PHM, "her şeye düşük/yüksek de" dejenere çözümlerini ödüllendirebiliyor.
  - **En net örnek: time_progress.** Eklenince PHM **yükseliyor** ama yatak-içi/arası korelasyon
    **negatife** dönüyor (model gerçekte öğrenmiyor, metriği sömürüyor) → bu yüzden kaldırıldı.
  - Bu yüzden her adım PHM **+ korelasyon** ile birlikte ölçüldü. (Bu, dürüstlük metodolojisinin temeli.)

### 3.6 Makine Öğrenmesi ile RUL Tahmini  `[BÜYÜK DEĞİŞİM]`
> **NOT: Sağlıklı-faz filtresi tezden TAMAMEN ÇIKARILDI** (kullanıcı kararı). 3.6.1 filtre bölümü,
> Tablo 3.5 ve filtreye dair tüm cümleler silinecek.
- **3.6.1 Algoritma Seçimi** `[KORUNUYOR — hiperparametreler aynı]`
  - RF (n_estimators=300, min_samples_leaf=5, max_depth=None).
  - XGBoost (lr=0.02, max_depth=6, subsample=0.8, colsample_bytree=0.8, 400 tur sabit).
  - *Not:* "early stopping filtre uyuşmazlığı" paragrafı **silinecek** (filtre artık yok); gerekiyorsa
    sabit 400 tur gerekçesi filtreye atıf yapmadan kısaca verilir.
  - *(Opsiyonel:* LightGBM/SVR 2.5'te anlatıldı ama uygulanmadıysa belirt.)
- **3.6.2 Performans (DÜRÜST)** `[DEĞİŞİYOR — sayılar]`
  - RF: PHM **0.152**, endpoint_r +0.158, traj_r 0.44, RMSE 56.1 dk, MAE 40.5 dk.
  - XGB: PHM **0.186**, endpoint_r −0.068, traj_r 0.47, RMSE 59.8 dk, MAE 44.5 dk.
  - *Yatak-başına tahmin tablosu* (notebook → `tablo_yatak_tahmin.csv`): her test yatağı için gerçek RUL,
    RF/XGB tahmini, hata, yatak-PHM. ML modellerinin nerede battığını gösterir (1_6 aşırı-tahmin,
    1_7 düşük-tahmin) → **RQ1 motivasyonu** (dizi-DL'e geçiş).

### 3.7 Derin Öğrenme ile RUL Tahmini  `[BÜYÜK DEĞİŞİM — SA-TCN çıktı]`
- **3.7.1 Özellik Çıkarımı vs Ham Veri** `[KORUNUYOR — RQ2 zemini]`
- **YENİ → 3.7.2 GRU Mimarisi (öznitelik dizisi, W=24)** — ana DL modeli.
- **YENİ → 3.7.3 TCN Mimarisi (öznitelik dizisi, W=24)** — kıyas.
- **YENİ → 3.7.4 raw-TCN (ham sinyal `(2560,2)`)** — RQ2 ablasyonu, otomatik FE.
- *Ortak:* HI hedefi, LOBO-γ, 5-seed ensemble, fraksiyon okuması.
- **SA-TCN bölümü tamamen silindi.** ("Neden LSTM değil / öz-dikkat" başlıkları da gider.)

---

## BÖLÜM 4 — DENEYSEL SONUÇLAR  `[YENİ İÇERİK — şu an boş]`
*(Tezde 4/5/6 yanlışlıkla Heading 3 → Heading 1/2 yapılacak.)*

### 4.1 Hata Metrikleri ve PHM Skoru — Birleşik Karşılaştırma
**5 model karşılaştırma tablosu** (LOBO-doğrulanmış):

| Model | Girdi | PHM | endpoint_r | traj_r | LOBO traj_r | RMSE(dk) | MAE(dk) |
|---|---|---|---|---|---|---|---|
| Random Forest | tek-pencere öznitelik | 0.152 | +0.158 | 0.44 | — | 56.1 | 40.5 |
| XGBoost | tek-pencere öznitelik | 0.186 | −0.068 | 0.47 | — | 59.8 | 44.5 |
| **GRU** | öznitelik dizisi (W=24) | **0.261** | **+0.676** | 0.34 | **0.584** | 45.1 | 27.2 |
| TCN | öznitelik dizisi (W=24) | 0.202 | +0.673 | 0.28 | 0.325 | 50.9 | 34.7 |
| raw-TCN | ham sinyal (2560,2) | 0.130 | −0.049 | 0.44 | 0.629 | 71.9 | 54.3 |

> **✅ Track B Colab koşusu tamamlandı (DOĞRULANDI).** Sayılar önceki koşularla birebir
> (GRU 0.2606 · TCN 0.2019 · raw-TCN 0.1301). RMSE değerleri yatak-başına gerçek tahminlerden
> hesaplandı (MAE ile tutarlı). raw-TCN v2 "anchored" denemesi iyileştirmedi (0.1418) → 5.2 dipnotu.

- **RQ1 cevabı (dikkatli):** **GRU açık ara en iyi** (0.261 vs XGB 0.186). **TCN (0.202) ise ML ile
  kıyaslanabilir seviyede** — XGB (0.186) ile farkı seed gürültüsü (±0.04) içinde; "TCN > ML" denmez.
  Bulgu, kısa-dizi/az-veride **recurrent (GRU) mimarinin** öne çıktığı yorumuyla tutarlı.
- **endpoint_r dikkat (C):** GRU endpoint_r=0.676, model 6 yatağı da gördüğü test koşusundan → iyimser.
  "Gerçekten öğreniyor" iddiası **endpoint_r'ye değil, LOBO traj_r=0.584'e** dayandırılmalı (genelleme kanıtı).
- ✅ **Track B Colab'da koşuldu ve teyit edildi** (`track_b_results.csv`); 5-model tablosu kesinleşti.

### 4.2 RUL Tahmin Görsel Analizi  `[notebook'ta üretilecek]`
- Trajektori grafikleri (gerçek vs tahmin HI/RUL), yatak-yatak.
- 1_7 / 2_3 mükemmel (düşük titreşim→uzun RUL doğru); 2_4/1_6 aynı kuralla yanlış (ani arıza).

---

## BÖLÜM 5 — TARTIŞMA  `[YENİ İÇERİK — şu an boş]`

### 5.1 Kesik Veri (Truncation) / Fiziksel Limit
- 4 yatak (1_6, 2_4, 2_6, 2_7): kesim anında **%78-81 bozuk** olmalarına rağmen ham titreşim
  bir SAĞLIKLI yatağınki kadar düşük (h_rms≈0.05-0.11, beklenen ~0.30).
- **Bearing2_6 (rms 0.05, 22 dk'da ölür) ≈ Bearing1_7 (rms 0.05, 125 dk yaşar)** → özniteliklerde
  ayırt edilemez. **Hiçbir model ayıramaz** = uyarısız/ani arıza modu, fiziksel sınır.
- Bunları kovalamak = test'e overfit; YAPILMADI (fiziksel sınır, model hatası değil).

### 5.2 Geleneksel Yöntemlerin Direnci (RQ2)
- **RQ2 cevabı:** raw-TCN bozulma **şeklini** en iyi genelliyor (LOBO traj_r 0.63) ama **mutlak RUL'u
  genelleyemiyor** (endpoint_r≈0). Otomatik FE, 6 eğride kalibrasyon için manuel FE'yi yakalayamıyor.
- v2 "HI baseline anchoring" iyileştirmedi → sınırlama temel, offset değil. **raw-TCN v1 raporlanır, v2 dipnot.**
- Doğru FE ile XGBoost gibi ağaç modeller kesik veride DL'den daha stabil kalabiliyor.

---

## BÖLÜM 6 — SONUÇ VE GELECEK ÇALIŞMALAR  `[YENİ İÇERİK]`

### 6.1 Genel Bulgular
- RQ1: dizi-DL (GRU) > tek-pencere ML, dürüst (LOBO) doğrulamayla.
- RQ2: ham otomatik-FE < manuel-FE (mutlak RUL kalibrasyonunda).
- HI metodolojisi + sızıntı temizliği + LOBO ile **dürüst ~0.26 PHM** (GRU).
- Fiziksel limit: bazı yataklar prensipte tahmin edilemez (truncation).

### 6.2 Gelecek Çalışmalar
- Daha fazla run-to-failure yatak (6 yatak genellemeye yetmiyor — LOBO bunu gösterdi).
- Ani-arıza modunu yakalamak için ek sensör/modalite.

---

## BÖLÜM 7 — KAYNAKÇA  `[KORUNUYOR + ekleme]`
- Mevcut ~60 kaynak korunuyor.
- *Ekle:* GRU/TCN, LOBO/cross-validation, PHM challenge kaynakları gerekiyorsa.

---

## GÖRSELLEŞTİRME (notebook: `notebooks/tez_gorseller.ipynb`)
> Yayın kalitesi (300 DPI, Türkçe, gri tonlamada okunur). PNG'ler: `experiments/results/figures/`.
> Generator: `notebooks/_build_notebook.py` (notebook'u yeniden üretir).

| Şekil | Dosya | Tez | Durum |
|---|---|---|---|
| H. HI etiketleme illüstrasyonu (eski Şekil 3.3 yerine) | `gorsel_hi_etiketleme.png` | 3.2.6 | ✅ ÜRETİLDİ |
| A. Yatak-başına bozulma sinyali şekilleri (küçük-çoklu) | `gorsel_sinyal_farkliliklari.png` | 3.1 / 5.1 | ✅ ÜRETİLDİ |
| B. Truncation: h_rms vs gerçek HI (2_6≈1_7 vurgulu) | `gorsel_truncation.png` | 5.1 | ✅ ÜRETİLDİ |
| C. RF/XGB HI trajektorileri (11 yatak) | `gorsel_trajektori_ml.png` | 4.2 | ✅ ÜRETİLDİ |
| D. Yatak-başına tahmin (tablo + bar) | `tablo_yatak_tahmin.csv` / `gorsel_yatak_tahmin.png` | 4.1 | ✅ ÜRETİLDİ |
| E. 5-model PHM/endpoint_r bar | `gorsel_model_karsilastirma.png` | 4.1 | ⏳ DL=placeholder (Track B) |
| F. GRU/TCN/raw-TCN rulman-başına RUL tahmini | `gorsel_dl_rulman_tahmin.png` | 4.2 | ✅ ÜRETİLDİ |
| G. raw-TCN vs feature-DL: LOBO traj_r vs endpoint_r (RQ2) | `gorsel_rq2_rawtcn.png` | 5.2 | ⏳ placeholder (Track B) |

## DOCX'TEKİ MEVCUT ŞEKİL/TABLO TUTARSIZLIKLARI (baştan-sona inceleme)
> Tezde hâlihazırda olan şekiller incelendi. Düzeltilmesi gerekenler:
- **Tablo 3.5 (Sağlıklı Faz Filtreleme)** → **SİL** (filtre tezden çıkarıldı). 3.6.1 metni de silinir.
- **Şekil 3.3 (Parçalı Doğrusal RUL Etiketleri)** → **DEĞİŞTİR**: yerine üretilen HI etiketleme
  illüstrasyonu (`gorsel_hi_etiketleme.png`, Şekil H) kullanılacak.
- **Şekil 3.6 / 3.7 (RF/XGB sonuçları)** → **GÜNCELLE**: eski sayılar (PHM 0.176/**0.2748**, RMSE
  29.7/29.1) filtreli/eski-okuma kurgusundan. Dürüst değerler: PHM 0.152/0.186, RMSE 56.1/59.8.
  Yeni Şekil D (yatak-başına) + Şekil E (model kıyas) bunların yerine geçer.
- **Şekil 3.1 (CUSUM), 3.2 (ön işleme), 3.4 (MI), 3.5 (PHM eğrisi), 2.1/2.2 (ANN/LSTM)** → korunuyor.
- **SA-TCN şekilleri / "öz-dikkat" / "neden LSTM değil"** → SİL (DL bölümü GRU/TCN/raw-TCN oldu).
- **Bölüm 4 / 5 / 6** → şekilsiz; notebook'taki A–G şekilleri buralara yerleşecek.

## AÇIK İŞLER (PROJE_DURUMU §10-11)
- [x] Track B (GRU/TCN/raw-TCN) Colab'da koşuldu → `track_b_results.csv` üretildi, tablo teyit edildi.
- [ ] 4/5/6 başlık seviyeleri Heading 1/2'ye düzeltilecek (docx).
- [ ] Tablo 3.5 ve Şekil 3.3/3.6/3.7'yi yukarıdaki notlara göre güncelle (docx).


güncellik kontrolü