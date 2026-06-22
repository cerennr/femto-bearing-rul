# FEMTO RUL — Proje Durumu ve Devir (Handoff)

> Bu dosya, bir önceki oturumda yapılan **tüm çalışmayı** kapsar. Yeni agent buradan
> kaldığı yerden devam edebilir. Hedef: tezin 3.6 sonrası (modelleme/sonuçlar) bölümünü
> bu bulgulara göre yeniden yazmak. **Tez metni düzenlemesi henüz YAPILMADI.**

---

## 1. Proje ve Hedef

- **Veri seti**: IEEE PHM 2012 (FEMTO-PRONOSTIA) rulman RUL. 6 eğitim (run-to-failure),
  11 test (truncated) yatağı; 3 çalışma koşulu. Ham titreşim `(2560, 2)` pencereleri
  (25.6 kHz, 0.1 s, her 10 s'de bir; yatay+dikey ivmeölçer).
- **Tez**: Ceren Üner, danışman Prof. Dr. H. Kıvanç Aksoy. Geleneksel ML vs derin öğrenme.
- **Metrik**: IEEE PHM 2012 skoru (asimetrik; geç tahmin sert, erken tahmin hafif cezalı),
  11 test yatağının ortalaması. **Hedef ~0.30** (≈ challenge kazananı seviyesi).

## 2. Başlangıç Durumu ve Teşhis

Notebook'taki SA-TCN (ham sinyal) **PHM ≈ 0.19**'da takılıydı. Denenen 3 DL mimarisi
(SA-TCN, CNN-LSTM+STFT, full TCN) hepsi ~0.19 → **sorun mimari değil, kurgu.**

**Kök sorunlar (deneyle kanıtlandı):**
1. **Mutlak RUL'u tek anlık pencereden** kestirmek ill-posed. Çözüm: HI (bozulma durumu) tahmin → RUL'a çevir.
2. **Etiket doygunluğu**: piecewise-RUL'de ömrün %50-73'ü tavanda (`y=1`) → model "her şeye yüksek RUL" → kısa yataklarda felaket fazla-tahmin.
3. **PHM tek başına yanıltıcı**: dejenere "her şeye düşük/yüksek de" çözümlerini ödüllendiriyor. → her adımda **PHM + korelasyon** ölçtük.

## 3. Veri Sızıntıları (çözüldü)

- **`deg_progress = (time_s − t_star_s)/cap_s`**: hedefi neredeyse birebir kopyalıyordu
  (t_star/cap test'te bilinmez). Sahte ~%100 başarı. → `META_COLS`'a eklenip dışlandı.
- **`time_progress = time_s/max(time_s)`**: test'te son pencerede **daima 1.0** (dejenere).
  PHM'i şişirir ama yatak-içi korelasyonu **negatife** çevirir. → `USE_TIME_PROGRESS=False`.
- **`'hi'` hedef sütunu** (reorganizasyon sırasında): `META_COLS`'a eklenmemişti, model
  hedefi öznitelik olarak gördü (endpoint_r=0.998 sahte). → düzeltildi.

## 4. Mevcut Yöntem (dürüst, sızıntısız)

- **Hedef = HI** (bozulma fraksiyonu): `HI = clip((time_s − t_star_s)/cap_s, 0, 1)`,
  sağlıklı=0, arıza=1. CUSUM-FPT (`t_star`) ve `cap` data_utils.py'de hesaplanıyor (DEĞİŞMEDİ).
- **HI → RUL** (fraksiyon, sızıntısız): `RUL = (time_last − t_star)·(1−HI_last)/HI_last`.
  `t_star` test'te nedensel bilinir; `cap` KULLANILMAZ. HI trajektorisi EMA ile yumuşatılır,
  `HI^gamma` ile kalibre edilir; **gamma LOBO ile seçilir** (test'e bakılmaz).
- **Eğitim**: 6 yatağın hepsi (val sadece model seçimi). Sağlıklı-faz filtresi **kapalı** (flag).
- **Değerlendirme**: PHM (ana) + endpoint_r (yataklar-arası RUL sıralaması) + traj_r
  (yatak-içi HI şekli) + LOBO HI_traj_r (genelleme) + RMSE/MAE.

## 5. SONUÇLAR (dürüst, LOBO-doğrulanmış)

> Track A bu oturumda **lokalde çalıştırıldı** (anaconda python). Track B değerleri
> önceki doğrulanmış **Colab** koşularından; yeni `train_track_b.py` Colab'da tekrar
> koşulunca aynısını vermeli (henüz koşulmadı).

| Model | Girdi | PHM | endpoint_r | traj_r | LOBO traj_r | RMSE (dk) | MAE (dk) |
|---|---|---|---|---|---|---|---|
| Random Forest | tek-pencere öznitelik | 0.152 | +0.158 | 0.44 | — | 56.1 | 40.5 |
| XGBoost | tek-pencere öznitelik | 0.186 | −0.068 | 0.47 | — | 59.8 | 44.5 |
| **GRU** | öznitelik dizisi (W=24) | **0.261** | **+0.676** | 0.34 | **0.584** | ~27 | 27.2 |
| TCN | öznitelik dizisi (W=24) | 0.202 | +0.673 | 0.28 | 0.325 | ~35 | 34.7 |
| raw-TCN | ham sinyal (2560,2) | 0.130 | −0.049 | 0.44 | 0.629 | ~54 | 54.3 |
| **CALCE (2012 kazanan)** | — | **~0.31** | — | — | — | — | — |

**Notlar:**
- **GRU en iyi** ve DÜRÜST (LOBO traj_r=0.58 > diğerleri → test'teki yüksek korelasyon şans değil).
- **raw-TCN**: bozulma ŞEKLİNİ en iyi genelliyor (LOBO traj_r=0.63) ama **mutlak RUL'u
  genelleyemiyor** (endpoint_r≈0) → **RQ2 bulgusu**: otomatik FE, 6 eğride kalibrasyon
  için manuel FE'yi yakalayamıyor. v2'de denenen "HI baseline anchoring" iyileştirmedi
  (→ sınırlama offset değil, temel). **raw-TCN v1 raporlanır, v2 tartışma dipnotu.**
- Track A'nın eski "**0.11**" değeri farklı bir kurguydu (doğrudan-RUL + tek son pencere).
  Şimdiki 0.152/0.186 **HI metodolojisiyle** (Track B ile adil kıyas). İkisi de dürüst.

## 6. CALCE Karşılaştırması (challenge kazananı, Sutrisno 2012)

Makale PHM skoru raporlamamış; Tablo II yatak-yatak hata %'si veriyor. Standart PHM
formülüyle hesap → **≈0.31**. Sadece **4/11** yatakta %10 hata altı.

**Kritik**: BİZ ve CALCE **farklı** yataklarda batıyoruz (tamamlayıcı):
- BİZ iyi / CALCE kötü: 1_7, 2_3 (uzun), 2_5 (−440% CALCE), 2_7 (−317% CALCE), 3_3.
- CALCE iyi / BİZ kötü: 1_6 (−5% CALCE vs −410% biz), 2_4 (+10% vs −437%), 2_6 (+49% vs −110%), 1_5, 1_3.
- **Ortak gerçek**: her iki yöntem de bazı ani-arıza yataklarında felaket fazla-tahmin yapıyor.

## 7. FİZİKSEL LİMİT (tezin 5.1 Truncation kanıtı)

4 yatak (1_6, 2_4, 2_6, 2_7) kesim anında **%78-81 bozulmuş** olmalarına rağmen ham
titreşim genliği SAĞLIKLI bir yatağınki kadar düşük (`h_rms≈0.05-0.11`, beklenen ~0.30).
**Bearing2_6 (rms 0.05, 22 dk'da ölüyor) ile Bearing1_7 (rms 0.05, 125 dk yaşıyor)
özniteliklerde birebir aynı** → hiçbir model ayıramaz. Uuyarısız/ani arıza modu. Bunları
kovalamak = test'e overfit; YAPILMADI. Challenge kazananı da bu sınıra çarpıyor.

## 8. Kod Yapısı (yeni, `src/` altında)

```
femto_rul/
├── src/
│   ├── config.py        Sabitler, yollar, FLAG'ler (USE_TIME_PROGRESS=False,
│   │                    USE_HEALTHY_FILTER=False), SEQ_LEN=24, SEEDS, GAMMA_GRID,
│   │                    META_COLS (hi/deg_progress/time_progress/t_star/cap dışlı)
│   ├── data_utils.py    FE çekirdeği: CUSUM-FPT (koşula özel eşik), wavelet db4,
│   │                    54 öznitelik (zaman16+frekans12+STFT24+çapraz2), RUL etiketi
│   │                    → features_{train,test}.csv üretir. **DEĞİŞTİRME.**
│   ├── preprocessing.py EMA(span20)+baseline+trend(diff5), MI top-25, RobustScaler.
│   │                    time_progress artık USE_TIME_PROGRESS flag'ine bağlı.
│   ├── hi_labeling.py   add_hi_label, rul_from_hi (fraksiyon), smooth_hi (HI^γ+EMA),
│   │                    lobo_gamma_records, pick_best_gamma, phm/endpoint_r/traj_r.
│   ├── sequence.py      make_feature_sequences (N,W,F); load_raw_bearing
│   │                    (.npz→baseline_standardize→HI hizalama).
│   ├── models.py        RandomForestModel, XGBoostModel (SVR yorumlu).
│   ├── dl_models.py     build_gru, build_tcn (padding='same'), build_raw_tcn (Keras).
│   │                    Doğrulanmış Colab mimarileriyle BİREBİR.
│   ├── evaluation.py    rmse/mae/r2, phm_accuracy/phm_score, grafikler.
│   ├── train_track_a.py Track A: RF+XGB, HI+fraksiyon+LOBO-γ, 6-bearing. (lokal)
│   └── train_track_b.py Track B: GRU+TCN+raw-TCN, LOBO-γ+5seed ensemble. (Colab+GPU)
├── data/{raw, numpy_data, processed}     experiments/{models, results}
├── archive/   (eski exp_*.py + dl_*_colab.py + data_generator.py — referans)
├── notebooks/son.ipynb    README.md    requirements.txt    PROJE_DURUMU.md (bu dosya)
```

## 9. Eski Tez (3.2–3.6) → Yeni Değişiklikler

**KORUNAN**: CUSUM-FPT (3.2.1), wavelet (3.2.2), 54 öznitelik (3.2.3), EMA+baseline+trend
(3.2.4-5), MI top-25 (3.4), PHM skoru (3.5.2), RF+XGBoost (3.6.2).

**DEĞİŞEN**:
- Hedef (3.2.6): piecewise-linear **RUL** → **HI** (bozulma fraksiyonu). CUSUM/cap aynı, biçim değişti.
- Okuma: tek son pencere → HI→RUL fraksiyon + LOBO-γ.
- Train: 3-bearing → 6-bearing final.
- Değerlendirme: yalnız PHM → PHM + endpoint_r + traj_r + LOBO.

**KALDIRILAN**: sağlıklı-faz filtresi (varsayılan kapalı, flag); time_progress (dejenere).

**EKLENEN**: Track B / DL (3.7): GRU + TCN (öznitelik dizisi) + raw-TCN (ham sinyal).

## 10. Doğrulanmış vs Bekleyen

- ✅ Track A lokalde koşuyor (RF 0.152, XGB 0.186). Sızıntı yakalandı/düzeltildi.
- ✅ Tüm modüller derleniyor; sequence.py gerçek veriyle test edildi.
- ⏳ **Track B yeni `train_track_b.py` Colab'da HENÜZ koşulmadı.** Önceki standalone
  Colab koşuları GRU 0.26 / TCN 0.20 / raw-TCN 0.13 verdi; yeni orkestratör aynısını
  vermeli ama **ilk koşuda runtime hatası çıkabilir** (orkestratör karmaşık) — kontrol et.

## 11. Sonraki Adımlar

1. **Track B'yi Colab'da koş** (`src/` + `data/` yükle, GPU). Çıktı: `track_b_results.csv`.
   Hata çıkarsa düzelt. GRU/TCN/raw-TCN sonuçlarını tabloya yaz.
2. **5-model + CALCE birleşik karşılaştırma tablosu** üret (PHM/endpoint_r/RMSE/MAE).
3. **Tez 3.6+ yeniden yazımı**: bölüm 9'daki değişiklikleri tez metnine dönüştür; RQ1
   (öznitelikte dizi-DL > tek-pencere ML), RQ2 (ham otomatik-FE < manuel-FE), 5.1
   (truncation/fiziksel limit), CALCE kıyası. **Bu chat'te YAPILMAYACAK** (yeni oturum).

## 12. Teknik Notlar

- **İki python**: `/usr/bin/python3` (3.9, xgboost YOK) vs **`/opt/anaconda3/bin/python`**
  (3.12, hepsi var). Track A için: `cd src && /opt/anaconda3/bin/python train_track_a.py`.
- **Track A flag'leri** (config.py): `USE_TIME_PROGRESS`, `USE_HEALTHY_FILTER` — tez
  ablasyonu için açılabilir (ör. sağlıklı-faz filtresinin etkisini göstermek).
- **Track A 0.11 sürümü** istenirse: doğrudan-RUL + tek son pencere okumayı flag olarak
  eklemek gerekir (şu an HI metodolojisi standart).
- Gerçek RUL değerleri `config.ACTUAL_RUL_SECONDS`'ta (test yataklarının cap'i bundan).
- Deney detayları/ara bulgular: bu projenin agent hafızasında (`femto-rul-diagnosis.md`).
