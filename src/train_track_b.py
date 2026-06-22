"""
train_track_b.py  (Colab + GPU)
-------------------------------
Track B: GRU + TCN (öznitelik dizisi) + raw-TCN (ham sinyal). Hedef = HI.
Track A ile AYNI değerlendirme: PHM + endpoint_r + traj_r + LOBO HI_traj_r.

Akış (her model):
  1. LOBO (6 fold, 1 seed) ile gamma* seç + LOBO HI_traj_r ölç
  2. 6 yatakla 5-seed ensemble eğit
  3. Test: HI tahmin → fraksiyon RUL (gamma*) → metrikler

Çalıştırma (Colab): src'yi path'e ekle, GPU runtime, python train_track_b.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
import tensorflow as tf

# Tekrarlanabilirlik: asıl stabilizatör 15-seed ensemble (config.SEEDS) + per-model
# set_random_seed. TAM op-determinizmi isteğe bağlı — aşağıyı açabilirsin AMA bazı
# cuDNN GPU ops'larında (özellikle GRU) eğitim sırasında HATA verebilir, dikkat:
os.environ.setdefault("PYTHONHASHSEED", "0")
# os.environ["TF_DETERMINISTIC_OPS"] = "1"; tf.config.experimental.enable_op_determinism()

from config import (FEATURES_TRAIN, FEATURES_TEST, LEARNING_BEARINGS,
                    VAL_BEARINGS, RESULTS_DIR, SEQ_LEN, SEEDS, RAW_SEEDS, EPOCHS, BATCH_SIZE)
from preprocessing import Preprocessor
from dl_models import build_gru, build_tcn, build_raw_tcn
from sequence import build_feature_dataset, make_feature_sequences, load_raw_bearing
from hi_labeling import (add_hi_label, actual_rul_min, rul_from_hi, smooth_hi,
                         phm_from_pairs, endpoint_corr, traj_corr,
                         lobo_gamma_records, pick_best_gamma)


def fit(build_fn, Xtr, ytr, Xv, yv, seed, epochs):
    tf.keras.utils.set_random_seed(seed)
    m = build_fn()
    cb = [tf.keras.callbacks.EarlyStopping("val_loss", patience=12, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=6)]
    m.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=epochs,
          batch_size=BATCH_SIZE, callbacks=cb, verbose=0)
    return m

def ensemble_predict(models, X):
    return np.clip(np.mean([m.predict(X, batch_size=256, verbose=0).ravel() for m in models], axis=0), 0, 1)


# ── Öznitelik-dizisi modelleri (GRU / TCN) ────────────────────────────────────

def run_feature_model(name, build_fn, tr, te):
    print(f"\n{'='*60}\n  {name} (öznitelik dizisi)\n{'='*60}")
    # LOBO: gamma + HI_traj_r
    recs, lobo_tr = [], []
    for bout in LEARNING_BEARINGS:
        sub = tr[tr.bearing != bout]
        prep = Preprocessor(n_features=25, use_pca=False, target_col="hi")
        Xtr, ytr, _ = build_feature_dataset_with_fit(prep, sub)
        vb = VAL_BEARINGS[0] if bout != VAL_BEARINGS[0] else VAL_BEARINGS[1]
        Xv, yv, _ = _seq_for(prep, sub[sub.bearing == vb])
        F = Xtr.shape[-1]
        m = fit(lambda: build_fn(F), Xtr, ytr, Xv, yv, seed=0, epochs=max(45, EPOCHS // 2))
        g = tr[tr.bearing == bout].sort_values("time_s")
        Xb, _, _ = _seq_for(prep, g)
        hi = ensemble_predict([m], Xb)
        lobo_tr.append(traj_corr(g["hi"].values, hi))
        recs.append(lobo_gamma_records(hi, g["time_s"].values, g["rul_s"].values, g["t_star_s"].iloc[0]))
    gstar = pick_best_gamma(recs)
    print(f"  LOBO HI_traj_r={np.nanmean(lobo_tr):+.3f}  gamma*={gstar}")

    # Final: 6 bearing, 5-seed
    prep = Preprocessor(n_features=25, use_pca=False, target_col="hi")
    Xtr, ytr, _ = build_feature_dataset_with_fit(prep, tr)
    Xv, yv, _ = _seq_for(prep, tr[tr.bearing == VAL_BEARINGS[-1]])
    F = Xtr.shape[-1]
    models = [fit(lambda: build_fn(F), Xtr, ytr, Xv, yv, s, EPOCHS) for s in SEEDS]
    return _eval(name, gstar, np.nanmean(lobo_tr),
                 lambda g: ensemble_predict(models, _seq_for(prep, g)[0]), te)

def build_feature_dataset_with_fit(prep, df):
    prep.fit_transform(df.copy())          # prep'i fit et (seçim+ölçek)
    return build_feature_dataset(prep, df) # sonra dizilere çevir

def _seq_for(prep, g):
    g = g.sort_values("time_s")
    Xf, _ = prep.transform(g.copy())
    return make_feature_sequences(Xf), g["hi"].values.astype(np.float32), g


# ── Ham sinyal modeli (raw-TCN) ───────────────────────────────────────────────

def run_raw_model(tr, te, meta):
    print(f"\n{'='*60}\n  raw-TCN (ham sinyal)\n{'='*60}")
    TR = {b: load_raw_bearing(b, False, meta) for b in LEARNING_BEARINGS}
    recs, lobo_tr = [], []
    for bout in LEARNING_BEARINGS:
        ins = [b for b in LEARNING_BEARINGS if b != bout]
        Xtr = np.concatenate([TR[b][0] for b in ins]); ytr = np.concatenate([TR[b][1] for b in ins])
        vb = ins[-1]; Xv, yv = TR[vb][0], TR[vb][1]
        m = fit(build_raw_tcn, Xtr, ytr, Xv, yv, seed=0, epochs=45)
        Xb, hi_true, times, ts, _ = TR[bout]
        hi = ensemble_predict([m], Xb)
        lobo_tr.append(traj_corr(hi_true, hi))
        recs.append(lobo_gamma_records(hi, times, _rul_s_from(TR[bout]), ts))
    gstar = pick_best_gamma(recs)
    print(f"  LOBO HI_traj_r={np.nanmean(lobo_tr):+.3f}  gamma*={gstar}")

    Xtr = np.concatenate([TR[b][0] for b in LEARNING_BEARINGS]); ytr = np.concatenate([TR[b][1] for b in LEARNING_BEARINGS])
    Xv, yv = TR[VAL_BEARINGS[-1]][0], TR[VAL_BEARINGS[-1]][1]
    models = [fit(build_raw_tcn, Xtr, ytr, Xv, yv, s, EPOCHS) for s in RAW_SEEDS]
    TE = {b: load_raw_bearing(b, True, meta) for b in te.bearing.unique()}
    acts, preds, tjs = [], [], []
    for b in te.bearing.unique():
        Xb, hi_true, times, ts, total = TE[b]
        hi = ensemble_predict(models, Xb)
        preds.append(rul_from_hi(hi, times, ts, gstar)); acts.append(min(total/60 - times[-1]/60, 125.0))
        tjs.append(traj_corr(hi_true, smooth_hi(hi, gstar)))
    return _summary("raw-TCN", gstar, np.nanmean(lobo_tr), acts, preds, tjs)

def _rul_s_from(tr_tuple):
    """raw eğitim yatağı için rul_s = total_life - times (run-to-failure)."""
    _, _, times, ts, total = tr_tuple
    return total - times


# ── Ortak değerlendirme ───────────────────────────────────────────────────────

def _eval(name, gstar, lobo_tr, predict_hi_fn, te):
    acts, preds, tjs = [], [], []
    for b, g in te.groupby("bearing"):
        g = g.sort_values("time_s")
        hi = predict_hi_fn(g)
        preds.append(rul_from_hi(hi, g["time_s"].values, g["t_star_s"].iloc[0], gstar))
        acts.append(actual_rul_min(g)); tjs.append(traj_corr(g["hi"].values, smooth_hi(hi, gstar)))
    return _summary(name, gstar, lobo_tr, acts, preds, tjs)

def _summary(name, gstar, lobo_tr, acts, preds, tjs):
    acts, preds = np.array(acts), np.array(preds)
    row = dict(model=name, gamma=gstar, phm=round(phm_from_pairs(acts, preds), 4),
               endpoint_r=round(endpoint_corr(acts, preds), 3),
               rmse=round(float(np.sqrt(np.mean((acts - preds) ** 2))), 1),
               mae=round(float(np.mean(np.abs(acts - preds))), 1),
               traj_r=round(float(np.nanmean(tjs)), 3),
               lobo_traj_r=round(float(lobo_tr), 3))
    print(f"  >>> {name}: PHM={row['phm']}  endpoint_r={row['endpoint_r']}  "
          f"RMSE={row['rmse']}  MAE={row['mae']}  traj_r={row['traj_r']}  LOBO_traj_r={row['lobo_traj_r']}")
    return row


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tr = add_hi_label(pd.read_csv(FEATURES_TRAIN))
    te = add_hi_label(pd.read_csv(FEATURES_TEST))
    meta = pd.concat([tr, te])[["bearing", "window_idx", "time_s", "rul_s", "t_star_s", "cap_s"]]
    rows = []
    rows.append(run_feature_model("GRU", build_gru, tr, te))
    rows.append(run_feature_model("TCN", build_tcn, tr, te))
    rows.append(run_raw_model(tr, te, meta))
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "track_b_results.csv", index=False)
    print("\n" + "=" * 60 + "\n  TRACK B SONUÇLARI\n" + "=" * 60)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
