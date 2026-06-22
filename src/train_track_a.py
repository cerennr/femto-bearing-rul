"""
train_track_a.py  (DÜRÜST sürüm)
--------------------------------
Track A: Random Forest + XGBoost, geleneksel ML.
Ortak metodoloji (Track B ile aynı, adil kıyas):
  - Hedef = HI (bozulma fraksiyonu), mutlak RUL değil
  - time_progress YOK, sağlıklı-faz filtresi varsayılan KAPALI (config flag)
  - Final model 6 öğrenme yatağıyla eğitilir
  - Okuma: HI → fraksiyon RUL + LOBO ile seçilen gamma
  - Değerlendirme: PHM + endpoint_r + yatak-içi traj_r + LOBO HI_traj_r + MAE

Çalıştırma:  cd src && python train_track_a.py
"""
import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from config import (FEATURES_TRAIN, FEATURES_TEST, LEARNING_BEARINGS,
                    RESULTS_DIR, MODELS_DIR, USE_HEALTHY_FILTER,
                    DEGRAD_THRESH_FRAC, RUL_CAP_MIN)
from preprocessing import Preprocessor
from models import RandomForestModel, XGBoostModel
from hi_labeling import (add_hi_label, actual_rul_min, rul_from_hi, smooth_hi,
                         phm_from_pairs, endpoint_corr, traj_corr,
                         lobo_gamma_records, pick_best_gamma)


def fit_prep_model(train_df, model_factory, n_features=25):
    """train_df üzerinde prep(HI hedefli) + model fit eder."""
    prep = Preprocessor(n_features=n_features, use_pca=False, target_col="hi")
    X, y = prep.fit_transform(train_df.copy())
    model = model_factory()
    model.fit(X, y)
    return prep, model

def predict_hi_bearing(prep, model, bdf):
    """Bir yatağın tüm pencereleri için tahmin HI (sıralı)."""
    bdf = bdf.sort_values("time_s")
    X, _ = prep.transform(bdf.copy())
    return np.clip(model.predict(X), 0, 1)


def main():
    print("\n" + "=" * 60 + "\n  TRACK A — Dürüst (HI + fraksiyon + LOBO-gamma)\n" + "=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True); MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tr = add_hi_label(pd.read_csv(FEATURES_TRAIN))
    te = add_hi_label(pd.read_csv(FEATURES_TEST))
    if USE_HEALTHY_FILTER:
        thr = RUL_CAP_MIN * DEGRAD_THRESH_FRAC
        keep = []
        for b, g in tr.groupby("bearing"):
            total = (g["time_s"] + g["rul_s"]).max()
            rmin = np.clip((total - g["time_s"]) / 60.0, 0, RUL_CAP_MIN)
            keep.append(g[rmin < thr])
        tr = pd.concat(keep)
        print(f"  [Filtre] açık → {len(tr)} pencere")

    factories = {
        "Random Forest": lambda: RandomForestModel(n_estimators=300, min_samples_leaf=5),
        "XGBoost":       lambda: XGBoostModel(n_estimators=400, learning_rate=0.02, max_depth=6),
    }

    rows = []
    for name, factory in factories.items():
        print(f"\n  >>> {name}  (LOBO gamma seçimi...)")
        recs = []
        for bout in LEARNING_BEARINGS:
            prep_f, mdl_f = fit_prep_model(tr[tr.bearing != bout], factory)
            g = tr[tr.bearing == bout].sort_values("time_s")
            hi_pred = predict_hi_bearing(prep_f, mdl_f, g)
            recs.append(lobo_gamma_records(hi_pred, g["time_s"].values,
                                           g["rul_s"].values, g["t_star_s"].iloc[0]))
        gstar = pick_best_gamma(recs)

        prep, model = fit_prep_model(tr, factory)
        model.save(MODELS_DIR / f"{name.replace(' ','_')}.pkl")

        acts, preds, tjs = [], [], []
        for b, g in te.groupby("bearing"):
            g = g.sort_values("time_s")
            hi_pred = predict_hi_bearing(prep, model, g)
            preds.append(rul_from_hi(hi_pred, g["time_s"].values, g["t_star_s"].iloc[0], gstar))
            acts.append(actual_rul_min(g))
            tjs.append(traj_corr(g["hi"].values, smooth_hi(hi_pred, gstar)))
        acts, preds = np.array(acts), np.array(preds)
        phm = phm_from_pairs(acts, preds); r = endpoint_corr(acts, preds)
        mae = float(np.mean(np.abs(acts - preds)))
        rmse = float(np.sqrt(np.mean((acts - preds) ** 2))); tjr = float(np.nanmean(tjs))
        print(f"      gamma*={gstar}  PHM={phm:.4f}  endpoint_r={r:+.3f}  RMSE={rmse:.1f}  MAE={mae:.1f}  traj_r={tjr:+.3f}")
        rows.append(dict(model=name, gamma=gstar, phm=round(phm, 4), endpoint_r=round(r, 3),
                         rmse=round(rmse, 1), mae=round(mae, 1), traj_r=round(tjr, 3)))

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "track_a_results.csv", index=False)
    print("\n" + "=" * 60 + "\n  TRACK A SONUÇLARI\n" + "=" * 60)
    print(res.to_string(index=False))
    print(f"\n  Kaydedildi: {RESULTS_DIR / 'track_a_results.csv'}")


if __name__ == "__main__":
    main()
