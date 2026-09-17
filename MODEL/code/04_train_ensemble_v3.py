"""
04_train_ensemble_v3.py
=============================================================================
SIH26082 - Stage 2b: maximum-accuracy ensemble (v3).

WHY V3 EXISTS
-------------
v2 reached R2 = 0.9634 (PM2.5 h1), beat the certified legacy model decisively at
h=1 and h=3, and hit parity at h=24/48 - but trailed by 1.2-2.4 points at h=6 and
h=12. Two causes were identified, and both are treated here:

  (1) UNDER-CAPACITY. v2 used learning_rate 0.045 / 1200 rounds; the certified
      legacy configuration was 0.03 / 2500 / 127 leaves. Short horizons saturate
      early so they showed no cost, while mid horizons - where the conditional
      signal is subtler - were still improving when training stopped. v3 restores
      the proven capacity.

  (2) NO VALIDATION DATA IN THE FIT. v2 trained on 2023-2024 and used 2025-H1
      purely for early stopping, discarding 43,440 rows - a quarter of the data,
      including a full winter. v3 SELECTS hyperparameters and stopping rounds on
      validation, then REFITS on train+validation with the round count scaled by
      the data ratio. Validation is still never used for evaluation, so this is
      legitimate and typically worth more than any hyperparameter change.

ENSEMBLE DIVERSITY
------------------
Members blended by the NNLS convex simplex (sum w = 1, w >= 0):
    LightGBM  regression_l1   - threshold/spike specialist
    XGBoost   reg:absoluteerror on CUDA (hist) - different split/regularisation
    Ridge     on median-imputed, standardised log-target - stable low-variance
Backed by: xgboost 2.1.3 CUDA is available (2x faster than CPU here); the pip
LightGBM 4.7.0 wheel is CPU-only, so GPU tree boosting is not an option for it.

REFIT DISCIPLINE
----------------
    fit on TRAIN, early-stop on VAL         -> best_iteration, VAL predictions
    NNLS weights + linear bias from VAL only -> never from TEST
    refit on TRAIN+VAL with scaled rounds    -> TEST predictions
    metrics on TEST                          -> touched exactly once

Run:  python 04_train_ensemble_v3.py
=============================================================================
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reuse Stage 2 helpers verbatim so the two stages cannot drift apart.
_spec = importlib.util.spec_from_file_location(
    "s2", os.path.join(HERE, "02_train_coupled_ensemble.py"))
s2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s2)

import coupled_physics as cp  # noqa: E402

ART_DIR = os.path.join(HERE, "artifacts_v3")
MODELS_DIR = os.path.join(ART_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

HORIZONS = s2.HORIZONS
POLLUTANTS = s2.POLLUTANTS
CPCB_LABEL = s2.CPCB_LABEL
TARGET_COL = s2.TARGET_COL
CLIM_Y = s2.CLIM_Y

# Restored to the capacity that produced the certified v1.0.0 results.
LGB_L1 = dict(objective="regression_l1", metric="rmse", boosting_type="gbdt",
              num_leaves=127, learning_rate=0.03, n_estimators=2200,
              min_child_samples=30, feature_fraction=0.6, bagging_fraction=0.8,
              bagging_freq=5, reg_alpha=0.1, reg_lambda=1.0,
              n_jobs=-1, verbose=-1, random_state=42, force_row_wise=True)

XGB_PARAMS = dict(objective="reg:absoluteerror", tree_method="hist",
                  max_depth=8, learning_rate=0.04, subsample=0.8,
                  colsample_bytree=0.7, min_child_weight=6, reg_lambda=1.5,
                  reg_alpha=0.1, verbosity=0, seed=42)

REFIT_GROWTH = 1.30          # rounds scaled by ~(1 + n_val/n_train)


def _detect_xgb_device():
    """Probe for a working CUDA device; fall back to CPU hist if unavailable."""
    try:
        Xp = np.random.rand(512, 8).astype(np.float32)
        xgb.train({**XGB_PARAMS, "device": "cuda"},
                  xgb.DMatrix(Xp, label=np.random.rand(512)),
                  num_boost_round=2)
        return "cuda"
    except Exception:
        return "cpu"


XGB_DEVICE = _detect_xgb_device()


def fit_lgb(Xtr, ytr, Xva, yva, wtr, params, rounds):
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr, free_raw_data=False)
    return lgb.train(dict(params), dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(60, verbose=False),
                                lgb.log_evaluation(0)])


def fit_xgb(Xtr, ytr, Xva, yva, wtr, rounds):
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    dva = xgb.DMatrix(Xva, label=yva)
    return xgb.train({**XGB_PARAMS, "device": XGB_DEVICE}, dtr,
                     num_boost_round=rounds, evals=[(dva, "val")],
                     early_stopping_rounds=60, verbose_eval=False)


def fit_ridge(Xtr, ytr, wtr):
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("mdl", Ridge(alpha=1.0))])
    pipe.fit(Xtr, ytr, mdl__sample_weight=wtr)
    return pipe


def train_one_v3(df, masks, poll, horizon, static_cols, drivers, y_tables):
    y_raw = df["target_%s_h%d" % (poll, horizon)].to_numpy(dtype=float)
    y_log = np.log1p(y_raw)
    w_all = s2.sample_weights(df, poll, masks)

    X = s2.build_design(df, horizon, static_cols, drivers, y_tables, poll)
    feat_names = list(X.columns)

    tr, va, te = masks["train"], masks["val"], masks["test"]
    ok_tr = tr & np.isfinite(y_log)
    ok_va = va & np.isfinite(y_log)
    ok_te = te & np.isfinite(y_log)

    Xtr = X.loc[ok_tr].to_numpy(dtype=np.float32)
    Xva = X.loc[ok_va].to_numpy(dtype=np.float32)
    Xte = X.loc[ok_te].to_numpy(dtype=np.float32)
    ytr, yva = y_log[ok_tr], y_log[ok_va]
    wtr = w_all[ok_tr]

    # ---- 1. SELECT: fit on train, early-stop on val ------------------------
    lgb_m = fit_lgb(Xtr, ytr, Xva, yva, wtr, LGB_L1, LGB_L1["n_estimators"])
    lgb_best = int(lgb_m.best_iteration or LGB_L1["n_estimators"])

    xgb_m = fit_xgb(Xtr, ytr, Xva, yva, wtr, 2200)
    xgb_best = int(getattr(xgb_m, "best_iteration", 2199) or 2199) + 1

    ridge_m = fit_ridge(Xtr, ytr, wtr)

    P_va = np.column_stack([
        np.expm1(np.clip(lgb_m.predict(Xva, num_iteration=lgb_best), 0, None)),
        np.expm1(np.clip(xgb_m.predict(xgb.DMatrix(Xva),
                                       iteration_range=(0, xgb_best)), 0, None)),
        np.expm1(np.clip(ridge_m.predict(Xva), 0, None)),
    ])
    y_va_phys = y_raw[ok_va]
    weights = s2.simplex_nnls(P_va, y_va_phys)

    va_ens = P_va @ weights
    denom = float(np.var(va_ens))
    a = float(np.clip(np.cov(va_ens, y_va_phys)[0, 1] / denom, 0.5, 1.5)) \
        if denom > 1e-9 else 1.0
    b = float(y_va_phys.mean() - a * va_ens.mean())

    # ---- 2. REFIT on train+val with the rounds the validation chose --------
    ok_fit = (tr | va) & np.isfinite(y_log)
    Xfit = X.loc[ok_fit].to_numpy(dtype=np.float32)
    yfit, wfit = y_log[ok_fit], w_all[ok_fit]

    lgb_r = fit_lgb(Xfit, yfit, Xva, yva, wfit, LGB_L1,
                    int(lgb_best * REFIT_GROWTH) + 50)
    xgb_r = fit_xgb(Xfit, yfit, Xva, yva, wfit, int(xgb_best * REFIT_GROWTH) + 50)
    ridge_r = fit_ridge(Xfit, yfit, wfit)

    # Honour the refit's own early stopping, not every boosted round.
    xgb_refit_iters = int(getattr(xgb_r, "best_iteration", None)
                          or (xgb_r.num_boosted_rounds() - 1)) + 1
    lgb_refit_iters = int(lgb_r.best_iteration or lgb_r.num_trees())

    P_te = np.column_stack([
        np.expm1(np.clip(lgb_r.predict(Xte, num_iteration=lgb_refit_iters), 0, None)),
        np.expm1(np.clip(xgb_r.predict(xgb.DMatrix(Xte),
                                       iteration_range=(0, xgb_refit_iters)), 0, None)),
        np.expm1(np.clip(ridge_r.predict(Xte), 0, None)),
    ])
    y_te = y_raw[ok_te]
    persist = df.loc[ok_te, TARGET_COL[poll]].to_numpy(dtype=float)

    ens_raw = P_te @ weights
    ens_cal = np.clip(a * ens_raw + b, 0.0, None)

    out = {
        "pollutant": poll, "horizon": horizon,
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "members": ["lgbm_l1", "xgb_l1", "ridge"],
        "nnls_weights": [float(w) for w in weights],
        "bias_a": a, "bias_b": b,
        "val_r2": s2.compute_metrics(y_va_phys, np.clip(a * va_ens + b, 0, None)).get("r2"),
        "test": s2.compute_metrics(y_te, ens_cal),
        "test_uncorrected": s2.compute_metrics(y_te, ens_raw),
        "persistence": s2.compute_metrics(y_te, persist),
        "lgb_best_iteration": lgb_best,
        "xgb_best_iteration": xgb_best,
        "lgb_refit_iteration": lgb_refit_iters,
        "xgb_refit_iteration": xgb_refit_iters,
        "xgb_device": XGB_DEVICE,
        "refit_rows": int(ok_fit.sum()),
    }
    out["test"]["skill_vs_persistence"] = (
        1.0 - out["test"]["rmse"] ** 2 / out["persistence"]["rmse"] ** 2
        if out["persistence"].get("rmse") else float("nan"))
    out["legacy_r2"] = s2.LEGACY_REF.get(poll, {}).get(horizon)
    out["delta_r2_vs_legacy"] = (out["test"]["r2"] - out["legacy_r2"]
                                 if out["legacy_r2"] is not None else None)

    with open(os.path.join(MODELS_DIR, "%s_h%d.pkl" % (poll, horizon)), "wb") as fh:
        pickle.dump({
            "pollutant": poll, "horizon": horizon,
            "feature_names": feat_names,
            "members": out["members"],
            "models": [lgb_r, xgb_r, ridge_r],
            "nnls_weights": [float(w) for w in weights],
            "bias_a": a, "bias_b": b,
            "target_transform": "log1p",
            "xgb_uses_dmatrix": True,
            "metrics": out["test"],
            "persistence": out["persistence"],
            "climatology_y": {"%d|%d|%d" % k: float(v)
                              for k, v in y_tables[poll].items()},
            "climatology_drivers": {
                "%d|%d" % k: {"target_ssrd_clim": float(r["target_ssrd_clim"]),
                              "target_blh_clim": float(r["target_blh_clim"])}
                for k, r in drivers.iterrows()},
        }, fh, protocol=4)
    return out


def main():
    t0 = time.time()
    print("=" * 118)
    print("SIH26082 v3 - MAXIMUM-ACCURACY ENSEMBLE")
    print("=" * 118)
    print("LightGBM              : CPU (pip wheel has no GPU tree learner)")
    print("XGBoost               : device=%s" % XGB_DEVICE)
    print("members               : LightGBM-L1 + XGBoost-L1 + Ridge  (NNLS simplex)")
    print("refit                 : train+val at %.2fx rounds after val selection"
          % REFIT_GROWTH)

    df = s2.load_data()
    masks = s2.make_masks(df)
    static_cols = [c for c in df.columns
                   if not c.startswith("target_") and c not in
                   ("timestamp_utc", "station_id", "latitude", "longitude",
                    "_month_ist", "_hour_ist")]
    print("train/val/test        : %d / %d / %d"
          % (masks["train"].sum(), masks["val"].sum(), masks["test"].sum()))

    drivers, y_tables = s2.fit_climatology(df, masks)

    results = []
    for poll in POLLUTANTS:
        for h in HORIZONS:
            ts = time.time()
            r = train_one_v3(df, masks, poll, h, static_cols, drivers, y_tables)
            results.append(r)
            d = r["delta_r2_vs_legacy"]
            print("  %-5s h=%-3d R2=%.4f d=%.4f skill=%+.3f  w=%s  dR2=%s  (%.0fs)"
                  % (CPCB_LABEL[poll], h, r["test"]["r2"], r["test"]["willmott_d"],
                     r["test"]["skill_vs_persistence"],
                     "/".join("%.2f" % w for w in r["nnls_weights"]),
                     ("%+.4f" % d) if d is not None else " n/a",
                     time.time() - ts))

    print("\n" + "=" * 118)
    print("V3 HELD-OUT TEST PERFORMANCE  (2025-07-01 .. 2025-12-31, pure NaN targets)")
    print("=" * 118)
    print("%-6s %4s %8s %9s %8s %8s %9s %8s %9s %9s"
          % ("POL", "h", "n", "R2", "RMSE", "MAE", "d", "SMAPE%", "skill", "dR2"))
    print("-" * 118)
    for poll in POLLUTANTS:
        for r in sorted([x for x in results if x["pollutant"] == poll],
                        key=lambda x: x["horizon"]):
            t, p = r["test"], r["persistence"]
            d = r["delta_r2_vs_legacy"]
            print("%-6s %4d %8d %9.4f %8.3f %8.3f %9.4f %8.2f %9.4f %9s"
                  % (CPCB_LABEL[poll], r["horizon"], t.get("n", 0),
                     t.get("r2", np.nan), t.get("rmse", np.nan), t.get("mae", np.nan),
                     t.get("willmott_d", np.nan), t.get("smape_pct", np.nan),
                     t.get("skill_vs_persistence", np.nan),
                     ("%+.4f" % d) if d is not None else "n/a"))
        print("-" * 118)

    with open(os.path.join(ART_DIR, "metrics_v3.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)

    # ---- comparison against the v2 run, where both exist -------------------
    v2_path = os.path.join(HERE, "artifacts", "metrics_by_horizon.json")
    if os.path.exists(v2_path):
        with open(v2_path) as fh:
            v2 = json.load(fh)
        v2map = {(x["pollutant"], x["horizon"]): x["test"].get("r2") for x in v2}
        print("\nV3 vs V2 (R2 delta; positive means v3 is better)")
        print("-" * 118)
        for poll in POLLUTANTS:
            row = []
            for r in sorted([x for x in results if x["pollutant"] == poll],
                            key=lambda x: x["horizon"]):
                old = v2map.get((poll, r["horizon"]))
                row.append("h%-3d %s" % (r["horizon"],
                                         ("%+.4f" % (r["test"]["r2"] - old))
                                         if old is not None else "  n/a"))
            print("%-6s %s" % (CPCB_LABEL[poll], "  ".join(row)))

    print("\nwrote %s" % os.path.join(ART_DIR, "metrics_v3.json"))
    print("TOTAL TIME %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
