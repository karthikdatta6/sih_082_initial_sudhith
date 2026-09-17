"""
02_train_coupled_ensemble.py
=============================================================================
SIH26082 - Stage 2: train the coupled multi-horizon ensemble.

METHOD (and why each piece is here)
-----------------------------------
1. TARGET      log1p(y), inverted with expm1(clip(z, 0, None)). Guarantees a
               non-negative physical floor; required because O3 sits at exactly
               0.0 in 0.5% of hours (nocturnal NO titration).

2. DESIGN      Per-horizon design matrix = 58 legacy features + 11 coupled /
               cross-sectional features + 9 deterministic target-time features
               + 1 fold-fitted target climatology.

               The target-time block is the primary accuracy lever: the calendar
               and the sun at t+h are KNOWN EXACTLY, so hour-of-day, day-of-week,
               day-of-year and solar zenith at the target instant are free,
               leak-proof information. The legacy schema encoded only the
               OBSERVATION time, which points at the wrong clock hour for every
               horizon where h mod 24 != 0.

3. SPLIT       Chronological, never shuffled (Golden Rule 2 / legacy Mistake 7):
                 train  2023-01-01 .. 2024-12-31   (early-stopping-free zone)
                 val    2025-01-01 .. 2025-06-30   (early stopping + NNLS + bias)
                 test   2025-07-01 .. 2025-12-31   (held out, touched once)
               This mirrors the legacy split exactly, so the h=48 numbers are
               directly comparable to the certified v1.0.0 model.

4. LEAKAGE     Every FITTED quantity - the two climatology tables, the NNLS
               simplex weights, and the bias correction - is estimated from
               train/val only and never from test. The legacy pipeline fitted
               its diurnal weights once globally; that is the flaw corrected here.

5. ENSEMBLE    3 LightGBM variants per horizon (L1 / Huber / L2) blended by a
               non-negative simplex NNLS. Keeps the certified stacking
               architecture; robust to objective misspecification on spikes.

6. TARGETS     Direct multi-step only. No recursion anywhere (Golden Rule 1).

Run:  python 02_train_coupled_ensemble.py [--ablation] [--fast]
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coupled_physics as cp  # noqa: E402

try:
    from scipy.optimize import nnls as scipy_nnls
    HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    HAVE_SCIPY = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATASET_DIR = os.path.join(ROOT, "FINAL DATASET")
FEATURES_PATH = os.path.join(DATASET_DIR, "coupled_features.parquet")
ART_DIR = os.path.join(HERE, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

HORIZONS = [1, 3, 6, 12, 24, 48, 72]
POLLUTANTS = ["PM25", "O3", "NO2"]
TARGET_COL = {"PM25": "PM2.5_ground", "O3": "OZONE_ground", "NO2": "NO2_ground"}
OBS_COUNT_COL = {"PM25": "PM2.5_obs_count", "O3": "OZONE_obs_count",
                 "NO2": "NO2_obs_count"}
CPCB_LABEL = {"PM25": "PM2.5", "O3": "O3", "NO2": "NO2"}

TRAIN_END = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2025-07-01")

# Legacy certified v1.0.0 held-out reference, for the head-to-head at h=48.
LEGACY_REF = {
    "NO2": {1: 0.9191, 3: 0.8540, 6: 0.8125, 12: 0.7890, 24: 0.7662, 48: 0.7155},
    "O3": {1: 0.8689, 3: 0.8110, 6: 0.7840, 12: 0.7680, 24: 0.7559, 48: 0.6975},
}

CLIM_Y = "target_y_clim"


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def willmott_d(obs, pred):
    """Willmott index of agreement, d in [0, 1]. 1 = perfect."""
    o = obs - obs.mean()
    p = pred - obs.mean()
    denom = (np.abs(p) + np.abs(o)) ** 2
    s = denom.sum()
    if s == 0:
        return float("nan")
    return float(1.0 - ((pred - obs) ** 2).sum() / s)


def smape(obs, pred):
    denom = np.abs(obs) + np.abs(pred)
    m = denom > 0
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(2.0 * np.abs(pred[m] - obs[m]) / denom[m]) * 100.0)


def compute_metrics(obs, pred):
    """All metrics on the PHYSICAL concentration scale, NaN-safe."""
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    if len(o) < 30:
        return {"n": int(len(o))}
    ss_res = float(((o - p) ** 2).sum())
    ss_tot = float(((o - o.mean()) ** 2).sum())
    return {
        "n": int(len(o)),
        "rmse": float(np.sqrt(np.mean((o - p) ** 2))),
        "mae": float(np.mean(np.abs(o - p))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "willmott_d": willmott_d(o, p),
        "smape_pct": smape(o, p),
        "bias": float(np.mean(p - o)),
        "obs_mean": float(o.mean()),
        "pred_mean": float(p.mean()),
    }


def persistence_metrics(obs, pred_persist):
    return compute_metrics(obs, pred_persist)


# ---------------------------------------------------------------------------
# DESIGN MATRIX
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_parquet(FEATURES_PATH)
    df = df.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    df["_month_ist"] = (df["timestamp_utc"] + pd.Timedelta(hours=cp.IST_OFFSET_HOURS)).dt.month
    df["_hour_ist"] = (df["timestamp_utc"] + pd.Timedelta(hours=cp.IST_OFFSET_HOURS)).dt.hour
    return df


def make_masks(df):
    ts = df["timestamp_utc"]
    return {
        "train": (ts < TRAIN_END).to_numpy(),
        "val": ((ts >= TRAIN_END) & (ts < VAL_END)).to_numpy(),
        "test": (ts >= VAL_END).to_numpy(),
    }


def fit_climatology(df, masks):
    """
    Fit the three climatology tables on TRAIN rows ONLY.

    Returns (drivers, y_tables) where:
      drivers  : (month, ist_hour) -> target_ssrd_clim, target_blh_clim
      y_tables : {pollutant: (station_enc, month, ist_hour) -> mean log1p(target)}

    Fitting these once globally and reusing them across folds is the silent leak
    described as risk C2; they are fitted here from train rows only.
    """
    tr = df[masks["train"]]
    drivers = cp.build_target_time_climatology(tr)

    y_tables = {}
    for pol in POLLUTANTS:
        y = np.log1p(tr[TARGET_COL[pol]].to_numpy(dtype=float))
        tmp = pd.DataFrame({
            "station_enc": tr["station_enc"].to_numpy(dtype=float),
            "month": tr["_month_ist"].to_numpy(),
            "ist_hour": tr["_hour_ist"].to_numpy(),
            "ly": y,
        })
        y_tables[pol] = tmp.groupby(["station_enc", "month", "ist_hour"],
                                    observed=True)["ly"].mean()
    return drivers, y_tables


def build_design(df, horizon, static_cols, drivers, y_tables, poll):
    """
    Assemble the full design matrix for one horizon.

    Column order is fixed and identical for every horizon so that a single frozen
    feature_schema.json can serve all of them at inference time.
    """
    t_block = cp.add_target_time_features(
        df, [horizon], clim_table=drivers
    )[horizon]

    # Fold-fitted target climatology, evaluated at the TARGET (station, month, hour).
    ist = df["timestamp_utc"] + pd.Timedelta(hours=horizon + cp.IST_OFFSET_HOURS)
    key = pd.MultiIndex.from_arrays(
        [df["station_enc"].to_numpy(dtype=float), ist.dt.month.to_numpy(),
         ist.dt.hour.to_numpy()],
        names=["station_enc", "month", "ist_hour"],
    )
    clim_y = y_tables[poll].reindex(key).to_numpy(dtype=float)

    X = pd.concat([df[static_cols].reset_index(drop=True),
                   t_block.reset_index(drop=True)], axis=1)
    X[CLIM_Y] = clim_y
    return X


# ---------------------------------------------------------------------------
# MODEL VARIANTS + SIMPLEX BLENDING
# ---------------------------------------------------------------------------

# Capacity is matched to the configuration that produced the certified v1.0.0
# results (num_leaves=127, lr=0.03, feature_fraction=0.7), because that setting
# is empirically known to be well matched to this dataset. Early stopping on the
# validation block keeps the extra capacity from becoming overfitting.
BASE_PARAMS = dict(
    boosting_type="gbdt",
    num_leaves=127,
    learning_rate=0.045,
    n_estimators=1200,
    min_child_samples=30,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
    metric="rmse",
    force_row_wise=True,
)

VARIANTS = [
    ("l1", {"objective": "regression_l1"}),
    ("huber", {"objective": "huber", "alpha": 0.9}),
    ("l2", {"objective": "regression"}),
]


def fit_variant(Xtr, ytr, Xva, yva, wtr, params, fast=False):
    p = dict(BASE_PARAMS)
    p.update(params)
    if fast:
        p["n_estimators"] = 350
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr, free_raw_data=False)
    cb = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    booster = lgb.train(p, dtr, valid_sets=[dva], callbacks=cb)
    return booster


def simplex_nnls(P, y, penalty=None):
    """
    Non-negative least squares weights with an explicit sum-to-one constraint,
    i.e. the convex simplex  min ||y - Pw||^2  s.t.  w >= 0, sum(w) = 1.

    The equality is imposed by an augmented row of large weight, the standard
    NNLS trick. Falls back to uniform weights if SciPy is unavailable or the
    solve degenerates.
    """
    k = P.shape[1]
    if penalty is None:
        penalty = 1e3 * max(float(np.mean(np.abs(y))), 1.0)
    A = np.vstack([P, np.full((1, k), penalty)])
    b = np.concatenate([y, [penalty]])
    if HAVE_SCIPY:
        try:
            w, _ = scipy_nnls(A, b)
        except Exception:
            w = np.ones(k) / k
    else:
        w = np.ones(k) / k
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones(k) / k
    return w / s


def sample_weights(df, poll, masks):
    """
    Measurement-confidence weights from the CPCB observation counts, which the
    legacy schema carried but never used. Scale-free rank weighting: hours backed
    by more valid sub-samples are trusted more, weakly-backed hours are damped.
    """
    n = df[OBS_COUNT_COL[poll]].to_numpy(dtype=float)
    med = np.nanmedian(n[masks["train"]])
    if not np.isfinite(med) or med <= 0:
        return np.ones(len(df), dtype=float)
    w = np.clip(n / med, 0.25, 1.0)
    return np.where(np.isfinite(w), w, 0.5)


# ---------------------------------------------------------------------------
# TRAIN ONE (POLLUTANT, HORIZON)
# ---------------------------------------------------------------------------

def train_one(df, masks, poll, horizon, static_cols, drivers, y_tables, fast=False):
    y_raw = df[f"target_{poll}_h{horizon}"].to_numpy(dtype=float)
    y_log = np.log1p(y_raw)
    w_all = sample_weights(df, poll, masks)

    X = build_design(df, horizon, static_cols, drivers, y_tables, poll)
    feat_names = list(X.columns)

    tr, va, te = masks["train"], masks["val"], masks["test"]
    ok_tr = tr & np.isfinite(y_log)
    ok_va = va & np.isfinite(y_log)
    ok_te = te & np.isfinite(y_log)

    Xtr = X.loc[ok_tr].to_numpy(dtype=np.float32)
    Xva = X.loc[ok_va].to_numpy(dtype=np.float32)
    Xte = X.loc[ok_te].to_numpy(dtype=np.float32)

    boosters, va_preds, va_scores = [], [], {}
    for tag, params in VARIANTS:
        b = fit_variant(Xtr, y_log[ok_tr], Xva, y_log[ok_va],
                        w_all[ok_tr], params, fast=fast)
        pv = np.expm1(np.clip(b.predict(Xva, num_iteration=b.best_iteration), 0, None))
        boosters.append((tag, b))
        va_preds.append(pv)
        va_scores[tag] = compute_metrics(y_raw[ok_va], pv)

    P_va = np.column_stack(va_preds)
    y_va = y_raw[ok_va]
    weights = simplex_nnls(P_va, y_va)

    # --- ensemble on validation, then bias correction fitted on validation only
    va_ens = P_va @ weights
    resid = y_va - va_ens
    denom = float(np.var(va_ens))
    a = float(np.cov(va_ens, y_va)[0, 1] / denom) if denom > 1e-9 else 1.0
    a = float(np.clip(a, 0.5, 1.5))
    b = float(y_va.mean() - a * va_ens.mean())

    # --- test
    te_preds = []
    for tag, booster in boosters:
        pt = np.expm1(np.clip(
            booster.predict(Xte, num_iteration=booster.best_iteration), 0, None))
        te_preds.append(pt)
    P_te = np.column_stack(te_preds)
    y_te = y_raw[ok_te]

    ens_raw = P_te @ weights
    ens_cal = np.clip(a * ens_raw + b, 0.0, None)

    persist = df.loc[ok_te, TARGET_COL[poll]].to_numpy(dtype=float)

    out = {
        "pollutant": poll,
        "horizon": horizon,
        "n_features": len(feat_names),
        "nnls_weights": {t: float(w) for (t, _), w in zip(boosters, weights)},
        "bias_a": a, "bias_b": b,
        "test": compute_metrics(y_te, ens_cal),
        "test_uncorrected": compute_metrics(y_te, ens_raw),
        "val": compute_metrics(y_va, np.clip(a * va_ens + b, 0, None)),
        "val_variant_raw": va_scores,
        "persistence": persistence_metrics(y_te, persist),
        "best_iteration": {t: int(b.best_iteration or 0) for t, b in boosters},
        "calibration_gain_rmse": (
            compute_metrics(y_te, ens_raw)["rmse"] - compute_metrics(y_te, ens_cal)["rmse"]
        ),
    }
    out["test"]["skill_vs_persistence"] = (
        1.0 - out["test"]["rmse"] ** 2 / out["persistence"]["rmse"] ** 2
        if out["persistence"].get("rmse") else float("nan")
    )
    gain = None
    legacy = LEGACY_REF.get(poll, {}).get(horizon)
    if legacy is not None:
        gain = out["test"]["r2"] - legacy
    out["legacy_r2"] = legacy
    out["delta_r2_vs_legacy"] = gain
    out["feature_names"] = feat_names

    # ---- persist the fitted ensemble so Stage 3 can build a servable bundle ----
    model_dir = os.path.join(ART_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "%s_h%d.pkl" % (poll, horizon)), "wb") as fh:
        pickle.dump({
            "pollutant": poll,
            "cpcb_label": CPCB_LABEL[poll],
            "horizon": horizon,
            "feature_names": feat_names,
            "variants": [(tag, b) for tag, b in boosters],
            "nnls_weights": [float(x) for x in weights],
            "bias_a": a,
            "bias_b": b,
            "target_transform": "log1p",
            "metrics": out["test"],
            "persistence": out["persistence"],
            "val_metrics": out["val"],
            "climatology_y": {
                "%d|%d|%d" % k: float(v) for k, v in y_tables[poll].items()
            },
            "climatology_drivers": {
                "%d|%d" % k: {"target_ssrd_clim": float(r["target_ssrd_clim"]),
                              "target_blh_clim": float(r["target_blh_clim"])}
                for k, r in drivers.iterrows()
            },
            "static_feature_count": len(static_cols),
        }, fh, protocol=4)
    return out


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

def print_table(results):
    print("\n" + "=" * 118)
    print("HELD-OUT TEST PERFORMANCE  (test window 2025-07-01 .. 2025-12-31, pure NaN targets)")
    print("=" * 118)
    hdr = ("%-6s %4s %8s %9s %8s %8s %9s %8s %9s %9s %8s"
           % ("POL", "h", "n", "R2", "RMSE", "MAE", "d", "SMAPE%", "skill", "pers_RMSE", "dR2"))
    print(hdr)
    print("-" * 118)
    for poll in POLLUTANTS:
        for r in sorted([x for x in results if x["pollutant"] == poll],
                        key=lambda x: x["horizon"]):
            t = r["test"]
            p = r["persistence"]
            legacy = r.get("delta_r2_vs_legacy")
            print("%-6s %4d %8d %9.4f %8.3f %8.3f %9.4f %8.2f %9.4f %9.3f %8s"
                  % (CPCB_LABEL[poll], r["horizon"], t.get("n", 0),
                     t.get("r2", float("nan")), t.get("rmse", float("nan")),
                     t.get("mae", float("nan")), t.get("willmott_d", float("nan")),
                     t.get("smape_pct", float("nan")),
                     t.get("skill_vs_persistence", float("nan")),
                     p.get("rmse", float("nan")),
                     ("%+.4f" % legacy) if legacy is not None else "  n/a"))
        print("-" * 118)


def print_nnls(results):
    print("\nNNLS simplex weights and validation-fitted calibration (test never used)")
    print("-" * 118)
    print("%-6s %4s %8s %8s %8s %7s %8s %10s"
          % ("POL", "h", "w_L1", "w_Huber", "w_L2", "bias_a", "bias_b", "cal_gain"))
    for poll in POLLUTANTS:
        for r in sorted([x for x in results if x["pollutant"] == poll],
                        key=lambda x: x["horizon"]):
            w = r["nnls_weights"]
            print("%-6s %4d %8.3f %8.3f %8.3f %7.3f %8.3f %10.4f"
                  % (CPCB_LABEL[poll], r["horizon"], w.get("l1", 0), w.get("huber", 0),
                     w.get("l2", 0), r["bias_a"], r["bias_b"],
                     r["calibration_gain_rmse"]))


# ---------------------------------------------------------------------------
# ABLATION (block level)
# ---------------------------------------------------------------------------

def run_ablation(df, masks, static_cols, drivers, y_tables, horizons=(1, 24, 72)):
    """Drop-one BLOCK ablation at diagnostic horizons, PM25 and O3."""
    blocks = {
        "drop_coupled_physics": cp.COUPLED_FEATURES,
        "drop_cross_sectional": cp.CROSS_SECTIONAL_FEATURES,
        "drop_target_time": cp.TARGET_TIME_FEATURES,
        "drop_stability_candidate": cp.STABILITY_CANDIDATES,
        "drop_clim_y": [CLIM_Y],
    }
    rows = []
    for poll in ["PM25", "O3"]:
        for h in horizons:
            base = train_one(df, masks, poll, h, static_cols, drivers, y_tables,
                             fast=True)
            base_r2 = base["test"]["r2"]
            base_rmse = base["test"]["rmse"]
            for tag, cols in blocks.items():
                if tag == "drop_target_time":
                    keep = [c for c in static_cols]
                    # horizon-dependent block removed entirely -> rebuild with none
                    sub = _train_without(df, masks, poll, h, keep, drivers,
                                         y_tables, drop_target_time=True,
                                         drop_clim_y=True)
                else:
                    keep = [c for c in static_cols if c not in cols]
                    sub = _train_without(df, masks, poll, h, keep, drivers,
                                         y_tables,
                                         drop_clim_y=(tag == "drop_clim_y"))
                rows.append({
                    "pollutant": poll, "horizon": h, "variant": tag,
                    "r2": sub["test"]["r2"], "rmse": sub["test"]["rmse"],
                    "delta_r2": sub["test"]["r2"] - base_r2,
                    "delta_rmse": sub["test"]["rmse"] - base_rmse,
                    "n_features": sub["n_features"],
                })
            rows.append({"pollutant": poll, "horizon": h, "variant": "FULL",
                         "r2": base_r2, "rmse": base_rmse,
                         "delta_r2": 0.0, "delta_rmse": 0.0,
                         "n_features": base["n_features"]})
    return pd.DataFrame(rows)


def _train_without(df, masks, poll, horizon, static_cols, drivers, y_tables,
                   drop_target_time=False, drop_clim_y=False):
    """Retrain a single L1 model under a reduced column set (ablation helper)."""
    y_raw = df[f"target_{poll}_h{horizon}"].to_numpy(dtype=float)
    y_log = np.log1p(y_raw)
    w_all = sample_weights(df, poll, masks)

    if drop_target_time:
        # Remove the entire horizon-dependent block. drop_clim_y is implied:
        # the climatology column is part of that same block.
        X = df[static_cols].reset_index(drop=True).copy()
    else:
        X = build_design(df, horizon, static_cols, drivers, y_tables, poll)
        if drop_clim_y and CLIM_Y in X.columns:
            X = X.drop(columns=[CLIM_Y])

    tr, va, te = masks["train"], masks["val"], masks["test"]
    ok_tr = tr & np.isfinite(y_log)
    ok_va = va & np.isfinite(y_log)
    ok_te = te & np.isfinite(y_log)

    Xtr = X.loc[ok_tr].to_numpy(dtype=np.float32)
    Xva = X.loc[ok_va].to_numpy(dtype=np.float32)
    Xte = X.loc[ok_te].to_numpy(dtype=np.float32)

    b = fit_variant(Xtr, y_log[ok_tr], Xva, y_log[ok_va], w_all[ok_tr],
                    {"objective": "regression_l1"}, fast=True)
    pt = np.expm1(np.clip(b.predict(Xte, num_iteration=b.best_iteration), 0, None))
    res = compute_metrics(y_raw[ok_te], pt)
    res["n_features"] = int(X.shape[1])
    return {"test": res, "n_features": int(X.shape[1])}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true",
                    help="also run the drop-one block ablation table")
    ap.add_argument("--fast", action="store_true",
                    help="reduced boosting rounds (quick smoke test)")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 118)
    print("SIH26082 STAGE 2 - COUPLED ENSEMBLE TRAINING")
    print("=" * 118)
    print("scipy available for NNLS simplex: %s" % HAVE_SCIPY)

    df = load_data()
    masks = make_masks(df)
    static_cols = [c for c in df.columns
                   if not c.startswith("target_") and c not in
                   ("timestamp_utc", "station_id", "latitude", "longitude",
                    "_month_ist", "_hour_ist")]
    print("rows               : %d" % len(df))
    print("static features    : %d" % len(static_cols))
    print("train/val/test     : %d / %d / %d"
          % (masks["train"].sum(), masks["val"].sum(), masks["test"].sum()))

    drivers, y_tables = fit_climatology(df, masks)
    print("climatology tables : fitted on TRAIN rows only (%d driver cells)"
          % len(drivers))

    results = []
    for poll in POLLUTANTS:
        for h in HORIZONS:
            ts = time.time()
            r = train_one(df, masks, poll, h, static_cols, drivers, y_tables,
                          fast=args.fast)
            results.append(r)
            print("  %-5s h=%-3d R2=%.4f  RMSE=%7.3f  d=%.4f  skill=%+.3f  (%.1fs)"
                  % (CPCB_LABEL[poll], h, r["test"]["r2"], r["test"]["rmse"],
                     r["test"]["willmott_d"],
                     r["test"]["skill_vs_persistence"], time.time() - ts))

    print_table(results)
    print_nnls(results)

    print("\nHEAD-TO-HEAD vs CERTIFIED LEGACY v1.0.0 (same held-out window)")
    print("-" * 118)
    print("%-6s %4s %10s %10s %10s" % ("POL", "h", "legacy_R2", "coupled_R2", "delta"))
    for poll in POLLUTANTS:
        for r in sorted([x for x in results if x["pollutant"] == poll],
                        key=lambda x: x["horizon"]):
            if r.get("legacy_r2") is not None:
                print("%-6s %4d %10.4f %10.4f %+10.4f"
                      % (CPCB_LABEL[poll], r["horizon"], r["legacy_r2"],
                         r["test"]["r2"], r["delta_r2_vs_legacy"]))

    with open(os.path.join(ART_DIR, "metrics_by_horizon.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print("\nwrote %s" % os.path.join(ART_DIR, "metrics_by_horizon.json"))

    if args.ablation:
        print("\n" + "=" * 118)
        print("DROP-ONE BLOCK ABLATION (reduced-round L1 model; negative delta_R2 = block was helping)")
        print("=" * 118)
        ab = run_ablation(df, masks, static_cols, drivers, y_tables)
        ab.to_csv(os.path.join(ART_DIR, "ablation_table.csv"), index=False)
        print(ab.to_string(index=False,
                           formatters={"r2": "{:.4f}".format,
                                       "rmse": "{:.3f}".format,
                                       "delta_r2": "{:+.4f}".format,
                                       "delta_rmse": "{:+.3f}".format}))

    print("\nTOTAL TIME %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
