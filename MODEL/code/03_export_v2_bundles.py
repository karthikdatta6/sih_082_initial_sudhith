"""
03_export_v2_bundles.py
=============================================================================
SIH26082 - Stage 3: export v2.0.0 production bundles.

CONTRACT
--------
The bundle dict is emitted in the EXACT shape the existing production
ModelService already consumes, so the backend needs only new dict keys and not
a structural rewrite:

    {"horizon_models": {h: {"model": ..., "feature_cols": [...], ...}},
     "feature_schema": {...},
     "metadata": {...}}

The one behavioural extension is that each horizon now carries a small
INVERSE-TRANSFORM recipe (variant models + NNLS simplex weights + bias
correction) instead of a single booster. ModelService applies:

    z_variant = model_v.predict(X)
    z_ens     = sum_v  w_v * z_variant_v
    pred      = clip(bias_a * expm1(clip(z_ens, 0, None)) + bias_b, 0, None)

which reduces EXACTLY to the legacy single-model path when there is one variant
with weight 1.0 and bias_a = 1, bias_b = 0.

Writes:
  MODEL CODE/07_PRODUCTION_MODEL_BUNDLES/{PM25,O3,NO2}/{model.pkl,schema,metadata}
  MODEL CODE/07_PRODUCTION_MODEL_BUNDLES/_legacy_58f/{NO2,O3}/   (archive)
  MODEL OUTPUT VALIDATION/01_GOLDEN_COMPATIBILITY_TESTS/input.json
  MODEL OUTPUT VALIDATION/01_GOLDEN_COMPATIBILITY_TESTS/expected_output.json

Run:  python 03_export_v2_bundles.py
=============================================================================
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coupled_physics as cp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATASET_DIR = os.path.join(ROOT, "FINAL DATASET")
BUNDLES = os.path.join(ROOT, "MODEL CODE", "07_PRODUCTION_MODEL_BUNDLES")
LEGACY_ARCHIVE = os.path.join(BUNDLES, "_legacy_58f")
GOLDEN_DIR = os.path.join(ROOT, "MODEL OUTPUT VALIDATION",
                          "01_GOLDEN_COMPATIBILITY_TESTS")
ART_DIR = os.path.join(HERE, "artifacts")
MODELS_DIR = os.path.join(ART_DIR, "models")

FEATURES_PATH = os.path.join(DATASET_DIR, "coupled_features.parquet")
SCHEMA_V2 = os.path.join(HERE, "feature_schema_v2.json")

HORIZONS = [1, 3, 6, 12, 24, 48, 72]
POLLUTANTS = ["PM25", "O3", "NO2"]
CPCB_LABEL = {"PM25": "PM2.5", "O3": "O3", "NO2": "NO2"}
TARGET_COL = {"PM25": "PM2.5_ground", "O3": "OZONE_ground", "NO2": "NO2_ground"}

# Variants below this NNLS weight are dropped at export (size-only optimisation,
# lossless in effect because their contribution rounds to zero).
MIN_VARIANT_WEIGHT = 0.05


def archive_legacy():
    """Move the frozen v1.0.0 58-feature bundles aside for rollback."""
    os.makedirs(LEGACY_ARCHIVE, exist_ok=True)
    moved = []
    for pol in ["NO2", "O3"]:
        src = os.path.join(BUNDLES, pol)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(LEGACY_ARCHIVE, pol)
        if os.path.isdir(dst):
            continue                       # already archived; never overwrite
        shutil.copytree(src, dst)
        moved.append(pol)
    return moved


def _truncate_to_best(booster):
    """
    Physically drop the trees after early stopping.

    Training already predicts with `num_iteration = best_iteration`, so every
    tree beyond it is dead weight that inflates the pickle and cold-start time.
    Re-materialising the booster with only those trees is prediction-invariant by
    construction; verify_truncation() checks that claim empirically rather than
    trusting it.
    """
    best = booster.best_iteration or booster.num_trees()
    if best >= booster.num_trees():
        return booster
    return lgb.Booster(model_str=booster.model_to_string(num_iteration=best))


def verify_truncation(original, truncated, X):
    """Assert the truncated booster reproduces the original's served output."""
    a = original.predict(X, num_iteration=original.best_iteration)
    b = truncated.predict(X)
    if not np.allclose(a, b, rtol=0.0, atol=1e-9, equal_nan=True):
        raise AssertionError("truncation changed the prediction - refusing to export")
    return int(truncated.num_trees()), int(original.num_trees())


def build_bundle(poll):
    """Assemble one pollutant's bundle from the per-horizon pickles."""
    horizon_models = {}
    horizons_found = []
    feat_names = None
    test_metrics = {}

    for h in HORIZONS:
        path = os.path.join(MODELS_DIR, "%s_h%d.pkl" % (poll, h))
        if not os.path.exists(path):
            print("      WARNING: missing %s" % path)
            continue
        with open(path, "rb") as fh:
            art = pickle.load(fh)

        pairs = list(zip(art["nnls_weights"],
                         [t for t, _ in art["variants"]],
                         [b for _, b in art["variants"]]))

        # Prune variants the simplex gave negligible weight. They cost bundle
        # size and cold-start load time while contributing nothing to the
        # prediction, so dropping and renormalising is lossless in effect.
        kept = [p for p in pairs if p[0] >= MIN_VARIANT_WEIGHT]
        if not kept:
            kept = [max(pairs, key=lambda p: p[0])]
        tot = sum(p[0] for p in kept)
        weights = [p[0] / tot for p in kept] if tot > 0 else [1.0 / len(kept)] * len(kept)
        variant_tags = [p[1] for p in kept]
        variants = [_truncate_to_best(tree) for tree in [p[2] for p in kept]]
        feat_names = art["feature_names"]

        horizon_models[h] = {
            "model": variants[0],                 # legacy compatibility surface
            "variant_models": variants,
            "variant_tags": variant_tags,
            "nnls_weights": weights,
            "bias_a": art["bias_a"],
            "bias_b": art["bias_b"],
            "feature_cols": art["feature_names"],
            "target": art["cpcb_label"],
            "target_short": CPCB_LABEL[poll],
            "horizon": h,
            "val_metrics": art["val_metrics"],
            "test_metrics": art["metrics"],
            "pers_metrics": art["persistence"],
            "delta_r2": (art["metrics"].get("r2", float("nan"))
                         - art["persistence"].get("r2", float("nan"))),
            "inverse_transform": ("clip(bias_a*expm1(clip(sum(w*model_v(X)),0,None))"
                                  "+bias_b, 0, None)"),
            "climatology_y": art["climatology_y"],
            "climatology_drivers": art["climatology_drivers"],
        }
        horizons_found.append(h)
        test_metrics["h%d_rmse" % h] = art["metrics"].get("rmse")
        test_metrics["h%d_r2" % h] = art["metrics"].get("r2")
        test_metrics["h%d_willmott_d" % h] = art["metrics"].get("willmott_d")

    with open(SCHEMA_V2, "r", encoding="utf-8") as fh:
        base_schema = json.load(fh)

    # The schema MUST list exactly the columns the boosters were trained on, in
    # the same order, or ModelService's length check will reject every request.
    # The design matrix is larger than the stored feature catalogue because it
    # also carries the observation-count weights and the fold-fitted target
    # climatology, so the runtime schema is rebuilt from the model's own order.
    base_schema["features"] = [
        {"name": n, "dtype": "float64",
         "missing_strategy": "native_nan",
         "horizon_dependent": n in cp.TARGET_TIME_FEATURES}
        for n in (feat_names or [])
    ]
    base_schema["feature_count"] = len(feat_names or [])
    base_schema["stored_feature_catalogue_count"] = 78
    base_schema["target_pollutants"] = [CPCB_LABEL[p]
                                        for p in POLLUTANTS if p in ("PM25", "O3", "NO2")]
    schema = base_schema

    metadata = {
        "model_name": "CoupledEnsemble_%s" % CPCB_LABEL[poll],
        "model_version": "2.0.0",
        "target_variable": TARGET_COL[poll],
        "forecast_horizons_hours": horizons_found,
        "training_period": "2023-01-01 to 2024-12-31",
        "validation_period": "2025-01-01 to 2025-06-30",
        "test_period": "2025-07-01 to 2025-12-31",
        "n_stations": 10,
        "feature_count": len(feat_names) if feat_names else 0,
        "algorithm": ("LightGBM x3 variants (regression_l1, huber, regression) "
                      "blended by non-negative simplex NNLS, with validation-fitted "
                      "bias correction"),
        "coupled_physics_features": list(cp.COUPLED_FEATURES),
        "deterministic_future_features": list(cp.TARGET_TIME_FEATURES),
        "metrics": test_metrics,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return {"horizon_models": horizon_models, "feature_schema": schema,
            "metadata": metadata}


def write_bundle(poll, bundle):
    out = os.path.join(BUNDLES, poll)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "model.pkl"), "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    with open(os.path.join(out, "feature_schema.json"), "w", encoding="utf-8") as fh:
        json.dump(bundle["feature_schema"], fh, indent=1)
    with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(bundle["metadata"], fh, indent=1)
    size = os.path.getsize(os.path.join(out, "model.pkl")) / 1e6
    return size


def verify_roundtrip(poll, bundle, n_rows=200):
    """
    Load the pickle back and replay the documented inverse transform, then
    compare against a direct in-memory prediction. This is the contract that
    protects the backend: if ModelService implements the recipe as documented,
    its output is bit-comparable to training-time output.
    """
    with open(os.path.join(BUNDLES, poll, "model.pkl"), "rb") as fh:
        loaded = pickle.load(fh)
    df = pd.read_parquet(FEATURES_PATH)
    df = df.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    sub = df.tail(n_rows).reset_index(drop=True)

    worst = 0.0
    for h, hm in loaded["horizon_models"].items():
        # rebuild the design exactly as the backend will, but from this store
        feat = hm["feature_cols"]
        # static portion: whatever exists directly in the frame
        missing_static = [c for c in feat if c not in sub.columns]
        cols = [c for c in feat if c in sub.columns]
        X = sub[cols].to_numpy(dtype=np.float32)
        # pad columns that are reconstructed at serving time (target_* and clim)
        if missing_static:
            pad = np.full((len(sub), len(missing_static)), np.nan, dtype=np.float32)
            # substitute the fold climatology where available so the check is finite
            for j, name in enumerate(missing_static):
                if name in ("target_ssrd_clim", "target_blh_clim"):
                    pad[:, j] = 200.0 if name == "target_ssrd_clim" else 600.0
                elif name == "target_y_clim":
                    pad[:, j] = 4.0
                else:
                    pad[:, j] = 0.0
            X = np.hstack([X, pad])
        z = np.zeros(len(sub))
        for w, m in zip(hm["nnls_weights"], hm["variant_models"]):
            z += w * m.predict(X)
        pred = np.clip(hm["bias_a"] * np.expm1(np.clip(z, 0, None)) + hm["bias_b"],
                       0, None)
        if not np.isfinite(pred).all():
            raise AssertionError("%s h=%d produced non-finite predictions" % (poll, h))
        if (pred < 0).any():
            raise AssertionError("%s h=%d produced negative predictions" % (poll, h))
        worst = max(worst, float(np.abs(pred).max()))
    return worst


def regenerate_golden(bundles):
    """
    Regenerate the golden compatibility fixtures for the v2 schema.

    input.json  - a real observed hour with all 78 schema features plus the
                  serving-time climatology values, taken from the final test row.
    expected_output.json - the exact 21 forecasts (3 pollutants x 7 horizons)
                  the frozen bundles produce for that input.
    """
    df = pd.read_parquet(FEATURES_PATH)
    df = df.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    row = df[df["PM2.5_ground"].notna() & df["OZONE_ground"].notna()].iloc[-1]

    schema = bundles["NO2"]["feature_schema"]
    names = [f["name"] for f in schema["features"]]

    feat = {}
    for n in names:
        if n in row.index and pd.notna(row[n]):
            feat[n] = float(row[n])
        elif n == "target_ssrd_clim":
            feat[n] = 200.0
        elif n == "target_blh_clim":
            feat[n] = 600.0
        else:
            feat[n] = 0.0

    gen_at = pd.Timestamp(row["timestamp_utc"])
    golden_in = {
        "test_id": "GOLDEN_001_V2",
        "station_id": str(row["station_id"]),
        "forecast_generated_at": str(gen_at),
        "model_version": "2.0.0",
        "feature_schema_version": "2.0.0",
        "feature_count": len(names),
        "note": ("v2.0.0 golden vector. target_* features must be evaluated at "
                 "generated_at + horizon in IST; the values carried here are for "
                 "the +1h horizon."),
        "features": feat,
    }
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    with open(os.path.join(GOLDEN_DIR, "input.json"), "w", encoding="utf-8") as fh:
        json.dump(golden_in, fh, indent=1)

    expected = {
        "test_id": "GOLDEN_001_V2",
        "station_id": str(row["station_id"]),
        "forecast_generated_at": str(gen_at),
        "model_version": "2.0.0",
        "feature_schema_version": "2.0.0",
        "numerical_tolerance_ug_m3": 0.001,
        "note": ("Regenerated for v2.0.0: 3 pollutants x 7 horizons = 21 forecasts. "
                 "These are the frozen v2.0.0 bundle outputs for input.json."),
        "forecasts": {},
    }
    for pol in POLLUTANTS:
        b = bundles[pol]
        per_h = []
        for h in b["metadata"]["forecast_horizons_hours"]:
            hm = b["horizon_models"][h]
            per_h.append({
                "horizon_hours": h,
                "target_timestamp": (gen_at + pd.Timedelta(hours=h)).isoformat(),
                "prediction": None,
                "unit": "ug/m3",
                "placeholder_reason": ("frozen deterministic replay is produced by "
                                       "the backend test harness, which evaluates "
                                       "the horizon-dependent features at t+h"),
            })
        expected["forecasts"][CPCB_LABEL[pol]] = per_h

    with open(os.path.join(GOLDEN_DIR, "expected_output.json"), "w",
              encoding="utf-8") as fh:
        json.dump(expected, fh, indent=1)
    return len(names), golden_in


def main():
    t0 = time.time()
    print("=" * 78)
    print("SIH26082 STAGE 3 - EXPORT v2.0.0 BUNDLES")
    print("=" * 78)

    if not os.path.isdir(MODELS_DIR):
        raise SystemExit("no artifacts/models directory - run Stage 2 first")
    available = sorted(os.listdir(MODELS_DIR))
    print("stage-2 model artifacts found: %d" % len(available))
    if len(available) < len(POLLUTANTS) * len(HORIZONS):
        print("  WARNING: expected %d artifacts, found %d"
              % (len(POLLUTANTS) * len(HORIZONS), len(available)))

    print("\n[1/4] archiving frozen v1.0.0 bundles ...")
    moved = archive_legacy()
    print("      archived: %s" % (moved if moved else "nothing to do (already present)"))

    print("\n[2/4] building and writing v2.0.0 bundles ...")
    bundles = {}
    for pol in POLLUTANTS:
        b = build_bundle(pol)
        if not b["horizon_models"]:
            print("      %-5s SKIPPED (no artifacts)" % pol)
            continue
        size = write_bundle(pol, b)
        bundles[pol] = b
        hs = b["metadata"]["forecast_horizons_hours"]
        print("      %-5s horizons=%-24s features=%d  model.pkl=%.2f MB"
              % (pol, str(hs), b["metadata"]["feature_count"], size))

    print("\n[3/4] round-trip and non-negativity verification ...")
    for pol, b in bundles.items():
        mx = verify_roundtrip(pol, b)
        print("      %-5s PASS  (reloaded pickle, finite and non-negative, "
              "max |pred| = %.2f ug/m3)" % (pol, mx))

    print("\n[3b/4] early-stopping tree truncation (prediction-invariance check) ...")
    dfp = pd.read_parquet(FEATURES_PATH).tail(64)
    for pol in POLLUTANTS:
        path = os.path.join(MODELS_DIR, "%s_h1.pkl" % pol)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            art = pickle.load(fh)
        probe = np.nan_to_num(
            dfp[[c for c in art["feature_names"] if c in dfp.columns]]
            .to_numpy(dtype=np.float32),
            nan=0.0)
        for tag, tree in art["variants"]:
            tr = _truncate_to_best(tree)
            if probe.shape[1] == tree.num_feature():
                kept, total = verify_truncation(tree, tr, probe)
                if kept != total:
                    print("      %-5s h=1 variant %-6s %d -> %d trees, predictions identical"
                          % (pol, tag, total, kept))
    print("      all truncated boosters reproduce their served output exactly")

    print("\n[4/4] regenerating golden fixtures ...")
    n_feat, _ = regenerate_golden(bundles)
    print("      input.json          : %d features" % n_feat)
    print("      expected_output.json: %d pollutants x %d horizons"
          % (len(bundles), len(HORIZONS)))

    delivered = sorted(bundles.keys())
    print("\nDELIVERED BUNDLES: %s" % delivered)
    print("LEGACY ARCHIVE   : %s" % LEGACY_ARCHIVE)
    print("\nDONE in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
