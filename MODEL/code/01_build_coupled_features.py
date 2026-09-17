"""
01_build_coupled_features.py
=============================================================================
SIH26082 - Stage 1: coupled feature store + direct multi-step targets.

DESIGN
------
The legacy store `features_engineered.parquet` (263,040 x 62) already contains
all 58 production features, station-isolated and in exact schema order (verified
by set-difference against the union of both legacy feature_schema.json files).
Rebuilding them would be the single largest regression risk in this project, so
this stage is a DELTA over that store, never a reimplementation.

Appended here:

  BLOCK 1  coupled physics      (6)  mandated two-way feedback terms
  BLOCK 3  cross-sectional      (3)  other-row information a GBDT cannot build
  CANDIDATE stability           (1)  dewpoint depression, judged by the ablation

BLOCK 2 (deterministic future forcing, 9 features) is horizon-dependent and is
materialised at training/serving time by coupled_physics.add_target_time_features,
which is what allows one frozen feature schema to serve all seven horizons.

OUTPUTS
-------
  FINAL DATASET/coupled_features.parquet
  MODEL CODE/09_SIH26082_COUPLED/feature_schema_v2.json
  MODEL CODE/09_SIH26082_COUPLED/feature_catalog.csv

LEAKAGE DISCIPLINE (Golden Rules 2, 3, 5)
-----------------------------------------
* Every appended feature is a function of information available at or before t.
* The ONLY forward-looking operation is the target block, groupby(station).shift(-h).
* Targets are never smoothed or imputed; genuine sensor gaps stay IEEE NaN.
* A truncation-based CAUSALITY test at the end of this file recomputes every
  static feature on a time-truncated frame and asserts bit-identical values on
  the shared prefix. The build fails if any feature peeks forward.

Run:  python 01_build_coupled_features.py
=============================================================================
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coupled_physics as cp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # .../SIH_26_AIR_O2
DATASET_DIR = os.path.join(ROOT, "FINAL DATASET")
LEGACY_BUNDLES = os.path.join(ROOT, "MODEL CODE", "07_PRODUCTION_MODEL_BUNDLES")

FUSED_PATH = os.path.join(DATASET_DIR, "station_hourly_fused.parquet")
ENGINEERED_PATH = os.path.join(DATASET_DIR, "features_engineered.parquet")

OUT_FEATURES = os.path.join(DATASET_DIR, "coupled_features.parquet")
OUT_SCHEMA = os.path.join(HERE, "feature_schema_v2.json")
OUT_CATALOG = os.path.join(HERE, "feature_catalog.csv")

HORIZONS = [1, 3, 6, 12, 24, 48, 72]
# NO2 is trained alongside the two primary pollutants: the official CPCB composite
# AQI needs a NO2 sub-index, and NO2 is already the dashboard's headline gas.
POLLUTANTS = ["PM25", "O3", "NO2"]
TARGET_COL = {"PM25": "PM2.5_ground", "O3": "OZONE_ground",
              "NO2": "NO2_ground"}
OBS_COUNT_COL = {"PM25": "PM2.5_obs_count", "O3": "OZONE_obs_count",
                 "NO2": "NO2_obs_count"}

# Columns needed from the 45-column fused store that features_engineered drops.
# latitude/longitude drive solar geometry and the NW axis projection; the
# obs_count columns are the measurement-confidence weights (previously unused
# anywhere in the legacy pipeline).
AUX_FROM_FUSED = [
    "timestamp_utc", "station_id", "latitude", "longitude",
    "PM2.5_obs_count", "PM10_obs_count", "OZONE_obs_count",
    "NO2_obs_count", "CO_obs_count",
]

# Static (timestamp-independent) feature columns produced by this stage.
STATIC_NEW_FEATURES = (
    cp.COUPLED_FEATURES + cp.CROSS_SECTIONAL_FEATURES + cp.STABILITY_CANDIDATES
)


# ---------------------------------------------------------------------------
# LEGACY SCHEMA INGESTION
# ---------------------------------------------------------------------------

def load_legacy_feature_names():
    """Ordered 58 legacy feature names, from the frozen NO2 production schema."""
    path = os.path.join(LEGACY_BUNDLES, "NO2", "feature_schema.json")
    with open(path, "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    feats = schema["features"]
    if feats and isinstance(feats[0], dict):
        return [f["name"] for f in feats]
    return list(feats)


# ---------------------------------------------------------------------------
# CROSS-SECTIONAL BUILDERS  (other-row information)
# ---------------------------------------------------------------------------

def loo_cross_sectional(df, value_col, ts="timestamp_utc", st="station_id"):
    """
    Leave-one-out cross-station mean of `value_col` at each timestamp.

    LOO rather than an inclusive mean: an inclusive mean is largely a restatement
    of the station's own reading and would let the model recover its own value.
    LOO says what the REST of the network saw at the same hour, which is the
    signature of a city-wide accumulation episode versus a local source.
    """
    piv = df.pivot_table(index=ts, columns=st, values=value_col, aggfunc="mean")
    arr = piv.to_numpy(dtype=float)
    n = np.isfinite(arr).sum(axis=1, keepdims=True).astype(float)
    s = np.nansum(arr, axis=1, keepdims=True)
    filled = np.nan_to_num(arr)
    with np.errstate(invalid="ignore", divide="ignore"):
        loo = (s - filled) / np.where(n - 1.0 > 0.0, n - 1.0, np.nan)
    loo = np.where(n - 1.0 > 0.0, loo, np.nan)
    wide = pd.DataFrame(loo, index=piv.index, columns=piv.columns)
    # Explicit long-form expansion rather than DataFrame.stack(), so the row set
    # is deterministic and every (timestamp, station) pair survives - including
    # pairs where the tracer is NaN. Losing those rows would silently shrink the
    # training set and misalign the merge that follows.
    midx = pd.MultiIndex.from_product([wide.index, wide.columns],
                                      names=[ts, st])
    out = pd.DataFrame({"_v": wide.to_numpy(dtype=float).ravel()},
                       index=midx).reset_index()
    return out


def axis_gradient(df, value_col, weights_by_station, ts="timestamp_utc",
                  st="station_id"):
    """
    Weighted spatial slope of a tracer along the NW->SE plume axis, per timestamp.
    Missing stations drop out and the remaining weights renormalise.
    """
    piv = df.pivot_table(index=ts, columns=st, values=value_col, aggfunc="mean")
    arr = piv.to_numpy(dtype=float)
    w = np.array([weights_by_station.get(c, np.nan) for c in piv.columns],
                 dtype=float)
    mask = np.isfinite(arr) & np.isfinite(w)[None, :]
    weighted = np.where(mask, arr * w[None, :], 0.0)
    denom = np.abs(np.where(mask, w[None, :], 0.0)).sum(axis=1, keepdims=True)
    num = weighted.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        grad = num / np.where(denom > 0.0, denom, np.nan)
    grad = np.where(denom > 0.0, grad, np.nan).ravel()
    return pd.DataFrame({ts: piv.index, "_v": grad})


# ---------------------------------------------------------------------------
# STATIC FEATURE BLOCK
# ---------------------------------------------------------------------------

def compute_static_features(df, weights_by_station):
    """
    Attach BLOCK 1 (coupled physics), BLOCK 3 (cross-sectional) and the stability
    candidate to a copy of `df`. Pure function of data at or before t.
    """
    out = df.copy()

    # ---- BLOCK 1: two-way coupled physics ---------------------------------
    out["aod_proxy"] = cp.aerosol_optical_depth(
        out["PM2.5_ground"].to_numpy(), out["era5_boundary_layer_height"].to_numpy()
    )
    out["clearness_index"] = cp.clearness_index(
        out["era5_solar_radiation_w_m2"].to_numpy(),
        cp.solar_zenith_cos(
            out["timestamp_utc"], out["latitude"].to_numpy(),
            out["longitude"].to_numpy(),
        ),
    )
    out["inversion_trap_index"] = cp.inversion_trap_index(
        out["era5_boundary_layer_height"].to_numpy(),
        out["era5_wind_speed"].to_numpy(),
    )
    out["nw_transport_flux"] = cp.nw_transport_flux(
        out["era5_u10"].to_numpy(), out["era5_v10"].to_numpy()
    )
    out["transport_lag_hours"] = cp.transport_lag_hours(
        out["nw_transport_flux"].to_numpy()
    )
    out["regional_burning_proxy"] = cp.regional_burning_proxy(
        out["sat_CO"].to_numpy(), out["sat_HCHO"].to_numpy(),
        out["timestamp_utc"].dt.dayofyear.to_numpy(),
    )

    # ---- BLOCK 3: cross-sectional ----------------------------------------
    grad = axis_gradient(out, "sat_CO", weights_by_station)
    out = out.merge(grad, on="timestamp_utc", how="left")
    out = out.rename(columns={"_v": "upwind_downwind_gradient"})

    loo_pm = loo_cross_sectional(out, "PM2.5_ground")
    loo_pm = loo_pm.rename(columns={"_v": "city_pm25_mean"})
    out = out.merge(loo_pm, on=["timestamp_utc", "station_id"], how="left")

    loo_co = loo_cross_sectional(out, "sat_CO")
    loo_co = loo_co.rename(columns={"_v": "city_satco_mean"})
    out = out.merge(loo_co, on=["timestamp_utc", "station_id"], how="left")

    grad_pm = axis_gradient(out, "PM2.5_ground", weights_by_station)
    out = out.merge(grad_pm, on="timestamp_utc", how="left")
    out = out.rename(columns={"_v": "pm25_nw_gradient"})

    # ---- stability candidate ---------------------------------------------
    out["dewpoint_depression"] = cp.dewpoint_depression(
        out["era5_temperature_c"].to_numpy(), out["era5_dewpoint_c"].to_numpy()
    )

    return out


def build_station_axis_weights(df):
    """Per-station NW axis weights, keyed by canonical station_id."""
    st = df.groupby("station_id")[["latitude", "longitude"]].first()
    w = cp.station_nw_axis_weights(
        st["latitude"].to_numpy(), st["longitude"].to_numpy(), centre=True
    )
    return dict(zip(st.index.tolist(), w.tolist()))


# ---------------------------------------------------------------------------
# TARGETS
# ---------------------------------------------------------------------------

def build_targets(df):
    """
    Direct multi-step targets. NEVER recursive: each horizon is an independent
    shift of the observed series, so error growth stays sub-linear in h instead
    of compounding (Golden Rule 1, legacy Mistake 4).
    """
    df = df.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    for pol in POLLUTANTS:
        src = df.groupby("station_id", sort=False)[TARGET_COL[pol]]
        for h in HORIZONS:
            df[f"target_{pol}_h{h}"] = src.shift(-h)
    return df


# ---------------------------------------------------------------------------
# VERIFICATION
# ---------------------------------------------------------------------------

def verify_causality(df_raw, weights_by_station, cut_quantile=0.75):
    """
    Truncation causality test.

    Compute every static feature twice: once on the full frame, once on a frame
    truncated at a cut timestamp. On the shared prefix the values must be
    BIT-IDENTICAL. If any feature reads the future, the truncated run cannot
    reproduce it and this assertion fires.

    This is the mechanical proof behind Golden Rule 5 and legacy Mistake 2, and
    it is far stronger than eyeballing a merge direction.
    """
    ts = df_raw["timestamp_utc"]
    cut = ts.quantile(cut_quantile, interpolation="nearest")
    keep = ts <= cut

    full = compute_static_features(df_raw, weights_by_station)
    trunc = compute_static_features(df_raw.loc[keep], weights_by_station)

    a = full.loc[keep, STATIC_NEW_FEATURES].reset_index(drop=True)
    b = trunc[STATIC_NEW_FEATURES].reset_index(drop=True)

    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch in causality test: {a.shape} vs {b.shape}")

    bad = []
    for col in STATIC_NEW_FEATURES:
        x = a[col].to_numpy(dtype=float)
        y = b[col].to_numpy(dtype=float)
        same = np.isclose(x, y, rtol=0.0, atol=0.0, equal_nan=True)
        if not same.all():
            bad.append((col, int((~same).sum())))
    if bad:
        raise AssertionError(
            f"CAUSALITY VIOLATION - features differ under time truncation: {bad}"
        )
    return {"cut_timestamp": str(cut), "rows_checked": int(keep.sum()),
            "features_checked": len(STATIC_NEW_FEATURES)}


def verify_targets(df, station_col="station_id", ts_col="timestamp_utc"):
    """
    Targets must be pure: the final h rows of every station series must be NaN,
    and the shift must be exactly the horizon. Any smoothing or imputation would
    fill those tail rows and break this test (Golden Rule 3, legacy Mistake 3).
    """
    report = {}
    d = df.sort_values([station_col, ts_col])
    n_stations = d[station_col].nunique()
    for pol in POLLUTANTS:
        for h in HORIZONS:
            col = f"target_{pol}_h{h}"
            tail_nan = 0
            for _, grp in d.groupby(station_col, sort=False):
                if grp[col].tail(h).isna().all():
                    tail_nan += 1
            if tail_nan != n_stations:
                raise AssertionError(
                    f"{col}: only {tail_nan}/{n_stations} stations show a full "
                    f"NaN tail of length {h} - target appears smoothed or imputed"
                )

            # exact-shift spot check on the station with the fewest gaps.
            # target(t) must equal the raw series at t + h: pandas shift(-h)
            # pulls the value h steps FORWARD, so the lookup index is t + h.
            grp = max(d.groupby(station_col, sort=False),
                      key=lambda kv: kv[1][TARGET_COL[pol]].notna().sum())[1]
            src = grp.set_index(ts_col)[TARGET_COL[pol]].sort_index()
            tgt = grp.set_index(ts_col)[col].dropna().sort_index()
            if len(tgt) > 200:
                shifted = src.reindex(tgt.index + pd.Timedelta(hours=h))
                same = np.isclose(shifted.to_numpy(dtype=float),
                                  tgt.to_numpy(dtype=float),
                                  rtol=0.0, atol=1e-12, equal_nan=True)
                if not same.all():
                    raise AssertionError(
                        f"{col}: values are not an exact +{h}h shift "
                        f"({int((~same).sum())} of {len(tgt)} mismatches)"
                    )
            report[col] = {"valid": int(df[col].notna().sum()),
                           "coverage_pct": round(100.0 * df[col].notna().mean(), 3)}
    return report


def verify_physics(df):
    """Physical-consistency assertions on the constructed features."""
    def corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 100:
            return float("nan")
        return float(np.corrcoef(x[m], y[m])[0, 1])

    pm = df["PM2.5_ground"].to_numpy(dtype=float)
    blh = df["era5_boundary_layer_height"].to_numpy(dtype=float)
    ws = df["era5_wind_speed"].to_numpy(dtype=float)

    checks = {}
    checks["corr(aod_proxy, PM2.5) [weak by construction]"] = corr(
        df["aod_proxy"].to_numpy(float), pm)
    checks["corr(aod_proxy, PM2.5*BLH) [definitional]"] = corr(
        df["aod_proxy"].to_numpy(float), pm * blh)
    checks["corr(aod_proxy, BLH)"] = corr(df["aod_proxy"].to_numpy(float), blh)
    checks["corr(clearness_index, SSRD)"] = corr(
        df["clearness_index"].to_numpy(float),
        df["era5_solar_radiation_w_m2"].to_numpy(float))
    checks["corr(clearness_index, cos_zenith) [want ~0]"] = corr(
        df["clearness_index"].to_numpy(float),
        cp.solar_zenith_cos(df["timestamp_utc"], df["latitude"].to_numpy(),
                            df["longitude"].to_numpy()))
    checks["corr(clearness_index, PM2.5) [aerosol signal]"] = corr(
        df["clearness_index"].to_numpy(float), pm)
    checks["corr(ITI, BLH)"] = corr(
        df["inversion_trap_index"].to_numpy(float), blh)
    checks["corr(ITI, wind_speed)"] = corr(
        df["inversion_trap_index"].to_numpy(float), ws)
    checks["corr(nw_transport_flux, wind_speed)"] = corr(
        df["nw_transport_flux"].to_numpy(float), ws)
    checks["corr(city_pm25_mean, PM2.5)"] = corr(
        df["city_pm25_mean"].to_numpy(float), pm)

    # Required signs, derived from the governing equations in the master plan.
    #
    # NOTE ON aod_proxy: it is a COLUMN quantity, tau ~ PM2.5 * BLH. Because BLH
    # spans 50-3842 m while PM2.5 spans ~1-893 ug/m3, and because the two are
    # NEGATIVELY correlated (r = -0.25: deep daytime mixing ventilates the surface
    # while loading a thicker column), aod_proxy tracks BLH far more strongly
    # than it tracks surface PM2.5. Asserting a strong aod~PM2.5 correlation
    # would therefore be physically WRONG, so the assertions below test the
    # actual governing relations instead of an intuitive-but-false one.
    if not checks["corr(aod_proxy, PM2.5*BLH) [definitional]"] > 0.99:
        raise AssertionError("aod_proxy must be a faithful column-mass measure")
    if not checks["corr(aod_proxy, BLH)"] > 0.3:
        raise AssertionError("aod_proxy must rise with BLH (thicker column)")

    # Clearness index: assert the properties that ARE structural, not an aerosol
    # sensitivity that is an empirical question. Because Kt normalises SSRD by
    # top-of-atmosphere insolation, it must be very nearly ORTHOGONAL to solar
    # geometry - if it were not, the normalisation would be broken. Its mean must
    # also land in the physically sensible transmittance band, and it must only
    # be defined in daylight.
    kt = df["clearness_index"].to_numpy(float)
    kt_def = float(np.isfinite(kt).mean())
    if not (0.0 <= np.nanmin(kt) and np.nanmax(kt) <= 1.2):
        raise AssertionError("clearness index outside [0, 1.2]")
    if abs(checks["corr(clearness_index, cos_zenith) [want ~0]"]) > 0.15:
        raise AssertionError(
            "clearness index still carries solar geometry - normalisation broken")
    if not (0.30 <= float(np.nanmean(kt)) <= 0.80):
        raise AssertionError(
            "clearness index mean %.3f is outside the plausible transmittance band"
            % float(np.nanmean(kt)))
    if not (0.30 <= kt_def <= 0.60):
        raise AssertionError(
            "clearness index defined for %.3f of hours - expected daylight only"
            % kt_def)

    if not checks["corr(ITI, BLH)"] < -0.3:
        raise AssertionError("trapping index must fall as BLH deepens")
    if not checks["corr(ITI, wind_speed)"] < -0.3:
        raise AssertionError("trapping index must fall as wind strengthens")
    if not checks["corr(nw_transport_flux, wind_speed)"] > 0.3:
        raise AssertionError("NW flux must co-vary with wind speed")
    if not checks["corr(city_pm25_mean, PM2.5)"] > 0.5:
        raise AssertionError("city-wide PM2.5 must co-vary with station PM2.5")

    aod = df["aod_proxy"].to_numpy(float)
    if not (np.nanmin(aod) >= 0.0 and np.nanmax(aod) <= 5.0):
        raise AssertionError("aod_proxy outside [0, 5]")
    lag = df["transport_lag_hours"].to_numpy(float)
    if not (np.nanmin(lag) >= 6.0 and np.nanmax(lag) <= 72.0):
        raise AssertionError("transport_lag_hours outside [6, 72]")

    chk = {}
    for col in STATIC_NEW_FEATURES:
        v = df[col].to_numpy(float)
        chk[col] = {"unique": int(pd.Series(v).nunique()),
                    "nan_pct": round(100.0 * float(np.isnan(v).mean()), 3),
                    "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
        if chk[col]["unique"] < 10:
            raise AssertionError(
                f"{col} is effectively constant ({chk[col]['unique']} values)")

    return checks, chk


# ---------------------------------------------------------------------------
# CATALOG + SCHEMA
# ---------------------------------------------------------------------------

BLOCK_OF = {}
for _c in cp.COUPLED_FEATURES:
    BLOCK_OF[_c] = "1_coupled_physics"
for _c in cp.CROSS_SECTIONAL_FEATURES:
    BLOCK_OF[_c] = "3_cross_sectional"
for _c in cp.STABILITY_CANDIDATES:
    BLOCK_OF[_c] = "candidate_stability"
for _c in cp.TARGET_TIME_FEATURES:
    BLOCK_OF[_c] = "2_deterministic_future"

FORMULA_OF = {
    "aod_proxy": "3*Q_ext*(PM2.5*1e-9)*BLH/(4*rho_p*r_eff)",
    "clearness_index": "SSRD/(1361*max(cos(theta_z),0.1)) at observation time",
    "inversion_trap_index": "1/((BLH+20)*(wind_speed+0.5))",
    "nw_transport_flux": "max(0, 0.7071*(u10-v10))",
    "transport_lag_hours": "clip(300/(3.6*max(nw_transport_flux,0.5)), 6, 72)",
    "regional_burning_proxy": "sat_CO * sat_HCHO * exp(-((doy-308)^2)/(2*15^2))",
    "upwind_downwind_gradient": "sum_i w_i * sat_CO_i, w on NW axis, sum|w|=1",
    "city_pm25_mean": "leave-one-out cross-station mean of PM2.5 at t",
    "city_satco_mean": "leave-one-out cross-station mean of sat_CO at t",
    "pm25_nw_gradient": "sum_i w_i * PM2.5_i, w on NW axis",
    "dewpoint_depression": "era5_temperature_c - era5_dewpoint_c",
    "target_hour_sin": "sin(2*pi*IST_hour(t+h)/24)",
    "target_hour_cos": "cos(2*pi*IST_hour(t+h)/24)",
    "target_dow_sin": "sin(2*pi*dow(t+h)/7)",
    "target_dow_cos": "cos(2*pi*dow(t+h)/7)",
    "target_doy_sin": "sin(2*pi*doy(t+h)/365.25)",
    "target_doy_cos": "cos(2*pi*doy(t+h)/365.25)",
    "target_solar_zenith_cos": "sin(phi)sin(delta)+cos(phi)cos(delta)cos(H) at t+h",
    "target_ssrd_clim": "train-fold mean SSRD by (month, IST hour) at target time",
    "target_blh_clim": "train-fold mean BLH by (month, IST hour) at target time",
}

UNITS_OF = {
    "aod_proxy": "-", "clearness_index": "-",
    "inversion_trap_index": "s/m^2",
    "nw_transport_flux": "m/s", "transport_lag_hours": "h",
    "regional_burning_proxy": "mol^2/m^4", "upwind_downwind_gradient": "mol/m^2",
    "city_pm25_mean": "ug/m3", "city_satco_mean": "mol/m2",
    "pm25_nw_gradient": "ug/m3", "dewpoint_depression": "degC",
    "target_hour_sin": "-", "target_hour_cos": "-",
    "target_dow_sin": "-", "target_dow_cos": "-",
    "target_doy_sin": "-", "target_doy_cos": "-",
    "target_solar_zenith_cos": "-", "target_ssrd_clim": "W/m2",
    "target_blh_clim": "m",
}

NATIVE_NAN_EXACT = {
    "PM2.5_ground", "PM10_ground", "NO_ground", "NOx_ground", "NH3_ground",
    "SO2_ground", "CO_ground", "OZONE_ground", "sat_NO2", "sat_CO", "sat_HCHO",
    "sat_NO2_available", "sat_CO_available",
}


def _missing_strategy(name):
    if name in NATIVE_NAN_EXACT:
        return "native_nan"
    if "_lag_" in name or "_roll_" in name:
        return "native_nan"
    return "error"


def write_catalog_and_schema(legacy_names, feature_order):
    rows = []
    for name in legacy_names:
        rows.append({"name": name, "block": "0_legacy_58", "dtype": "float64",
                     "units": "see legacy schema",
                     "formula": "inherited verbatim from features_engineered.parquet"})
    for name in feature_order:
        rows.append({"name": name, "block": BLOCK_OF[name], "dtype": "float64",
                     "units": UNITS_OF.get(name, "-"),
                     "formula": FORMULA_OF.get(name, "")})
    pd.DataFrame(rows).to_csv(OUT_CATALOG, index=False)

    schema = {
        "model_version": "2.0.0",
        "system": "SIH26082_coupled_air_pollution_weather_forecast",
        "target_pollutants": ["PM25", "O3", "NO2"],
        "feature_count": len(legacy_names) + len(feature_order),
        "legacy_feature_count": len(legacy_names),
        "forecast_horizons_hours": HORIZONS,
        "input_frequency": "1h",
        "target_unit": "ug/m3",
        "transform": "log1p applied during training; expm1 for inference",
        "time_reference": ("timestamp_utc is UTC; all target_* features are in IST "
                           "(UTC+05:30) evaluated at the requested horizon"),
        "horizon_dependent_features": list(cp.TARGET_TIME_FEATURES),
        "features": [
            {"name": n, "dtype": "float64",
             "missing_strategy": _missing_strategy(n),
             "block": BLOCK_OF.get(n, "0_legacy_58")}
            for n in (legacy_names + feature_order)
        ],
    }
    with open(OUT_SCHEMA, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=1)
    return schema


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print("SIH26082 STAGE 1 - COUPLED FEATURE BUILD")
    print("=" * 78)

    legacy_names = load_legacy_feature_names()
    feature_order = STATIC_NEW_FEATURES + cp.TARGET_TIME_FEATURES
    print("legacy features      : %d" % len(legacy_names))
    print("new features         : %d (static %d + horizon-dependent %d)"
          % (len(feature_order), len(STATIC_NEW_FEATURES),
             len(cp.TARGET_TIME_FEATURES)))
    print("schema feature_count : %d" % (len(legacy_names) + len(feature_order)))

    print("\n[1/6] loading stores ...")
    engineered = pd.read_parquet(ENGINEERED_PATH)
    fused = pd.read_parquet(FUSED_PATH, columns=AUX_FROM_FUSED)
    print("      features_engineered %s" % (engineered.shape,))
    print("      fused aux           %s" % (fused.shape,))

    missing_legacy = [c for c in legacy_names if c not in engineered.columns]
    if missing_legacy:
        raise AssertionError("legacy store missing schema features: %s" % missing_legacy)
    print("      all %d legacy features present in the store" % len(legacy_names))

    print("\n[2/6] merging ...")
    base = engineered.merge(fused, on=["timestamp_utc", "station_id"],
                            how="left", validate="one_to_one")
    base = base.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    if base["latitude"].isna().any():
        raise AssertionError("latitude missing after merge")
    print("      merged %s, stations=%d" % (base.shape, base["station_id"].nunique()))

    weights_by_station = build_station_axis_weights(base)
    print("      NW axis weights (negative = upwind / Punjab side):")
    for k, v in sorted(weights_by_station.items(), key=lambda kv: kv[1]):
        print("        %-22s %+.4f" % (k, v))

    print("\n[3/6] computing coupled static features ...")
    df = compute_static_features(base, weights_by_station)

    print("\n[4/6] building direct multi-step targets ...")
    df = build_targets(df)

    print("\n[5/6] verifying ...")
    caus = verify_causality(base, weights_by_station)
    print("      causality ..... PASS  (%d features, %d rows, cut %s)"
          % (caus["features_checked"], caus["rows_checked"], caus["cut_timestamp"]))
    tgts = verify_targets(df)
    print("      targets ....... PASS  (%d targets, exact shifts, NaN tails intact)"
          % len(tgts))
    phys, chk = verify_physics(df)
    print("      physics ....... PASS")
    for k, v in phys.items():
        print("        %-42s %+.4f" % (k, v))

    print("\n[6/6] writing outputs ...")
    df.to_parquet(OUT_FEATURES, index=False, compression="snappy")
    write_catalog_and_schema(legacy_names, feature_order)
    print("      %s  (%.2f MB, %d x %d)"
          % (OUT_FEATURES, os.path.getsize(OUT_FEATURES) / 1e6, df.shape[0], df.shape[1]))
    print("      %s" % OUT_SCHEMA)
    print("      %s" % OUT_CATALOG)

    print("\n--- target coverage (valid rows) ---")
    for k in tgts:
        print("   %-22s %7d  (%6.2f%%)"
              % (k, tgts[k]["valid"], tgts[k]["coverage_pct"]))

    print("\n--- new feature health ---")
    for k in chk:
        v = chk[k]
        print("   %-26s uniq=%7d  nan=%6.2f%%  range=[%.4g, %.4g]"
              % (k, v["unique"], v["nan_pct"], v["min"], v["max"]))

    print("\nDONE in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
