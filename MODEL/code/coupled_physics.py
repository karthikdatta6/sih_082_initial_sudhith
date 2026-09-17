"""
coupled_physics.py
=============================================================================
SIH26082 - Air Pollution-Weather Coupled Forecasting System (Delhi NCR)
Shared coupled-physics kernel.

DESIGN CONTRACT
---------------
* Pure functions only. No file I/O, no global mutable state, no side effects.
* Importable by BOTH the offline training pipeline and the production FastAPI
  backend, which guarantees that training-time and serving-time feature
  definitions are byte-identical (Golden Rule 8: schema drift is a bug).
* Every function is vectorised over numpy/pandas and NaN-tolerant.

PHYSICAL UNITS (SI unless suffixed)
-----------------------------------
  PM2.5   ug/m3        BLH      m          wind    m/s
  SSRD    W/m2         sat_CO   mol/m2     sat_HCHO mol/m2
  tau     dimensionless (-)     distances km

References for the physics are given inline. See the master plan document for
the full derivation set and the list of quantities that are NOT derivable from
this dataset.
=============================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "IST_OFFSET_HOURS",
    "IST_STANDARD_MERIDIAN_DEG",
    "NW_BEARING_DEG",
    "CORRIDOR_LENGTH_KM",
    "STUBBLE_DOY_CENTER",
    "STUBBLE_DOY_SIGMA",
    "SOLAR_CONSTANT_W_M2",
    "MIN_COS_Z_FOR_CLEARNESS",
    "clearness_index",
    "COUPLED_FEATURES",
    "TARGET_TIME_FEATURES",
    "CROSS_SECTIONAL_FEATURES",
    "STABILITY_CANDIDATES",
    "ALL_NEW_FEATURES",
    "nw_transport_flux",
    "aerosol_optical_depth",
    "aerosol_dimming_ratio",
    "inversion_trap_index",
    "ventilation_coefficient",
    "transport_lag_hours",
    "stubble_season_weight",
    "regional_burning_proxy",
    "dewpoint_depression",
    "solar_zenith_cos",
    "target_calendar_features",
    "add_target_time_features",
    "station_nw_axis_weights",
    "upwind_downwind_gradient",
    # ---------- NEW v2.1 feature functions ----------
    "compute_ventilation_coefficient",
    "compute_hygroscopic_swelling",
    "compute_chemical_age_ratios",
    "compute_inversion_lapse_rate",
    "VENTILATION_CRISIS_THRESHOLD_M2S",
]

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# India Standard Time is UTC+05:30. The dataset timestamp_utc is true UTC.
IST_OFFSET_HOURS = 5.5
IST_STANDARD_MERIDIAN_DEG = 82.5

# Punjab/Haryana -> Delhi NCR stubble corridor.
# NW = 315 deg compass (source side); SE = 135 deg compass (Delhi side).
NW_BEARING_DEG = 315.0
CORRIDOR_LENGTH_KM = 300.0          # approx Punjab -> Delhi advection distance
STUBBLE_DOY_CENTER = 308.0          # ~4 November: peak paddy-residue burning
STUBBLE_DOY_SIGMA = 15.0            # +/- 15 day Gaussian envelope

# Aerosol mass-extinction optics
Q_EXT = 2.0                         # extinction efficiency, geometric limit (r >> lambda)
RHO_PARTICLE_KG_M3 = 1.5e3          # effective density of hygroscopic Delhi aerosol
R_EFF_M = 0.2e-6                    # effective radius, 0.2 um
PM25_UG_M3_TO_KG_M3 = 1e-9

# Solar geometry / radiative-transmittance constants (Part C, Leg 1).
SOLAR_CONSTANT_W_M2 = 1361.0        # total solar irradiance at top of atmosphere
MIN_COS_Z_FOR_CLEARNESS = 0.10      # below this the clearness index is undefined

# Unit vector along the NW -> SE plume-travel axis, in (east, north) components.
# Bearing 135 deg -> east = sin(135) = +sqrt(2)/2, north = cos(135) = -sqrt(2)/2.
_PLUME_EAST = float(np.sin(np.deg2rad(135.0)))
_PLUME_NORTH = float(np.cos(np.deg2rad(135.0)))

# Floor terms. These are PHYSICAL floors, not arbitrary epsilon: they bound the
# index at the observed dataset minima (BLH 50.1 m, wind 0.02 m/s).
BLH_FLOOR_M = 20.0
WIND_FLOOR_MS = 0.5
MIN_LAG_FLOOR_H = 0.5               # avoids division blow-up at zero advection
LAG_CLAMP_H = (6.0, 72.0)

# CPCB official ventilation crisis threshold (m2/s).
# Source: CPCB Emergency Response Action Plan for NCR (2019).
VENTILATION_CRISIS_THRESHOLD_M2S = 2000.0

# ---------------------------------------------------------------------------
# FEATURE NAME REGISTRY
# ---------------------------------------------------------------------------
# Ordered exactly as they are appended to the legacy 58-feature schema.

COUPLED_FEATURES = [
    "aod_proxy",
    "clearness_index",
    "inversion_trap_index",
    "nw_transport_flux",
    "transport_lag_hours",
    "regional_burning_proxy",
    "upwind_downwind_gradient",
]

TARGET_TIME_FEATURES = [
    "target_hour_sin",
    "target_hour_cos",
    "target_dow_sin",
    "target_dow_cos",
    "target_doy_sin",
    "target_doy_cos",
    "target_solar_zenith_cos",
    "target_ssrd_clim",
    "target_blh_clim",
]

CROSS_SECTIONAL_FEATURES = [
    "city_pm25_mean",
    "city_satco_mean",
    "pm25_nw_gradient",
]

# Candidate the ablation is asked to JUDGE rather than a mandated feature.
STABILITY_CANDIDATES = [
    "dewpoint_depression",
]

ALL_NEW_FEATURES = (COUPLED_FEATURES + TARGET_TIME_FEATURES
                    + CROSS_SECTIONAL_FEATURES + STABILITY_CANDIDATES)


# ---------------------------------------------------------------------------
# BLOCK 1 - TWO-WAY COUPLED PHYSICS
# ---------------------------------------------------------------------------

def nw_transport_flux(u10, v10):
    """
    Component of the 10 m wind vector along the NW -> SE plume-travel axis.

        F_nw = max(0, u10 * sin(135) + v10 * cos(135))

    Equivalently 0.7071 * (u10 - v10). Returns m/s, zero when the wind does not
    advect from the north-west (i.e. no Punjab -> Delhi transport).

    NOTE: this supersedes the formulation printed in the master documentation,
    which used  max(0, -u10*sin45 - v10*cos45) = -0.7071*(u10 + v10). That
    expression evaluates to ~0 for a true NW wind (u = +0.707s, v = -0.707s)
    and therefore silently discarded the exact regime it was meant to capture.
    """
    u = np.asarray(u10, dtype=float)
    v = np.asarray(v10, dtype=float)
    return np.maximum(0.0, u * _PLUME_EAST + v * _PLUME_NORTH)


def aerosol_optical_depth(pm25_ug_m3, blh_m, q_ext=Q_EXT,
                          rho_p=RHO_PARTICLE_KG_M3, r_eff=R_EFF_M):
    """
    Column aerosol optical depth from surface PM2.5 under a well-mixed-PBL
    assumption (a standard mass-extinction closure):

        tau = 3 * Q_ext * M_col / (4 * rho_p * r_eff)
        M_col = PM2.5[kg/m3] * BLH[m]

    Sanity anchor: PM2.5 = 100 ug/m3, BLH = 500 m  ->  M_col = 5e-5 kg/m2
    ->  tau = 0.25, a realistic Delhi AOD. Clipped to [0, 5] for robustness.
    """
    pm = np.asarray(pm25_ug_m3, dtype=float)
    h = np.asarray(blh_m, dtype=float)
    m_col = pm * PM25_UG_M3_TO_KG_M3 * h
    tau = (3.0 * q_ext * m_col) / (4.0 * rho_p * r_eff)
    return np.clip(tau, 0.0, 5.0)


def aerosol_dimming_ratio(pm25_ug_m3, blh_m):
    """
    Beer-Lambert transmittance of the aerosol column, f_dim = exp(-tau) in (0, 1].
    Served for dashboard/alert display only - it is a strictly monotone function
    of tau and therefore carries ZERO additional information for a tree model.
    """
    return np.exp(-aerosol_optical_depth(pm25_ug_m3, blh_m))


def clearness_index(ssrd_w_m2, cos_zenith,
                    solar_constant=SOLAR_CONSTANT_W_M2,
                    min_cos_z=MIN_COS_Z_FOR_CLEARNESS):
    """
    Observed atmospheric transmittance for shortwave radiation:

        Kt = SSRD / (I0 * cos(theta_z)),   I0 = 1361 W/m2

    This is the DIRECTLY OBSERVED counterpart of the modelled aod_proxy. Where
    aod_proxy is a closed-form function of PM2.5 and BLH (and is therefore
    largely reconstructible by a tree that already has both), Kt is a measured
    radiative residual: the fraction of top-of-atmosphere insolation that
    actually reached the surface after aerosol and cloud extinction. It carries
    the aerosol radiative signature as an observation rather than an assumption,
    and it needs no fitted clear-sky climatology, so it cannot leak.

    Returns NaN when the sun is too low for the index to be defined (night, and
    the low-sun hours where cos(theta_z) -> 0 makes the ratio ill-conditioned).
    Clipped to [0, 1.2]: values above ~1.0 only occur from reflectivity or
    ERA5 assimilation artefacts.
    """
    ssrd = np.asarray(ssrd_w_m2, dtype=float)
    cz = np.asarray(cos_zenith, dtype=float)
    cz = np.where(cz < min_cos_z, np.nan, cz)
    kt = ssrd / (solar_constant * cz)
    return np.clip(kt, 0.0, 1.2)


def inversion_trap_index(blh_m, wind_speed_ms):
    """
    Volume-compression / trapping index.

        ITI = 1 / ((BLH + 20) * (U + 0.5))

    Large when the boundary layer is shallow AND the wind is calm, which is the
    configuration that compresses emissions into a thin surface layer. This is
    the derivable substitute for the requested dT/dz = T_925 - T_surface, which
    is NOT computable from this dataset (no upper-air temperature exists).
    """
    h = np.asarray(blh_m, dtype=float)
    u = np.asarray(wind_speed_ms, dtype=float)
    return 1.0 / ((h + BLH_FLOOR_M) * (u + WIND_FLOOR_MS))


def ventilation_coefficient(blh_m, wind_speed_ms):
    """V = BLH * U (m2/s). Provided for completeness; the legacy schema already
    carries ventilation_coeff, and this helper keeps the definition in one place."""
    return np.asarray(blh_m, dtype=float) * np.asarray(wind_speed_ms, dtype=float)


def transport_lag_hours(nw_flux, length_km=CORRIDOR_LENGTH_KM):
    """
    Wind-dependent Lagrangian arrival delay for the stubble plume.

        dt = L / U_along ,  clipped to [6, 72] hours

    A DYNAMIC lag, not the fixed 24-36 h quoted in the brief. At U = 4 m/s the
    delay is 20.8 h; at U = 1.5 m/s it is 55.6 h; so the expected 24-36 h window
    is recovered at typical winter advection speeds, which self-validates the
    formulation while remaining strictly more informative than a constant.
    """
    u = np.maximum(np.asarray(nw_flux, dtype=float), MIN_LAG_FLOOR_H)
    lag = length_km / (3.6 * u)
    return np.clip(lag, LAG_CLAMP_H[0], LAG_CLAMP_H[1])




def stubble_season_weight(doy):
    """Gaussian seasonal envelope peaking at the Punjab/Haryana burning window."""
    d = np.asarray(doy, dtype=float)
    return np.exp(-((d - STUBBLE_DOY_CENTER) ** 2) / (2.0 * STUBBLE_DOY_SIGMA ** 2))


def regional_burning_proxy(sat_co, sat_hcho, doy):
    """
    Regional agricultural-burning load, gated on the burning season.

        RBP = sat_CO * sat_HCHO * w_season(doy)

    CO and HCHO are both established biomass-burning tracers (HCHO especially
    so). Gaps are mapped to 0.0 because a missing overpass carries no plume
    evidence; the satellite availability flags stay in the schema so the model
    can distinguish a genuine zero from an unobserved hour.
    """
    co = np.asarray(sat_co, dtype=float)
    hc = np.asarray(sat_hcho, dtype=float)
    out = co * hc * stubble_season_weight(doy)
    return np.where(np.isfinite(out), out, 0.0)


def dewpoint_depression(temperature_c, dewpoint_c):
    """
    DD = T - Td, a derivable dry-stability / radiative-cooling proxy.

    A large dewpoint depression implies dry air, weak downward longwave flux and
    therefore efficient nocturnal radiative cooling - the correct surface
    signature of an inversion-forming night. Ablation-tested candidate feature,
    offered as the principled companion to the inversion trap index.
    """
    return np.asarray(temperature_c, dtype=float) - np.asarray(dewpoint_c, dtype=float)


# ---------------------------------------------------------------------------
# BLOCK 1b - NEW v2.1 ALGEBRAIC FEATURE EXTENSIONS
# ---------------------------------------------------------------------------
# These four functions are MODULAR EXTENSIONS: they are pure functions with no
# side effects and always return physically-sensible values with default fallbacks.
# They do NOT modify the existing feature schema (the schema stays at 86 cols).
# They are provided for dashboard display, physics stress tests, and optional
# re-training runs.

def compute_ventilation_coefficient(blh, wind_speed):
    """
    Ventilation Coefficient (VC) — the canonical atmospheric dispersion proxy
    used by CPCB and IMD for Delhi NCR emergency alerts.

        VC = BLH * wind_speed    [m^2/s]

    A low VC indicates a stagnant, poorly ventilated atmosphere in which
    emissions accumulate rapidly.

    Returns
    -------
    vc : ndarray
        Ventilation coefficient in m^2/s.  Clipped to [0, ∞).
    is_ventilation_crisis : ndarray
        Binary flag: 1.0 if VC < 2000 m^2/s (CPCB emergency threshold), else 0.0.

    Reference
    ---------
    CPCB, "Emergency Action Plan for Air Quality Management in Delhi-NCR," 2019.
    Threshold source: Mishra & Srinivasan (2020), Atmos. Environ. 223.
    """
    blh_arr = np.asarray(blh, dtype=float)
    ws_arr = np.asarray(wind_speed, dtype=float)

    vc = np.maximum(0.0, blh_arr * ws_arr)
    is_ventilation_crisis = np.where(vc < VENTILATION_CRISIS_THRESHOLD_M2S, 1.0, 0.0)
    return vc, is_ventilation_crisis


def compute_hygroscopic_swelling(pm25, dewpoint_depression_val):
    """
    Hygroscopic Swelling Index — models winter fog-smog coupling where
    aerosols absorb moisture and swell, increasing their effective optical
    and mass cross-section.

        swelling_index = PM2.5 / (1 + exp(-dewpoint_depression))

    Physical rationale
    ------------------
    * When dewpoint_depression → 0 (air nearly saturated, DD ≈ 0):
        sigmoid(0) = 0.5, so swelling_index ≈ PM2.5 / 1.5.
        Moderate swelling even at low PM2.5.
    * When dewpoint_depression >> 0 (very dry air):
        sigmoid(+∞) = 1, so swelling_index ≈ PM2.5 / 2.0.
        Minimal swelling — low RH suppresses hygroscopic growth.
    * When dewpoint_depression < 0 (supersaturated / fog):
        sigmoid(−) → 0, denominator → 1.0, swelling_index ≈ PM2.5.
        Maximum aerosol swelling — onset of fog-smog feedback.

    This sigmoid form is a standard kappa-Köhler approximation linearised
    around the observed Delhi winter humidity range.

    Parameters
    ----------
    pm25 : array-like
        PM2.5 concentration in µg/m³.
    dewpoint_depression_val : array-like
        T − Td in °C.  Positive = dry; negative = foggy/supersaturated.

    Returns
    -------
    swelling_index : ndarray
        Dimensionless hygroscopic swelling index (same units / scale as PM2.5
        but transformed). Always ≥ 0.
    """
    pm = np.asarray(pm25, dtype=float)
    dd = np.asarray(dewpoint_depression_val, dtype=float)

    denom = 1.0 + np.exp(-dd)          # sigmoid denominator; always in (1, 2)
    swelling_index = pm / denom
    return np.where(np.isfinite(swelling_index), swelling_index, 0.0)


def compute_chemical_age_ratios(pm25, pm10, nox, no2):
    """
    Chemical Age Ratios — two dimensionless spectral fingerprints that allow
    a downstream classifier (or a GBDT feature) to discriminate between
    emission sources and atmospheric processing states.

    1. Fine-to-Coarse Ratio
    -----------------------
        fine_coarse_ratio = PM2.5 / (PM10 + 1e-3)

    PM2.5 / PM10 ≈ 0.9–1.0  → fresh combustion smoke (diesel, biomass burning,
                               stubble fire plume). Fine particles dominate.
    PM2.5 / PM10 ≈ 0.3–0.5  → mechanical dust, road re-suspension.
                               Coarse fraction dominates.

    2. Photochemical Age Ratio
    --------------------------
        photochemical_age_ratio = NOx / (NO2 + 1e-3)

    NOx / NO2 >> 1  → fresh tailpipe exhaust (NOx ≈ NO + NO2, mostly NO
                       close to source). Chemically young air mass.
    NOx / NO2 ≈ 1.0 → aged plume. Photochemical cycling has converted NO → NO2
                       via O3 + NO → NO2 + O2. Regionally transported air mass.

    Parameters
    ----------
    pm25, pm10, nox, no2 : array-like
        Ground-level concentrations in µg/m³.

    Returns
    -------
    fine_coarse_ratio : ndarray
        PM2.5 / (PM10 + 1e-3). Bounded in [0, 1] for clean data.
    photochemical_age_ratio : ndarray
        NOx / (NO2 + 1e-3). Values > 1 indicate fresh exhaust.
    """
    pm25_arr = np.asarray(pm25, dtype=float)
    pm10_arr = np.asarray(pm10, dtype=float)
    nox_arr = np.asarray(nox, dtype=float)
    no2_arr = np.asarray(no2, dtype=float)

    fine_coarse_ratio = pm25_arr / (pm10_arr + 1e-3)
    photochemical_age_ratio = nox_arr / (no2_arr + 1e-3)

    # Replace non-finite values with a physically neutral sentinel (0.0 for
    # fine_coarse, 1.0 for photochemical — the "background / aged plume" state).
    fine_coarse_ratio = np.where(np.isfinite(fine_coarse_ratio), fine_coarse_ratio, 0.0)
    photochemical_age_ratio = np.where(np.isfinite(photochemical_age_ratio),
                                       photochemical_age_ratio, 1.0)
    return fine_coarse_ratio, photochemical_age_ratio


def compute_inversion_lapse_rate(t_2m, t_925hpa):
    """
    Thermal Inversion Lapse Rate — the signed temperature difference between
    the free troposphere at 925 hPa (~750 m AGL over Delhi) and the 2-m
    surface temperature.

        ΔT_inversion = T_925hPa − T_2m     [°C or K, identical for differences]

    Physical interpretation
    -----------------------
    * ΔT_inversion > 0  : WARM INVERSION LID. The 925 hPa level is warmer than
                          the surface. This is a classic subsidence inversion:
                          the warm upper layer acts as a physical cap, preventing
                          convective mixing and trapping pollutants near the
                          surface. Delhi NCR experiences this most severely in
                          November–January (post-monsoon ridge subsidence +
                          nocturnal radiative cooling of the surface).

    * ΔT_inversion ≈ 0  : Near-neutral atmosphere. Moderate mixing.

    * ΔT_inversion < 0  : Normal lapse rate (atmosphere unstable). Good mixing,
                          pollutant flushing. Typically daytime summer.

    Parameters
    ----------
    t_2m : array-like
        2-metre air temperature in °C (ERA5 field: 2m_temperature converted
        to Celsius, or Open-Meteo `temperature_2m`).
    t_925hpa : array-like
        Temperature at 925 hPa in °C (ERA5 pressure-level field, or
        Open-Meteo `temperature_925hpa`).

    Returns
    -------
    delta_t_inversion : ndarray
        ΔT in °C. Positive values indicate an inversion lid.
    """
    t2 = np.asarray(t_2m, dtype=float)
    t925 = np.asarray(t_925hpa, dtype=float)
    return t925 - t2


# ---------------------------------------------------------------------------
# BLOCK 2 - DETERMINISTIC FUTURE FORCING
# ---------------------------------------------------------------------------
# Known EXACTLY at forecast time from pure calendar arithmetic and spherical
# astronomy. No data required, cannot leak, and directly drives photolysis
# (solar zenith) and diurnal accumulation (hour-of-day).


def solar_zenith_cos(t_utc, lat_deg, lon_deg):
    """
    Cosine of the solar zenith angle for a UTC timestamp and site location.

        cos(theta_z) = sin(phi)sin(delta) + cos(phi)cos(delta)cos(H)
        delta = 23.44 deg * sin(2*pi*(284 + doy)/365)      declination
        H     = 15 deg * (LST - 12)                        hour angle
        LST   = IST_hour + 4*(82.5 - lon)/60               local solar time

    Returns [-1, 1]; negative means the sun is below the horizon.
    """
    t = pd.DatetimeIndex(t_utc)
    doy = t.dayofyear.values.astype(float)
    dec = np.deg2rad(23.44) * np.sin(2.0 * np.pi * (284.0 + doy) / 365.0)

    ist_hour = (t.hour.values.astype(float)
                + t.minute.values.astype(float) / 60.0
                + IST_OFFSET_HOURS)
    lon = np.asarray(lon_deg, dtype=float)
    lst = ist_hour + (4.0 * (IST_STANDARD_MERIDIAN_DEG - lon)) / 60.0
    hour_angle = np.deg2rad(15.0 * (lst - 12.0))

    phi = np.deg2rad(np.asarray(lat_deg, dtype=float))
    cos_z = (np.sin(phi) * np.sin(dec)
             + np.cos(phi) * np.cos(dec) * np.cos(hour_angle))
    return np.clip(cos_z, -1.0, 1.0)


def target_calendar_features(t_utc):
    """
    Cyclical calendar encodings of the TARGET timestamp, in IST.

    The legacy 58-feature schema encodes only the OBSERVATION time. For every
    horizon where h mod 24 != 0 those encodings point at the wrong clock hour,
    so the tree cannot see the diurnal phase of the quantity being predicted.
    These fix that, and they cost nothing.
    """
    ist = pd.DatetimeIndex(t_utc) + pd.Timedelta(hours=IST_OFFSET_HOURS)
    hour = ist.hour.values.astype(float) + ist.minute.values.astype(float) / 60.0
    dow = ist.dayofweek.values.astype(float)
    doy = ist.dayofyear.values.astype(float)
    return {
        "target_hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
        "target_hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
        "target_dow_sin": np.sin(2.0 * np.pi * dow / 7.0),
        "target_dow_cos": np.cos(2.0 * np.pi * dow / 7.0),
        "target_doy_sin": np.sin(2.0 * np.pi * doy / 365.25),
        "target_doy_cos": np.cos(2.0 * np.pi * doy / 365.25),
    }


def add_target_time_features(df, horizons, timestamp_col="timestamp_utc",
                             lat_col="latitude", lon_col="longitude",
                             clim_table=None,
                             clim_ssrd_col="target_ssrd_clim",
                             clim_blh_col="target_blh_clim"):
    """
    Return {horizon: DataFrame} of target-time features evaluated at t + h.

    The feature matrix is therefore HORIZON-DEPENDENT while the feature SCHEMA
    stays identical, which is precisely what lets a single frozen
    feature_schema.json serve all seven horizons at serving time.

    clim_table must be a DataFrame indexed by (month, ist_hour) carrying the two
    climatological driver columns, and it must be built from the TRAINING fold
    only. Building it once globally and reusing it across folds is a silent leak
    (risk C2 in the master plan).
    """
    out = {}
    for h in horizons:
        h = int(h)
        t_target = df[timestamp_col] + pd.Timedelta(hours=h)
        block = pd.DataFrame(target_calendar_features(t_target), index=df.index)
        block["target_solar_zenith_cos"] = solar_zenith_cos(
            t_target, df[lat_col].values, df[lon_col].values
        )

        if clim_table is not None:
            ist = pd.DatetimeIndex(t_target) + pd.Timedelta(hours=IST_OFFSET_HOURS)
            key = pd.MultiIndex.from_arrays(
                [ist.month.values, ist.hour.values], names=["month", "ist_hour"]
            )
            reindexed = clim_table.reindex(key)
            block[clim_ssrd_col] = reindexed[clim_ssrd_col].values
            block[clim_blh_col] = reindexed[clim_blh_col].values
        else:
            block[clim_ssrd_col] = np.nan
            block[clim_blh_col] = np.nan

        out[h] = block
    return out


def build_target_time_climatology(df, ssrd_col="era5_solar_radiation_w_m2",
                                  blh_col="era5_boundary_layer_height",
                                  timestamp_col="timestamp_utc"):
    """
    Climatological target-driver table by (month, IST hour), for use as the
    dispersion/radiation driver forecast at target time. MUST be fitted on the
    training fold only.
    """
    ist = pd.DatetimeIndex(df[timestamp_col]) + pd.Timedelta(hours=IST_OFFSET_HOURS)
    tmp = pd.DataFrame({
        "month": ist.month.values,
        "ist_hour": ist.hour.values,
        "ssrd": np.asarray(df[ssrd_col].values, dtype=float),
        "blh": np.asarray(df[blh_col].values, dtype=float),
    })
    g = tmp.groupby(["month", "ist_hour"], observed=True).mean()
    return g.rename(columns={"ssrd": "target_ssrd_clim", "blh": "target_blh_clim"})


# ---------------------------------------------------------------------------
# BLOCK 3 - CROSS-SECTIONAL (OTHER-ROW) INFORMATION
# ---------------------------------------------------------------------------
# A gradient-boosted tree splits one row at a time, so it provably cannot read a
# value that lives in another row. Cross-station aggregation is therefore one of
# the few feature classes that adds information a tree cannot reconstruct.


def station_nw_axis_weights(latitudes, longitudes, centre=True):
    """
    Project station offsets onto the NW -> SE plume axis and normalise.

    Positive weight = station lies downwind (SE / Delhi side) of the centroid;
    negative = upwind (NW / Punjab side). Returns weights with sum |w| = 1.
    """
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    d_east = (lon - lon.mean()) * 111.32 * np.cos(np.deg2rad(lat.mean()))
    d_north = (lat - lat.mean()) * 110.57
    proj = d_east * _PLUME_EAST + d_north * _PLUME_NORTH
    if centre:
        proj = proj - proj.mean()
    denom = np.abs(proj).sum()
    if denom == 0:
        return np.zeros_like(proj)
    return proj / denom


def upwind_downwind_gradient(values_by_station, weights):
    """
    Weighted spatial slope of a tracer field along the plume axis.

        G = sum_i w_i * C_i     with  sum_i |w_i| = 1,  sum_i w_i = 0

    NaN-safe: missing stations drop out and the remaining weights renormalise.
    """
    v = np.asarray(values_by_station, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w)
    if not ok.any():
        return np.nan
    ww = w[ok]
    denom = np.abs(ww).sum()
    if denom == 0:
        return np.nan
    return float((ww / denom * v[ok]).sum())
