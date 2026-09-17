# 🔌 03 · Outputs Catalog, System Integration & Future Roadmap

**Project Identity:** Smart India Hackathon 2026 — Problem Statement **SIH26082**  
**Title:** Operational Integration Blueprint & Next-Generation Architectural Roadmap  
**Scope:** Production Deployment, Multi-Modal Ingestion, REST API Serving & Long-Term Enhancements  
**Target Platform:** Python FastAPI Microservice + React 18 TypeScript GIS Dashboard  

---

## 1. Catalog of Generated Model Outputs

All artifacts produced during the data engineering, training, and validation phases are organized into standardized, production-ready files:

```
MODEL/
├── results/
│   ├── metrics_by_horizon.json       # Complete quantitative validation metrics across all 21 models
│   ├── ablation_table.csv            # Feature-block ablation study quantifying Delta R² and Delta RMSE
│   ├── feature_catalog.csv           # Master tabular data dictionary: names, blocks, dtypes, formulas
│   ├── feature_schema_v2.json        # Authoritative JSON schema defining the 78 input features
│   └── training_run.log              # Verbatim stdout/stderr execution log from the complete training run
│
└── code/
    ├── coupled_physics.py            # Reusable mathematical physics kernel functions
    ├── 01_build_coupled_features.py  # Stage 1: Feature engineering and direct multi-step targets
    ├── 02_train_coupled_ensemble.py  # Stage 2: LightGBM ensemble training & NNLS simplex stacking
    ├── 03_export_v2_bundles.py       # Stage 3: Production model.pkl packaging & golden test fixtures
    ├── 04_train_ensemble_v3.py       # Algorithmic diversity / XGBoost-CUDA exploration script
    └── requirements.txt              # Exact Python dependencies for reproducing the pipeline
```

### 1.1 Output File Specifications
1. **`metrics_by_horizon.json` (100.7 KB):**  
   Contains complete nested JSON performance dictionaries for every (pollutant $\times$ horizon) combination. Each record stores sample counts, $R^2$, Willmott index of agreement ($d$), RMSE, MAE, SMAPE, persistence baseline RMSE, skill score, NNLS simplex weights ($w_{L1}, w_{\text{Huber}}, w_{L2}$), and validation calibration slope ($a$) and intercept ($b$).
2. **`ablation_table.csv` (3.7 KB):**  
   Tabulates 37 systematic ablation experiments isolating the predictive value of the coupled physics block, cross-sectional station gradients, deterministic future calendar, and stability features.
3. **`feature_catalog.csv` (8.1 KB) & `feature_schema_v2.json` (11.1 KB):**  
   Defines the 78 canonical input features structured into 5 functional blocks:
   * `0_legacy_58`: 58 certified features inherited from the base architecture.
   * `1_coupled_physics`: 7 non-linear interaction features ($ITI, K_t$, corrected $NW$ flux, stubble lag proxy, upwind-downwind gradient).
   * `2_deterministic_future`: 9 target-time astronomical and calendar terms evaluated at $t+h$.
   * `3_cross_sectional`: 3 leave-one-out spatial citywide means and regional gradients.
   * `candidate_stability`: 1 thermodynamic surface saturation deficit (`dewpoint_depression`).

---

## 2. End-to-End System Integration Architecture

The trained models are designed to operate within a high-throughput, low-latency microservice architecture:

```
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   REAL-TIME INGESTION LAYER                                 │
 ├───────────────────────────────┬─────────────────────────────┬───────────────────────────────┤
 │ 1. CPCB CAAQMS Ground Telemetry│ 2. Open-Meteo / ERA5 NWP     │ 3. Sentinel-5P TROPOMI Orbit  │
 │ Live hourly ground monitors   │ Hourly forecast grids       │ Daily satellite overpass      │
 │ (PM2.5, PM10, NO2, O3, CO)    │ (BLH, SSRD, Temp, Wind)     │ (Tropospheric NO2, CO, HCHO)  │
 └───────────────────────────────┴─────────────────────────────┴───────────────────────────────┘
                                                 │
                                                 ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │                    FEATURE BUILDER & COUPLED PHYSICS KERNEL (Sub-2ms)                       │
 ├─────────────────────────────────────────────────────────────────────────────────────────────┤
 │ • Constructs 58 legacy features (lags, rolling stats, exponential moving averages)          │
 │ • Computes 7 coupled physics terms (Inversion Trap Index ITI, Clearness Index Kt, NW flux)   │
 │ • Evaluates 9 deterministic target-time features at t + h (astronomy, target-time calendar) │
 │ • Computes Leave-One-Out spatial means across active Delhi CAAQMS stations                   │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │                    PRODUCTION INFERENCE ENGINE (ModelService Singleton)                     │
 ├─────────────────────────────────────────────────────────────────────────────────────────────┤
 │ • In-memory loaded bundles: PM25, O3, NO2 across 7 horizons (+1h, +3h, +6h, +12h, 24h, 48h, 72h)
 │ • Evaluates 3 LightGBM tree models (L1, Huber, L2) per horizon                              │
 │ • Blends predictions via pre-fitted Non-Negative Least Squares (NNLS) simplex weights       │
 │ • Applies target stabilization: ŷ = expm1(clip(z, 0)) + Post-hoc linear bias calibration    │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │                    CPCB NAQI AIR QUALITY INDEX (AQI) ENGINE                                 │
 ├─────────────────────────────────────────────────────────────────────────────────────────────┤
 │ • Calculates sub-indices for PM2.5, O3, NO2, PM10, CO using official Indian CPCB breakpoints│
 │ • Computes composite AQI: AQI = max(Sub-Index_i) across all validated pollutants            │
 │ • Generates clinical health advisories and Graded Response Action Plan (GRAP) alerts        │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │                    INTERACTIVE PRESENTATION LAYER (React 18 + Leaflet GIS)                  │
 ├─────────────────────────────────────────────────────────────────────────────────────────────┤
 │ • Multi-horizon forecast slider (+1h to +72h)                                               │
 │ • Color-coded geospatial risk heatmap across Delhi NCR                                      │
 │ • Real-time inversion warning banner triggered when ITI > threshold                         │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Connection Guide

### 3.1 Connecting Models to the FastAPI Backend
The backend loads serialized model bundles into a singleton `ModelService` instance during startup to guarantee sub-10ms response times.

```python
# backend/app/services/model_service.py
import joblib
import json
import numpy as np
import os

class ModelService:
    def __init__(self, bundle_dir: str):
        self.bundles = {}
        for pollutant in ["PM25", "O3", "NO2"]:
            pollutant_path = os.path.join(bundle_dir, pollutant)
            with open(os.path.join(pollutant_path, "model.pkl"), "rb") as f:
                model_artifact = joblib.load(f)
            with open(os.path.join(pollutant_path, "feature_schema.json"), "r") as f:
                schema = json.load(f)
            self.bundles[pollutant] = {
                "models": model_artifact["models"],     # dict of {horizon: ensemble}
                "weights": model_artifact["weights"],   # NNLS weights per horizon
                "calib": model_artifact.get("calib", {}),
                "schema": schema["features"]
            }
            
    def predict(self, pollutant: str, horizon: int, feature_vector: np.ndarray) -> float:
        bundle = self.bundles[pollutant]
        model_pack = bundle["models"][horizon]
        weights = bundle["weights"][horizon]
        
        # 1. Evaluate LightGBM objective variants (log space)
        pred_l1 = model_pack["l1"].predict(feature_vector)[0]
        pred_huber = model_pack["huber"].predict(feature_vector)[0]
        pred_l2 = model_pack["l2"].predict(feature_vector)[0]
        
        # 2. Simplex convex blend
        z_blend = (weights["w_L1"] * pred_l1 + 
                   weights["w_Huber"] * pred_huber + 
                   weights["w_L2"] * pred_l2)
        
        # 3. Non-negative inverse transform
        y_raw = np.expm1(np.maximum(0.0, z_blend))
        
        # 4. Post-hoc linear calibration
        cal = bundle["calib"].get(horizon, {"a": 1.0, "b": 0.0})
        y_cal = cal["a"] * y_raw + cal["b"]
        
        return float(np.maximum(0.0, y_cal))
```

### 3.2 Feature Vector Assembly via `FeatureBuilder`
At serving time, `FeatureBuilder` accepts incoming telemetry, calls `coupled_physics.py`, and aligns the array to `feature_schema_v2.json`:

```python
# backend/app/utils/feature_builder.py
import numpy as np
import pandas as pd
from app.utils.coupled_physics import (
    compute_inversion_trap_index,
    compute_clearness_index,
    compute_nw_transport_flux,
    add_target_time_features
)

def build_serving_features(station_row: pd.Series, horizon: int, schema: list) -> np.ndarray:
    feat = dict(station_row)
    
    # Compute coupled physics
    feat["inversion_trap_index"] = compute_inversion_trap_index(
        blh=feat["era5_boundary_layer_height"],
        wind_speed=feat["era5_wind_speed"]
    )
    feat["clearness_index"] = compute_clearness_index(
        ssrd=feat["era5_solar_radiation_w_m2"],
        lat=feat["latitude"],
        lon=feat["longitude"],
        timestamp_utc=feat["timestamp_utc"]
    )
    feat["nw_transport_flux"] = compute_nw_transport_flux(
        u10=feat["era5_u10"],
        v10=feat["era5_v10"]
    )
    
    # Compute target-time deterministic astronomical features at t + horizon
    target_dt = pd.to_datetime(feat["timestamp_utc"]) + pd.Timedelta(hours=horizon)
    target_features = add_target_time_features(target_dt, feat["latitude"], feat["longitude"])
    feat.update(target_features)
    
    # Order strictly according to the frozen schema
    vector = np.array([feat.get(col, 0.0) for col in schema], dtype=np.float32)
    return vector.reshape(1, -1)
```

### 3.3 Computing Official CPCB NAQI AQI
The National Air Quality Index (NAQI) of India translates multi-pollutant concentrations into a single risk index ($0\text{--}500$) using piecewise linear interpolation:

$$I_p = I_{\text{low}} + \frac{I_{\text{high}} - I_{\text{low}}}{C_{\text{high}} - C_{\text{low}}} \cdot (C_p - C_{\text{low}})$$

$$\text{Composite AQI} = \max_{p} (I_p) \quad \text{subject to at least 3 pollutants being present (one being } PM_{2.5} \text{ or } PM_{10}\text{)}$$

#### CPCB Sub-Index Breakpoints:
| Category | AQI Range | $PM_{2.5}$ (24h avg) | $NO_2$ (24h avg) | $O_3$ (8h / 1h avg) | Health Statement |
|---|:---:|:---:|:---:|:---:|---|
| **Good** | 0 – 50 | 0 – 30 | 0 – 40 | 0 – 50 | Minimal Impact |
| **Satisfactory** | 51 – 100 | 31 – 60 | 41 – 80 | 51 – 100 | Minor breathing discomfort to sensitive people |
| **Moderate** | 101 – 200 | 61 – 90 | 81 – 180 | 101 – 168 | Breathing discomfort to people with lungs/asthma |
| **Poor** | 201 – 300 | 91 – 120 | 181 – 280 | 169 – 208 | Breathing discomfort to most people on prolonged exposure |
| **Very Poor** | 301 – 400 | 121 – 250 | 281 – 400 | 209 – 748 | Respiratory illness on prolonged exposure |
| **Severe** | 401 – 500 | 250+ | 400+ | 748+ | Healthy people affected, serious impacts on those with existing disease |

---

## 4. Production Runbook: How to Execute & Reproduce

All commands can be executed in under 15 minutes on any standard CPU workstation:

```bash
# 1. Navigate to the code directory
cd MODEL/code

# 2. Install lightweight dependencies
pip install -r requirements.txt

# 3. Build coupled physics features and direct multi-step targets (~9 seconds)
python 01_build_coupled_features.py

# 4. Train all 21 models + NNLS simplex stacking (~14 minutes on CPU)
python 02_train_coupled_ensemble.py

# 5. Export serialized model bundles and golden test fixtures (~12 seconds)
python 03_export_v2_bundles.py
```

---

## 5. Next-Generation Roadmap: How to Make the Current Model Even Better

While version `v2.0.0` achieves state-of-the-art accuracy on tabular reanalysis data, four strategic advancements will elevate the platform into a world-class operational system:

```
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                         NEXT-GENERATION ADVANCEMENT ROADMAP                            │
 ├─────────────────────────┬────────────────────────────┬─────────────────────────────────┤
 │ PHASE 1: Active Fires   │ PHASE 2: Dynamic NWP       │ PHASE 3: Spatial GNN            │
 │ NASA FIRMS (VIIRS/MODIS)│ WRF-Chem & ECMWF GraphCast │ Spatio-Temporal Graph Attention │
 │ Near-real-time FRP (MW) │ Prognostic BLH & Inversions│ Physical Advection Edges        │
 └─────────────────────────┴────────────────────────────┴─────────────────────────────────┤
                                           │
                                           ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ PHASE 4: Conformal Prediction & Decision Support for Graded Response Action Plan (GRAP)│
 │ • Rigorous 90% confidence bands: P(PM2.5 > 450 µg/m³) for triggering GRAP Stage IV      │
 │ • Counterfactual emission intervention simulation (truck ban / school closure impacts) │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Real-Time Satellite Active Fire Ingestion (NASA FIRMS)
* **Limitation of Current Model:** The stubble burning term currently relies on satellite column $CO$ and a day-of-year seasonal Gaussian kernel.
* **The Upgrade:** Integrate the **NASA Fire Information for Resource Management System (FIRMS)** API.
* **Mechanism:** Harvest near-real-time (NRT) 375-meter VIIRS and 1-km MODIS active fire detections over Punjab and Haryana every 3 hours. Compute the **total Fire Radiative Power (FRP, in Megawatts)** in the upwind North-West bounding box ($29.5^\circ\text{N}\text{--}32.0^\circ\text{N},\; 74.0^\circ\text{E}\text{--}76.5^\circ\text{E}$).
* **Expected Benefit:** Direct observational detection of massive stubble burning spikes 36 hours before smoke reaches Delhi, boosting $+48\text{h}$ and $+72\text{h}$ $PM_{2.5}$ $R^2$ by an estimated $+0.04$ to $+0.07$.

### Phase 2: High-Resolution Prognostic Weather Modeling (WRF-Chem / GraphCast)
* **Limitation of Current Model:** For future horizons ($+24\text{h}$ to $+72\text{h}$), boundary layer height ($BLH$) is currently approximated using train-fold historical climatology.
* **The Upgrade:** Couple the inference pipeline with live Numerical Weather Prediction (NWP) forecasts from ECMWF GraphCast or NCMRWF Unified Model.
* **Mechanism:** Ingest forecasted hourly vertical temperature profiles ($T_{925\text{hPa}} - T_{\text{surface}}$) and projected dynamic boundary layer collapse heights.
* **Expected Benefit:** Captures unseasonal western disturbances, sudden winter rain washouts, and unpredicted thermal inversion ceilings with true prognostic physics.

### Phase 3: Spatio-Temporal Graph Neural Networks (ST-GNN)
* **Limitation of Current Model:** Spatial information is currently captured through static GIS road buffers and Leave-One-Out spatial scalar means.
* **The Upgrade:** Deploy a **Spatio-Temporal Graph Attention Network (ST-GAT)** connecting the 10 Delhi CAAQMS stations as nodes in a dynamic graph.
* **Mechanism:** Edge weights between monitoring stations are dynamically parameterized by the instantaneous wind vector ($U_{10}, V_{10}$):

$$A_{ij}(t) = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2}\right) \cdot \max\Big(0,\; \hat{\mathbf{w}}(t) \cdot \frac{\mathbf{x}_j - \mathbf{x}_i}{\|\mathbf{x}_j - \mathbf{x}_i\|}\Big)$$

* An upwind station (e.g., `JAHANGIRPURI` in North Delhi) actively passes its current emission state to a downwind station (e.g., `ITO` or `OKHLA`) proportional to wind speed and alignment.
* **Expected Benefit:** Superior spatial tracking of mobile pollution plumes across the urban core.

### Phase 4: Conformal Prediction & Emergency GRAP Decision Support
* **Limitation of Current Model:** Outputs are deterministic point forecasts ($\mu\text{g/m}^3$).
* **The Upgrade:** Implement **Split Conformal Prediction** to output mathematically certified prediction intervals:

$$P\big(Y_{t+h} \in [\hat{y}^{\text{lower}}_{t+h},\; \hat{y}^{\text{upper}}_{t+h}]\big) \ge 1 - \alpha \quad (\text{e.g. } 90\% \text{ coverage guarantee})$$

* **Policy Value:** Enables the Commission for Air Quality Management (CAQM) to assess probabilistic emergency thresholds:
  *"There is an 88% probability that Anand Vihar PM2.5 will exceed 450 µg/m³ for > 48 consecutive hours starting Friday night."*
  This provides institutional justification for enacting Graded Response Action Plan (GRAP) Stage IV restrictions (halting construction, commercial diesel bans) **prior** to toxic accumulation.

---
*Certified Integration & Roadmap Blueprint — Problem Statement SIH26082*
