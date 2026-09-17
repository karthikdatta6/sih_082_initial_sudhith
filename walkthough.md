# AIRO2 — Complete Codebase Walkthrough & File-by-File Guide

**Project**: Smart India Hackathon (SIH 25178) — AI/ML Ground-Level Ozone ($O_3$) and Nitrogen Dioxide ($NO_2$) Multi-Horizon Forecasting System  
**Repository**: `SIH_26_AIR_O2`  
**Live Frontend**: [https://sih26airo2fe-seven.vercel.app](https://sih26airo2fe-seven.vercel.app)  
**Live Backend**: [https://sih-26-air-o2-backend.onrender.com/docs](https://sih-26-air-o2-backend.onrender.com/docs)  

---

## 1. Executive System Overview

The **AIRO2** platform is an end-to-end institutional atmospheric chemistry intelligence system designed to tackle severe urban air pollution in the National Capital Region (Delhi NCR) of India. 

Unlike traditional air quality index (AQI) dashboards that only display historical or present measurements, AIRO2 forecasts ground-level concentrations of **Nitrogen Dioxide ($NO_2$)** and **Ground-Level Ozone ($O_3$)** natively in $\mu g/m^3$ and CPCB NAQI sub-indices across **6 discrete, non-recursive forward horizons**:
$$\text{Forecast Horizons: } +1\text{h},\; +3\text{h},\; +6\text{h},\; +12\text{h},\; +24\text{h},\; +48\text{h}$$

The solution is split into distinct engineering phases:
1. **Phase 1 (Data Acquisition)**: Ground truth sensors from Central Pollution Control Board (CPCB), European Centre for Medium-Range Weather Forecasts (ECMWF ERA5) reanalysis, European Space Agency (ESA) Sentinel-5P TROPOMI satellite columns, and OpenStreetMap (OSM) / GIS urban spatial features.
2. **Phase 2 (Harmonization & Validation)**: Spatio-temporal nearest-neighbor and inverse-distance weighting (IDW) alignment, zero future leakage verification, and 138-feature engineering stored in high-performance Apache Parquet formats.
3. **Phase 3 (AI/ML Modeling)**: Hybrid stacked ensemble combining LightGBM GBDTs, PyTorch Bidirectional LSTM with temporal self-attention, Non-Negative Least Squares (NNLS) simplex stacking, and diurnal photochemical calibration.
4. **Phase 4 (Production Backend)**: Asynchronous FastAPI microservice with live CAMS/Open-Meteo fallbacks, SQLite caching, sub-10ms inference, TreeSHAP feature attributions, and early warning webhooks.
5. **Phase 7 (Interactive Frontend)**: React 18, TypeScript, Vite, TailwindCSS, Leaflet GIS Delhi maps with real-time station telemetry, and a rotating 3D D3-Geo atmospheric globe.

---

## 2. Directory Tree Structure

```
SIH_26_AIR_O2/
├── README.md                                  # Primary repository overview & quickstart
├── FOLDER STRUCTURE.md                        # Master itemized directory catalog
├── walkthough.md                              # This exhaustive file & folder walkthrough
├── LINK.md / LINK.txt / LIVE_LINK.txt         # Plain-language project explanation & live URLs
├── MASTER_PHASE1_CPCB_AND_ERA5.md             # Ground & meteorological data ingestion spec
├── MASTER_PHASE1_SENTINEL5P_AND_GEOSPATIAL.md # Satellite & spatial layers ingestion spec
├── app.py / setup.py / requirements.txt       # Production root deployment entrypoints
├── runtime.txt / .python-version / .gitignore # Python environment configurations
│
├── DATASET FUSION/                            # Phase 2 Harmonization & ETL Pipelines
│   ├── config/                                # Pipeline YAML configurations & station catalogs
│   ├── docs/                                  # Processing methodologies & QA specifications
│   ├── metadata/                              # Data dictionary & coordinates
│   └── scripts/                               # Data aggregation & leakage audit scripts
│
├── DATASET VALIDATION/                        # Phase 2 Data Quality & Verification Suite
│   ├── 01_SOURCE_STREAM_VALIDATORS/           # Stream-specific data integrity validators
│   ├── 02_FUSION_INTEGRITY_AND_LEAKAGE/       # Temporal leakage & missingness checks
│   ├── 03_QUALITY_AND_AUDIT_REPORTING/        # Automated QA reporting & logging scripts
│   ├── DOCUMENTATION/                         # Formal Phase 2 audit certification
│   └── RESULTS/                               # 12 empirical QA benchmark CSV reports
│
├── FINAL DATASET/                             # Production Parquet Datasets & Stores
│   ├── station_hourly_fused.parquet           # Fused hourly dataset (2019-2024)
│   ├── features_engineered.parquet            # Complete 138-feature engineered store
│   ├── station_static_features.parquet        # Fixed GIS characteristics per station
│   ├── anand_vihar_pilot.parquet              # High-resolution Anand Vihar pilot dataset
│   ├── metadata/                              # Authoritative feature dictionaries
│   └── quality_reports/                       # Benchmark QA verification CSVs
│
├── MODEL_ARCHITECTURE_RESEARCH/               # Theoretical Formulations & Comparative Research
│   ├── Leighton photochemical relationship analysis & diurnal photolytic studies
│   ├── GBDTs vs RNNs vs Transformers comparative benchmarks
│   └── Live real-time data ingestion trade-off analyses
│
├── MODEL CODE/                                # Phase 3 ML/DL Training Pipelines & Models
│   ├── 01_MACHINE_LEARNING_MODELS/            # Feature engineering & LightGBM training
│   ├── 02_DEEP_LEARNING_MODELS/               # PyTorch BiLSTM + Self-Attention models
│   ├── 03_ENSEMBLE_AND_META_STACKING/         # NNLS Convex Simplex Meta-Stacking
│   ├── 04_DIURNAL_CALIBRATION_MODEL/          # Solar angle diurnal weight calibration
│   ├── 05_TRAINING_PIPELINE_AND_CV/           # Expanding-window CV, EDA & SHAP explainability
│   ├── 06_PRODUCTION_INFERENCE_SERVICES/      # Runtime model services & AQI calculators
│   ├── 07_PRODUCTION_MODEL_BUNDLES/           # Serialized model weights (.pkl) & schemas
│   └── 08_MODEL_DOCUMENTATION/                # Architecture blueprints & handoff contracts
│
├── MODEL RESULTS/                             # Phase 3 Benchmark Metrics & Visualizations
│   ├── 01_BENCHMARK_AND_METRICS_CSVS/         # Quantitative validation metrics per horizon
│   ├── 02_VISUALIZATIONS_AND_SHAP/            # SHAP beeswarm plots & prediction curves
│   └── 03_EVALUATION_AND_ACCURACY_REPORTS/    # Scientific audit & evaluation reports
│
├── MODEL OUTPUT VALIDATION/                   # Phase 3 -> Phase 4 Golden Verification Suite
│   ├── 01_GOLDEN_COMPATIBILITY_TESTS/         # Golden vectors & automated PyTest suite
│   ├── 02_PHYSICAL_PLAUSIBILITY_AND_INVARIANTS# Scientific invariant checklists
│   └── 03_READINESS_AND_FIT_FOR_USE_CERTIFICATES# Fit-for-use readiness certificates
│
├── backend/ & PRODUCTION BACKEND SERVICE/     # Phase 4 Production FastAPI REST API
│   ├── app/
│   │   ├── api/                               # FastAPI dependencies & injectables
│   │   ├── config.py                          # Dynamic paths & model bundle resolution
│   │   ├── data/                              # SQLite databases for ring-buffer telemetry
│   │   ├── main.py                            # FastAPI lifespan, routes, CORS & middlewares
│   │   ├── middleware/                        # Security headers, rate limiters, error envelopes
│   │   ├── providers/                         # Historical & live multi-stream data providers
│   │   ├── routers/                           # Modular API endpoints (stations, model, alerts)
│   │   ├── scheduler.py                       # Background hourly inference refresh loop
│   │   ├── schemas/                           # Pydantic data validation schemas
│   │   ├── services/                          # Model inference service & database managers
│   │   ├── static/                            # Built-in standalone web dashboard (index.html)
│   │   └── utils/                             # CPCB AQI converters & feature builders
│   └── tests/                                 # 21 automated unit, live & golden test suites
│
└── FRONTEND/                                  # Phase 7 React + TypeScript + Vite Dashboard
    ├── public/                                # Static assets, SVG icons, and favicons
    └── src/
        ├── App.tsx                            # Root React component & view routing
        ├── main.tsx                           # Application DOM mount
        ├── index.css                          # Global TailwindCSS styling & animations
        ├── components/
        │   ├── common/                        # Status pills, clocks, trend charts
        │   ├── hero/                          # 3D interactive rotating Globe & Hero section
        │   ├── layout/                        # Topbar, Footer, and live AQI ticker
        │   ├── map/                           # RealDelhiMap (Leaflet GIS) & radar overlays
        │   └── station/                       # Station dashboard & +1h to +48h forecast panels
        ├── data/                              # Delhi boundaries & global TopoJSON
        └── lib/                               # API client, CPCB NAQI math, and mock fallbacks
```

---

## 3. Comprehensive File-by-File Documentation

### 3.1. Repository Root
* **`README.md`**: Presentation-grade master documentation covering the SIH 25178 challenge, system architecture diagrams, benchmark $R^2$ scores, quickstart guide, API documentation, and compliance matrix.
* **`FOLDER STRUCTURE.md`**: Master index detailing the organizational taxonomy of all 11 directories and 150+ individual codebase artifacts.
* **`walkthough.md`**: This comprehensive walkthrough explaining each file, folder, technology, and integration role.
* **`LINK.md` / `LINK.txt` / `LIVE_LINK.txt`**: One-paragraph-per-section executive summaries written in plain language explaining the real-world health crisis of air pollution, the chemistry of $NO_2$ and $O_3$, the AIRO2 solution, and links to the live Vercel web app and Render API.
* **`MASTER_PHASE1_CPCB_AND_ERA5.md`**: Detailed technical specification for Phase 1 data ingestion, defining raw sensor registers, CPCB ground-truth stations, ECMWF ERA5 meteorological parameters, and temporal resolution.
* **`MASTER_PHASE1_SENTINEL5P_AND_GEOSPATIAL.md`**: Specification document for Sentinel-5P TROPOMI satellite products ($NO_2$ tropospheric vertical column, $O_3$ total column) and static GIS layers (DEM elevation, road density, railway distance, land use).
* **`app.py`**: Clean root Python entrypoint for WSGI/ASGI application runners (Uvicorn / Gunicorn) that resolves project paths and exposes the FastAPI `app` instance.
* **`setup.py`**: Standard Python setuptools configuration package for building, packaging, and installing the AIRO2 backend service.
* **`requirements.txt`**: Pinned Python dependencies for the production server environment (`fastapi`, `uvicorn`, `pydantic`, `lightgbm`, `scikit-learn`, `numpy`, `scipy`, `pandas`, `pyarrow`, `httpx`, `reportlab`, `requests`).
* **`runtime.txt` & `.python-version`**: Pins Python version to `3.11.0` to ensure seamless pre-compiled binary wheel installations on cloud platforms (Render, Koyeb, Docker).
* **`.gitignore`**: Excludes temporary files, Python bytecode (`__pycache__`, `*.pyc`), SQLite test databases (`test_*.db`), node_modules, and build directories from Git tracking.

---

### 3.2. `DATASET FUSION/` (Phase 2 Data Harmonization)
* **`README.md`**: Pipeline instructions, data schema definitions, and execution guide for the multi-stream dataset fusion engine.
* **`requirements.txt`**: Specific dependencies required for dataset ETL and geospatial processing.
* **`config/phase2.yaml`**: Main YAML configuration file establishing historical date windows (2019–2024), geographical bounding boxes, spatial interpolation rules, and outlier screening limits.
* **`config/stations.csv`**: Master catalog of the 10 canonical Delhi NCR CPCB stations containing IDs, official names, latitudes, longitudes, and urban typology classifications.
* **`docs/PHASE_2_COMPLETE_DOCUMENTATION.md`**: Full documentation of data cleaning rules, stuck-sensor detection algorithms, and missingness imputation policies.
* **`docs/PHASE_2_FUSION_METHODOLOGY.md`**: Mathematical specification of inverse-distance weighting (IDW) spatial aggregation and hourly timestamp alignment across satellite, weather, and ground stations.
* **`metadata/data_dictionary.csv`**: Comprehensive field dictionary outlining 138 engineered features, physical units, data types, and sensor sources.
* **`metadata/station_locations.csv` & `metadata/station_metadata.csv`**: Ground-truth geospatial coordinates, station elevations, and local microclimate surroundings.
* **`scripts/build_fused_dataset.py`**: Core ETL script that reads raw data streams, synchronizes time stamps to Asia/Kolkata (IST) / UTC, applies spatial matching, and outputs harmonized Parquet files.
* **`scripts/run_phase2_pipeline.py`**: Master CLI orchestrator executing the full data pipeline from raw file ingestion to final feature stores.
* **`scripts/validate_inputs.py`**: Validates input file schemas, column names, and timestamp continuity before fusion begins.
* **`scripts/validate_cpcb.py`**: Ground-truth validator screening raw CPCB sensor readings for negative concentrations, out-of-range sensor spikes, and flatline values.
* **`scripts/validate_era5.py`**: Meteorology validator testing temperature ($K$), surface pressure ($Pa$), wind components ($u_{10}, v_{10}$), boundary layer height ($m$), and relative humidity for physical boundaries.
* **`scripts/validate_sentinel5p.py`**: Satellite quality-assurance filter enforcing ESA recommendations (`qa_value >= 0.5`) on TROPOMI granules to remove cloud-contaminated pixels.
* **`scripts/validate_geospatial.py`**: GIS coordinate reference system (CRS) and bounding box validator for Delhi NCR.
* **`scripts/leakage_check.py`**: Temporal boundary audit script ensuring rolling aggregates and lag features do not peek into future time steps.
* **`scripts/missingness_analysis.py`**: Evaluates sensor dropout patterns, consecutive NaN blocks, and imputation completeness.
* **`scripts/independent_audit.py`**: Autonomous verification script checking cross-stream distributions, z-score thresholds, and correlation sanity.

---

### 3.3. `DATASET VALIDATION/` (Phase 2 Rigorous QA & Audit)
* **`README.md`**: Instructions for running the validation suite and interpreting QA benchmark reports.
* **`run_all_validations.py`**: Automated test harness running every source validator, leakage test, and audit generator in a single execution.
* **`01_SOURCE_STREAM_VALIDATORS/`**: Individual validation scripts targeting each data provider: CPCB ground sensors, ERA5 reanalysis, Sentinel-5P, Geospatial, and raw input files.
* **`02_FUSION_INTEGRITY_AND_LEAKAGE/`**: Integrity scripts performing rolling lag leakage checks, missingness audits, independent cross-checks, and single-station pilot deep-dives (`inspection_station_pilot.py`).
* **`03_QUALITY_AND_AUDIT_REPORTING/`**: Scripts that compile, merge, and export formal quality reports into standardized CSV benchmark summaries.
* **`DOCUMENTATION/PHASE_2_AUDIT_REPORT.md`**: Formal audit report confirming zero temporal leakage, >98% data fidelity, and mathematical soundness.
* **`DOCUMENTATION/PHASE_2_FUSION_METHODOLOGY.md`**: Detailed methodology report explaining spatial raster sampling and multi-sensor synchronization.
* **`RESULTS/`**: Directory containing 12 published CSV validation reports:
  - `cpcb_quality_report.csv`: CPCB data completeness and valid percentage per pollutant.
  - `era5_quality_report.csv`: Weather parameter range checks and completeness.
  - `sentinel5p_quality_report.csv`: Satellite granule coverage and QA filtering yields.
  - `geospatial_quality_report.csv`: Spatial buffer calculations and distance-to-road checks.
  - `fusion_quality_report.csv` & `independent_dataset_audit.csv`: Master data consistency audits.
  - `leakage_report.csv`: Mathematical proof of zero future data leakage across all folds.
  - `missingness_report.csv`: Missing value distributions and imputation audit.
  - `phase2_input_inventory.csv`: Complete inventory of all ingested source files.
  - `spatial_matching_report.csv`: Verification of spatial coordinates and interpolation radii.
  - `station_coverage_report.csv`: Station-by-station timestamp coverage from 2019 to 2024.
  - `temporal_matching_report.csv`: Verification of hourly UTC/IST temporal alignment.

---

### 3.4. `FINAL DATASET/` (Production Data & Feature Stores)
* **`README.md`**: Dataset documentation, schema layouts, column counts, and code snippets demonstrating loading via Pandas, DuckDB, or PyArrow.
* **`station_hourly_fused.parquet`**: Master hourly dataset spanning 2019–2024 across 10 Delhi CPCB stations containing synchronized ground concentrations, ERA5 meteorological parameters, and Sentinel-5P satellite columns.
* **`features_engineered.parquet`**: Complete engineered feature repository containing 138 domain features (cyclical hour/day/month encodings, 1h–72h multi-scale lags, rolling statistics, solar zenith angles, ventilation coefficients).
* **`station_static_features.parquet`**: Fixed urban spatial characteristics for each station (elevation, road distance, commercial/residential/park land-use fractions).
* **`anand_vihar_pilot.parquet`**: Focused pilot dataset for Anand Vihar used for rapid baseline testing and benchmark comparisons.
* **`metadata/data_dictionary.csv`**: Authoritative data dictionary detailing all 138 feature names, physical dimensions, and origins.
* **`metadata/station_locations.csv` & `metadata/station_metadata.csv`**: Ground-truth geographical coordinates, commissioning dates, and site typologies.
* **`quality_reports/`**: Mirror of the 12 QA benchmark verification CSVs for offline compliance audits.

---

### 3.5. `MODEL_ARCHITECTURE_RESEARCH/` (Atmospheric Science & Modeling Research)
* **`README.md`**: Survey of atmospheric physics literature, research papers, and theoretical foundations informing AIRO2.
* **`MODEL_ARCHITECTURE_RESEARCH.md`**: Comprehensive research review analyzing Tree Ensembles (LightGBM, XGBoost, CatBoost), Sequential Deep Learning (LSTM, BiLSTM, GRU), and Spatial-Temporal Transformers.
* **`ML_RESEARCHER_THEORETICAL_ANALYSIS.md`**: Detailed mathematical formulation of the non-linear photochemical Leighton cycle:
  $$NO_2 + h
u (\lambda < 420\text{nm}) \xrightarrow{J_{NO2}} NO + O(^3P)$$
  $$O(^3P) + O_2 + M \rightarrow O_3 + M$$
  $$O_3 + NO \xrightarrow{k} NO_2 + O_2$$
  Explains why linear and uncalibrated models fail during daytime ozone peaks and nighttime titration sinks.
* **`DIURNAL_CALIBRATION_ANALYSIS.md`**: Scientific analysis of solar zenith angle fluctuations, boundary layer collapse, and empirical diurnal weighting curves.
* **`COMPARATIVE_ANALYSIS_LIVE_DATA_INGESTION_METHODS.md`**: Trade-off analysis comparing REST polling, WebSocket streaming, and scheduled event triggers for real-time inference.
* **`MODEL_BUILD_RECOMMENDATIONS_ANALYSIS.md`**: Technical recommendations for loss function selection (Huber loss to resist sensor spikes vs Pinball loss for uncertainty quantiles) and direct vs recursive multi-horizon strategies.
* **`PROPOSAL_HYBRID_SATELLITE_VS_CPCB.md`**: Mathematical framework for combining sparse, orbit-dependent satellite columns with high-frequency in-situ ground sensors.
* **`SUDHITH_EXTRA_FEATURES_ANALYSIS.md`**: Exploration of advanced atmospheric features including Planetary Boundary Layer Ventilation Index ($V = \text{BLH} \times \text{wind\_speed}$), temperature inversions, and photochemical aging ratios ($[O_3] / [NO_x]$).

---

### 3.6. `MODEL CODE/` (Phase 3 Machine Learning & Inference Engines)
* **`README.md`**: Guide to model training, cross-validation protocols, and production bundle export.
* **`01_MACHINE_LEARNING_MODELS/feature_engineering.py`**: Production feature engineering pipeline computing 138 lag, rolling, cyclical time, and interaction features.
* **`01_MACHINE_LEARNING_MODELS/train_lightgbm.py`**: LightGBM training pipeline training direct, non-recursive models across all 6 horizons (+1h, +3h, +6h, +12h, +24h, +48h) for $NO_2$ and $O_3$ with early stopping and Bayesian hyperparameter tuning.
* **`02_DEEP_LEARNING_MODELS/train_bilstm_attention.py`**: PyTorch Bidirectional LSTM model with temporal self-attention that captures sequential atmospheric trends and long-term lag dependencies.
* **`03_ENSEMBLE_AND_META_STACKING/nnls_simplex_stacking.py`**: Non-Negative Least Squares (NNLS) convex simplex meta-stacker combining LightGBM and BiLSTM predictions subject to $\sum w_i = 1, w_i \ge 0$, guaranteeing non-negative outputs without arbitrary clipping.
* **`04_DIURNAL_CALIBRATION_MODEL/diurnal_calibration.py`**: Post-processing calibration module that adjusts raw ensemble outputs using empirical solar radiation and hour-of-day curves.
* **`04_DIURNAL_CALIBRATION_MODEL/fit_diurnal_weights.py`**: Computes station-specific hourly adjustment coefficients from historical training sets.
* **`05_TRAINING_PIPELINE_AND_CV/run_master_pipeline.py`**: Master orchestration script executing data loading, expanding-window CV, LightGBM training, BiLSTM training, NNLS stacking, and production bundle export.
* **`05_TRAINING_PIPELINE_AND_CV/temporal_cross_validation.py`**: Strict 5-fold expanding-window temporal cross-validation module preventing future data leakage across train and test folds.
* **`05_TRAINING_PIPELINE_AND_CV/evaluation_metrics.py`**: Comprehensive metric computation suite calculating $R^2$, RMSE, MAE, symmetric MAPE (sMAPE), Peak-F1, and Directional Accuracy.
* **`05_TRAINING_PIPELINE_AND_CV/eda_analysis.py`**: Automated exploratory data analysis generating correlation matrices, diurnal boxplots, and seasonal pollution cycles.
* **`05_TRAINING_PIPELINE_AND_CV/shap_attribution.py`**: TreeSHAP explainability engine computing Shapley values to identify the primary meteorological and chemical drivers behind each forecast.
* **`06_PRODUCTION_INFERENCE_SERVICES/model_service.py`**: Lightweight, ultra-fast production inference runtime loading serialized model bundles and executing predictions in $<10\text{ms}$.
* **`06_PRODUCTION_INFERENCE_SERVICES/feature_builder.py`**: Converts raw sensor readings and live weather inputs into the exact 58-feature vector required by production models.
* **`06_PRODUCTION_INFERENCE_SERVICES/aqi_calculator.py`**: Official Indian National Air Quality Index (NAQI) calculator using CPCB piecewise linear interpolation and sub-index breakpoints.
* **`07_PRODUCTION_MODEL_BUNDLES/NO2/`**:
  - `model.pkl`: Serialized production LightGBM + NNLS model weights for $NO_2$.
  - `feature_schema.json`: Strict schema defining the expected names and order of the 58 input features.
  - `metadata.json`: Model version, training date, target pollutant, evaluation metrics, and SHA-256 integrity hashes.
* **`07_PRODUCTION_MODEL_BUNDLES/O3/`**:
  - `model.pkl`: Serialized production model weights for $O_3$.
  - `feature_schema.json`: Strict feature schema for $O_3$.
  - `metadata.json`: Model metadata, performance benchmarks, and training parameters.
* **`08_MODEL_DOCUMENTATION/`**:
  - `MODEL_ARCHITECTURE.md`: Complete architectural blueprint of the stacked hybrid system.
  - `MODEL_CONTRACT.md`: API input/output contract defining payload schemas, physical units ($\mu g/m^3$), and error envelopes.
  - `PHASE_3_COMPLETE_MASTER_HANDOUT.md`: Executive summary of model performance, cross-validation results, and key findings.
  - `PHASE_3_MODEL_DEVELOPER_SPECIFICATION.md`: Developer specification detailing integration patterns for backend engineers.
  - `PHASE_3_PIPELINE_GUIDE.md`: Operations manual for retraining, tuning, and re-exporting model bundles.
  - `PHASE_3_SUDHITH_IMPLEMENTATION_PLAN.md`: Technical design document tracking model engineering deliverables.
  - `PHASE_3_TO_PHASE_4_HANDOFF_REQUIREMENTS.md`: Handoff contract defining latency limits ($<200\text{ms}$), 6 discrete non-recursive horizons, and zero-leakage constraints.

---

### 3.7. `MODEL RESULTS/` (Phase 3 Benchmarks & Visualizations)
* **`README.md`**: Guide to model evaluation artifacts, metric tables, and visual figures.
* **`FINAL_DOC.md`**: Master synthesis report summarizing cross-validation benchmarks, error bounds, and scientific validation.
* **`01_BENCHMARK_AND_METRICS_CSVS/`**:
  - `phase3_evaluation_summary.csv`: Master performance comparison across all models and horizons.
  - `lightgbm_evaluation_summary.csv`: Quantitative metrics ($R^2$, RMSE, MAE) for LightGBM.
  - `bilstm_evaluation_summary.csv`: Quantitative metrics for BiLSTM with Attention.
  - `ensemble_evaluation_summary.csv`: Benchmark results showing stacked ensemble performance gains.
  - `station_evaluation_summary.csv`: Granular performance broken down by each of the 10 Delhi CPCB stations.
  - `cv_fold_boundaries.csv`: Exact start and end timestamps for the 5 expanding-window cross-validation splits.
* **`02_VISUALIZATIONS_AND_SHAP/`**:
  - `forecast_vs_actual_NO2_ITO.png`: Time-series plot comparing forecasted vs actual $NO_2$ at ITO station.
  - `forecast_vs_actual_O3_ITO.png`: Time-series plot comparing forecasted vs actual $O_3$ at ITO station.
  - `horizon_degradation_NO2.png` & `horizon_degradation_O3.png`: Curves showing forecast accuracy ($R^2$) across +1h, +3h, +6h, +12h, +24h, and +48h horizons.
  - `shap_summary_NO2.png` & `shap_summary_O3.png`: Beeswarm summary plots illustrating global feature importances (solar radiation, wind speed, boundary layer height).
  - `shap_top10_NO2.csv` & `shap_top10_O3.csv`: Ranked CSV tables of the top 10 most influential features and their mean absolute SHAP values.
* **`03_EVALUATION_AND_ACCURACY_REPORTS/`**:
  - `FINAL_EXECUTION_REPORT.md`: Formal verification report certifying compliance with SIH problem requirements.
  - `PHASE_3_EVALUATION_REPORT.md`: Comprehensive model evaluation analyzing residual distributions and peak smog detection.
  - `PHASE_3_INTEGRITY_AND_ACCURACY_REPORT.md`: Audit certifying physical consistency (no negative concentrations).
  - `ULTRA_DETAILED_EVALUATION_METRICS_AND_RESEARCH_AUDIT.md`: Deep research audit reviewing boundary layer physics and atmospheric chemistry alignment.

---

### 3.8. `MODEL OUTPUT VALIDATION/` (Golden Tests & Certification)
* **`README.md`**: Guide to running the golden test harness and certifying backend compatibility before release.
* **`verify_model_readiness.py`**: Standalone CLI validation script testing bundle availability, feature count (58), inference speed, and non-negativity.
* **`01_GOLDEN_COMPATIBILITY_TESTS/`**:
  - `input.json`: Authoritative golden test vector representing realistic sensor observations and weather conditions.
  - `expected_output.json`: Frozen ground-truth expected output containing exact 12 predictions (2 pollutants $\times$ 6 horizons) and AQI sub-indices.
  - `test_phase3_phase4_compatibility.py`: Automated PyTest suite asserting that Phase 4 production backend inference matches Phase 3 frozen models within strict numerical tolerance ($\le 0.001\,\mu g/m^3$).
  - `README.md`: Guide to executing compatibility tests.
* **`02_PHYSICAL_PLAUSIBILITY_AND_INVARIANTS/`**:
  - `MODEL_OUTPUT_INVARIANTS_CHECKLIST.md`: Scientific checklist verifying physical invariants ($O_3 \ge 0$, diurnal photochemical peak timing, boundary layer inverse relationship).
  - `PHASE_3_INTEGRITY_AND_ACCURACY_REPORT.md`: Integrity assessment under extreme meteorological stress tests.
  - `test_live_observation_pipeline.py`: Unit test suite validating live data buffering, priority selection, and missing data fallbacks.
* **`03_READINESS_AND_FIT_FOR_USE_CERTIFICATES/`**:
  - `MODEL_FIT_FOR_USE_CERTIFICATE.md`: Formal engineering certificate declaring the models fit for institutional operational deployment.
  - `FINAL_MASTER_AUDIT_REPORT.md`: Final consolidated pipeline audit.
  - `ULTRA_DETAILED_EVALUATION_METRICS_AND_RESEARCH_AUDIT.md`: Research audit detailing domain validity.

---

### 3.9. `backend/` and `PRODUCTION BACKEND SERVICE/` (Phase 4 FastAPI Microservice)
*Note: Both directories maintain synchronized, production-grade copies of the FastAPI service.*
* **`__init__.py`**: Marks directory as an importable Python package.
* **`requirements.txt`**: Pinned Python dependencies for the backend microservice.
* **`app/__init__.py`**: Initializes the main application package.
* **`app/main.py`**: FastAPI application entry point:
  - Configures lifespan context manager for one-time startup model bundle loading.
  - Implements outer-to-inner security middlewares (SecurityHeaders, RateLimiter, PayloadSizeLimit, RequestCorrelation).
  - Registers standardized error envelopes (`error_handler.py`).
  - Configures CORS policies for web frontends.
  - Mounts API routers (`stations`, `model`, `explain`, `alerts`, `simulate`, `spatial`, `report`, `live`).
  - Exposes Kubernetes / cloud health probes (`/healthz`).
* **`app/config.py`**: Master configuration file dynamically resolving model bundles, golden test inputs, SQLite database paths, canonical 10-station metadata, and 6 forecast horizons.
* **`app/scheduler.py`**: Asynchronous background task running an hourly forecast refresh loop in live provider mode.
* **`app/api/deps.py`**: FastAPI dependency injection helpers providing database connections and service handles to router endpoints.
* **`app/data/forecast_store.db`**: SQLite database storing cached station predictions and historical forecast logs.
* **`app/data/live_observations.db`**: SQLite ring-buffer database caching recent observations for lag construction.
* **`app/middleware/security.py`**: Enterprise security middleware implementing:
  - `SecurityHeadersMiddleware`: Injects Content-Security-Policy (CSP), Strict-Transport-Security (HSTS), X-Frame-Options, X-Content-Type-Options, and Referrer-Policy headers.
  - `RateLimiterMiddleware`: In-memory token-bucket rate limiter enforcing 180 requests per minute per client IP.
  - `PayloadSizeLimitMiddleware`: Rejects incoming payloads exceeding 2MB.
  - `RequestCorrelationMiddleware`: Injects unique `X-Request-ID` and `X-Response-Time-Ms` tracking headers.
* **`app/middleware/error_handler.py`**: Global exception interceptor transforming Python exceptions and HTTP 4xx/5xx errors into clean, sanitized JSON envelopes:
  `{"error": {"code": "RESOURCE_NOT_FOUND", "message": "...", "request_id": "..."}}`.
* **`app/providers/base.py`**: Defines the abstract `Observation` dataclass and `BaseObservationProvider` interface.
* **`app/providers/historical.py`**: Historical provider reading past measurements directly from the Parquet feature store.
* **`app/providers/live/store.py`**: In-memory ring buffer storing the last 72 hours of station observations for calculating dynamic lag and rolling features.
* **`app/providers/live/cams.py`**: CAMS (Copernicus Atmosphere Monitoring Service) / Open-Meteo air chemistry client with automatic fallback for ground-level precursor observations.
* **`app/providers/live/weather.py`**: Live numerical weather prediction client fetching live temperature, wind speed, relative humidity, pressure, and solar radiation.
* **`app/providers/live/cpcb_manual.py`**: Provider allowing manual injection of real-time CPCB telemetry.
* **`app/providers/live/sentinel.py`**: Satellite observation client providing tropospheric $NO_2$ and $O_3$ columns.
* **`app/providers/live/live_provider.py`**: Master provider coordinating live weather, CAMS, and CPCB feeds into a unified observation stream.
* **`app/routers/stations.py`**:
  - `GET /api/v1/stations`: Returns list of all 10 CPCB monitoring stations with live health status, coordinates, and typology.
  - `GET /api/v1/stations/{station_id}/forecast`: Main forecast endpoint emitting 12 discrete predictions (2 pollutants $\times$ 6 horizons) plus CPCB NAQI sub-indices.
* **`app/routers/model.py`**:
  - `GET /api/v1/model`: Returns model metadata, architecture description, 58-feature schemas, and training metrics.
* **`app/routers/explain.py`**:
  - `GET /api/v1/stations/{station_id}/forecast/explanation`: Returns SHAP feature importance attributions explaining top atmospheric drivers.
* **`app/routers/alerts.py`**:
  - `GET /api/v1/alerts/feed`: Returns real-time early warning alerts for stations exceeding CPCB thresholds.
  - `POST /api/v1/alerts/subscribe`: Allows external systems to register webhook callbacks for automated threshold breach alerts.
  - `GET /api/v1/alerts/custom-forecast`: Computes dynamic weather and forecasts for ANY custom GPS coordinate in India.
* **`app/routers/simulate.py`**:
  - `POST /api/v1/simulate`: Policy "What-If" sandbox simulating the impact of traffic reductions and industrial emission cuts on downwind ozone and $NO_2$.
* **`app/routers/spatial.py`**:
  - `GET /api/v1/spatial`: Generates 2D continuous air quality heatmap grids across Delhi NCR using Gaussian spatial interpolation.
* **`app/routers/report.py`**:
  - `GET /api/v1/report/daily`: Generates automated daily air quality briefing reports in downloadable PDF and CSV formats.
* **`app/schemas/station.py`**: Pydantic models for station registry, geographical coordinates, and health status envelopes.
* **`app/schemas/forecast.py`**: Pydantic schemas validating multi-horizon forecast responses, pollutant breakdowns, and simulated policy payloads.
* **`app/services/model_service.py`**: Core model execution service that holds loaded LightGBM models in memory, performs inference, enforces non-negativity, and formats outputs.
* **`app/services/feature_service.py`**: Dynamic feature assembler building the 58-feature vector from live ring-buffer observations.
* **`app/services/forecast_database.py`**: SQLite database manager persisting forecast histories and audit trails.
* **`app/static/index.html`**: Complete, self-contained interactive web control center dashboard featuring live Leaflet GIS maps, station selectors, and prediction tables.
* **`app/utils/aqi.py`**: Official CPCB NAQI piecewise linear interpolation calculation logic and category classification (Good, Satisfactory, Moderate, Poor, Very Poor, Severe).
* **`app/utils/feature_builder.py`**: Authoritative builder enforcing exact 58-feature schema ordering as defined in `feature_schema.json`.
* **`tests/test_phase3_phase4_compatibility.py`**: 13 automated golden compatibility tests asserting zero drift between Phase 3 models and Phase 4 API outputs.
* **`tests/test_live_observation_pipeline.py`**: 5 unit tests verifying observation store ring buffers, missingness fallbacks, and CPCB priority selection.
* **`tests/test_location_and_webhook_alerts.py`**: 3 integration tests checking location-agnostic geocoding, custom coordinate forecasts, and webhook simulation payloads.

---

### 3.10. `FRONTEND/` (Phase 7 React 18 Web Application)
* **`package.json`**: NPM package configuration declaring dependencies (`react`, `react-dom`, `leaflet`, `three`, `three-globe`, `d3-geo`, `framer-motion`, `lucide-react`, `recharts`, `tailwindcss`, `vite`).
* **`vite.config.ts`**: Vite build configuration with React plugin and TailwindCSS integrations.
* **`tsconfig.json` / `tsconfig.app.json` / `tsconfig.node.json`**: TypeScript compiler options enforcing strict type-checking.
* **`index.html`**: Single-page application HTML entrypoint loading Google Fonts (`JetBrains Mono`, `Anton`) and mounting the React root.
* **`public/favicon.svg` & `public/icons.svg`**: Application branding icons and shared SVG sprite sheets.
* **`src/main.tsx`**: Application entrypoint mounting `<App />` into the DOM.
* **`src/App.tsx`**: Root React component managing active navigation tabs, selected station states, real-time forecast fetching, and view switching.
* **`src/index.css`**: Global TailwindCSS style definitions, custom scrollbars, and radar animation keyframes.
* **`src/components/common/`**:
  - `LiveClock.tsx`: Displays current Indian Standard Time (IST) with live second updates.
  - `ScrambleText.tsx`: Cyberpunk-style animated text decryption effect for headers and values.
  - `SectionHeader.tsx`: Reusable section header component with decorative badges.
  - `SegmentBar.tsx`: Visual horizontal segmented bar rendering AQI severity ranges.
  - `StatusPill.tsx`: Color-coded badge showing operational status (Normal, Elevated, Severe, Offline).
  - `TrendChart.tsx`: Recharts-powered trend line displaying $+1\text{h}$ to $+48\text{h}$ predicted trajectories.
  - `TypewriterText.tsx`: Smooth typewriter animation component for technical briefings.
* **`src/components/hero/`**:
  - `Hero.tsx`: High-impact landing page banner displaying system telemetry and key metrics.
  - `Globe.tsx`: Interactive 3D rotating Earth globe rendered via D3-Geo showcasing regional atmospheric circulation patterns.
  - `PixelTransition.tsx`: Smooth pixelated transition effect for screen navigation.
* **`src/components/layout/`**:
  - `Topbar.tsx`: Navigation bar with system health badge, active station indicator, and quick links.
  - `Footer.tsx`: Institutional footer with problem statement credits, data sources, and copyright.
  - `Ticker.tsx`: Scrolling news-ticker displaying real-time AQI readings across all 10 Delhi stations.
* **`src/components/map/`**:
  - `RealDelhiMap.tsx`: Interactive Leaflet GIS map rendered with dark CartoDB tiles, Delhi administrative boundary GeoJSON, color-coded station pins, and radar scanlines.
  - `DelhiMap.tsx`: Static SVG map fallback for offline or low-bandwidth environments.
  - `ScanOverlay.tsx`: Animated canvas radar beam sweeping across Delhi NCR coordinates.
  - `StationInfoPanel.tsx`: Floating map card displaying detailed station readings upon marker click.
* **`src/components/station/`**:
  - `StationDashboard.tsx`: Comprehensive station view showing current readings, AQI gauges, and forecast charts.
  - `ForecastPanel.tsx`: Interactive horizon selector allowing users to inspect $+1\text{h}, +3\text{h}, +6\text{h}, +12\text{h}, +24\text{h}, +48\text{h}$ pollutant concentrations.
  - `HistoryPanel.tsx`: Historical pollutant trend comparison over the past 24 to 72 hours.
  - `QuickScanTable.tsx`: Summary matrix comparing all 10 stations simultaneously by AQI, $NO_2$, and $O_3$.
  - `ReadingCard.tsx`: Metric card presenting concentration values, percentage change, and health advisory tags.
* **`src/data/`**:
  - `countries-110m.json`: TopoJSON land boundaries for the 3D interactive globe.
  - `delhi_boundary.json`: Official GeoJSON polygon defining the Delhi NCR state borders for Leaflet GIS rendering.
* **`src/lib/`**:
  - `api.ts`: Typed TypeScript client for calling backend REST endpoints (`fetchStations`, `fetchForecast`, `fetchExplanation`).
  - `aqi.ts`: Client-side CPCB NAQI calculation logic for instant client-side updates.
  - `alerts.ts`: Threshold evaluation engine classifying alert severity (Moderate, Poor, Severe).
  - `mockData.ts`: High-fidelity offline fallback dataset enabling full dashboard operation even when disconnected from the backend.
  - `stations.ts`: Station catalog containing coordinates, IDs, typologies, and descriptions.
  - `delhiBoundary.ts`: Helper utility parsing and projecting Delhi GeoJSON polygons.
  - `time.ts`: Date and time formatting utilities for IST and UTC conversions.
  - `trend.ts`: Trend direction calculators determining whether pollution is rising or falling.
  - `useInView.ts`: React intersection observer hook for triggering scroll-based animations.
  - `PageTransition.tsx`: Route transition wrapper providing smooth view switches.

---

## 4. Key Architectural Highlights & Verification

### 4.1. The 10 Target Delhi NCR Stations
1. **Anand Vihar** (`ANAND_VIHAR`) — Traffic + Residential Hub (East Delhi border & ISBT bus terminal)
2. **ITO** (`ITO`) — Heavy Traffic Corridor (Central government & commercial intersection)
3. **Okhla Phase-2** (`OKHLA_PHASE_2`) — Industrial + Residential (South-East industrial estate)
4. **Aya Nagar** (`AYA_NAGAR`) — Suburban + Green (South Delhi border, baseline background)
5. **R.K. Puram** (`RK_PURAM`) — Dense Residential (South Delhi residential & institutional hub)
6. **Major Dhyan Chand National Stadium** (`DHYAN_CHAND_STADIUM`) — Urban Background + Park (Central Delhi near India Gate)
7. **Mandir Marg** (`MANDIR_MARG`) — Commercial + Traffic (Central Delhi arterial corridor)
8. **Punjabi Bagh** (`PUNJABI_BAGH`) — Residential + Traffic (West Delhi Ring Road corridor)
9. **Jahangirpuri** (`JAHANGIRPURI`) — Industrial + Dense Residential (North Delhi industrial pocket)
10. **Dwarka Sector-8** (`DWARKA_SECTOR_8`) — Planned Residential (South-West Delhi near airport corridor)

### 4.2. Verification & Automated Testing (100% Pass)
The entire test suite is verified via PyTest:
```bash
pytest backend/tests -v
```
All **21 automated tests pass (100%)**:
- **13 Golden Compatibility Tests**: Exact numeric assertion matching Phase 3 frozen models ($\le 0.001\,\mu g/m^3$ tolerance).
- **5 Live Pipeline Tests**: Multi-stream ring buffer, CPCB priority, and missing data fallback validation.
- **3 Alert & Geocoding Tests**: Location-agnostic geocoding, custom coordinate forecasts, and webhook alerts.

---

## 5. Live Production Deployment Links

| Resource | URL | Hosting Platform |
|---|---|---|
| **Interactive Frontend Web App** | [https://sih26airo2fe-seven.vercel.app](https://sih26airo2fe-seven.vercel.app) | **Vercel** |
| **FastAPI REST API Server** | [https://sih-26-air-o2-backend.onrender.com](https://sih-26-air-o2-backend.onrender.com) | **Render (Singapore)** |
| **Interactive Swagger API Docs** | [https://sih-26-air-o2-backend.onrender.com/docs](https://sih-26-air-o2-backend.onrender.com/docs) | **Render** |
| **Backend Health Probe** | [https://sih-26-air-o2-backend.onrender.com/healthz](https://sih-26-air-o2-backend.onrender.com/healthz) | **Render** |
| **Built-in Standalone Web UI** | [https://sih-26-air-o2-backend.onrender.com/static/index.html](https://sih-26-air-o2-backend.onrender.com/static/index.html) | **Render** |
| **GitHub Source Code** | [https://github.com/Sudhith/SIH_26_AIR_O2](https://github.com/Sudhith/SIH_26_AIR_O2) | **GitHub** |

========================================================================================
Submitted for Smart India Hackathon (SIH 2026) | Problem Statement 25178 | Team Zephr
========================================================================================
