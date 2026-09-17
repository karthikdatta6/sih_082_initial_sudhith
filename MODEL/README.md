# 🏛️ MODEL · SIH26082 Coupled Air Pollution–Weather Forecasting Engine

**Smart India Hackathon 2026 — Problem Statement SIH26082**  
**Nodal Organization:** Ministry of Earth Sciences (MoES) / NCMRWF  
**Architecture:** Two-Way Atmospheric Chemical-Meteorological Coupled Stacking Ensemble  
**Domain:** Delhi National Capital Region (NCR), India — 10 Canonical CPCB CAAQMS Stations  
**Coverage:** 263,040 Time Steps (3 Unbroken Years: 2023–2025) · 7 Horizons (+1h to +72h) · 3 Pollutants ($PM_{2.5}, O_3, NO_2$)  

---

## 📑 Directory Navigation Index

This directory represents the self-contained, certified modeling and documentation suite for problem statement **SIH26082**:

| Document / Subfolder | Scope & Contents |
|---|---|
| **[`01_APPROACH_RESEARCH_AND_INITIAL_PLAN.md`](01_APPROACH_RESEARCH_AND_INITIAL_PLAN.md)** | **Exhaustive Scientific Dossier:** Atmospheric boundary layer dynamics, Leighton photolysis cycle, literature survey of 13 papers, what we initially thought vs what we discovered, initial plan vs what we built, and step-by-step mathematical derivations. |
| **[`02_RESULTS_AND_BENCHMARKS_EXPLAINED.md`](02_RESULTS_AND_BENCHMARKS_EXPLAINED.md)** | **Empirical Benchmark Report:** All 21 models evaluated on the held-out test set (July–December 2025). Full $R^2$, Willmott $d$, RMSE, MAE, SMAPE, Skill vs Persistence, head-to-head comparison vs legacy v1.0.0, ablation study, and NNLS weights. |
| **[`03_OUTPUTS_INTEGRATION_AND_FUTURE_ROADMAP.md`](03_OUTPUTS_INTEGRATION_AND_FUTURE_ROADMAP.md)** | **System Integration & Next-Gen Roadmap:** Complete output inventory, how to connect outputs to FastAPI backend and React frontend, real-time NAQI AQI calculation, local runbook, and 4-phase roadmap (NASA FIRMS, WRF-Chem, ST-GNN, Conformal Prediction). |
| **[`code/`](code/)** | **Reproducible Execution Pipeline:** Standalone Python scripts (`01_build_coupled_features.py`, `02_train_coupled_ensemble.py`, `03_export_v2_bundles.py`, `04_train_ensemble_v3.py`, `coupled_physics.py`, and `requirements.txt`). |
| **[`results/`](results/)** | **Frozen Verification Artifacts:** Authoritative metrics JSON (`metrics_by_horizon.json`), ablation table CSV (`ablation_table.csv`), feature schema (`feature_schema_v2.json`), feature catalog (`feature_catalog.csv`), and verbatim training log (`training_run.log`). |

---

## 🚀 Performance Highlights (Held-Out Test Set: Jul–Dec 2025)

Every metric below is computed on un-imputed, pure NaN ground truth targets across 44,160 held-out hourly time steps:

```
┌──────────────┬────────────────────────┬────────────────────────┬────────────────────────┐
│  Pollutant   │       +1 Hour          │       +24 Hours        │       +72 Hours        │
├──────────────┼────────────────────────┼────────────────────────┼────────────────────────┤
│ PM2.5        │ R² 0.9634 · d 0.9906   │ R² 0.7492 · d 0.9211   │ R² 0.6426 · d 0.8661   │
│              │ RMSE 21.34 · Skill +18%│ RMSE 55.93 · Skill +28%│ RMSE 66.90 · Skill +33%│
├──────────────┼────────────────────────┼────────────────────────┼────────────────────────┤
│ Ozone (O₃)   │ R² 0.9201 · d 0.9806   │ R² 0.7557 · d 0.9275   │ R² 0.6621 · d 0.8862   │
│              │ RMSE 10.16 · Skill +57%│ RMSE 17.79 · Skill +15%│ RMSE 21.00 · Skill +15%│
├──────────────┼────────────────────────┼────────────────────────┼────────────────────────┤
│ NO₂          │ R² 0.9378 · d 0.9842   │ R² 0.7629 · d 0.9314   │ R² 0.6603 · d 0.8777   │
│              │ RMSE  9.34 · Skill +38%│ RMSE 18.25 · Skill +15%│ RMSE 21.88 · Skill +14%│
└──────────────┴────────────────────────┴────────────────────────┴────────────────────────┘
```

* **Skill vs. Persistence is strictly positive across all 21 (pollutant $\times$ horizon) models.** The AI model consistently beats "tomorrow equals today".
* **Head-to-head gains over legacy v1.0.0:** Ozone $+1\text{h}$ explained variance jumped by **$+0.0512$ $R^2$**, $NO_2$ $+1\text{h}$ gained **$+0.0187$ $R^2$**, while $PM_{2.5}$ and the $+72\text{h}$ horizon are brand new capabilities.

---

## 🏗️ The 2-Tier Stacked Ensemble Architecture

```
                                 t = Observation Instant (UTC)
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
       Static & Historical Features (t)                         Target-Time Forcing (t + h)
       ├── 58 Legacy temporal/spatial features                  ├── Astronomical Solar Zenith Angle cos θz
       ├── Atmospheric Inversion Trapping Index (ITI)           ├── Target Diurnal & Annual Sine/Cosine
       ├── Aerosol Clearness Index (Kt)                         └── Historical Climatological Priors
       ├── Corrected NW Stubble Plume Inflow (m/s)
       └── Cross-Sectional LOO Citywide Means
                                                │
                                                ▼
                         78-Feature Vector Aligned to Frozen Schema
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
          LightGBM (L1 / MAE)          LightGBM (Huber Loss)          LightGBM (L2 / MSE)
          (Median Robustness)          (Smooth Outlier Reject)        (Variance Optimization)
                 │                              │                              │
                 └──────────────────────────────┼──────────────────────────────┘
                                                ▼
                                 NNLS Convex Simplex Stacking
                                  ŷ_blend = ∑ w_k · ŷ_k  (w_k ≥ 0, ∑w = 1)
                                                │
                                                ▼
                                  Target Inverse Bounding
                                     ŷ_raw = expm1(max(0, ŷ_blend))
                                                │
                                                ▼
                                    Post-Hoc Linear Calibration
                                       ŷ_final = a · ŷ_raw + b
                                                │
                                                ▼
                                 Direct Multi-Step Forecast:
                                 PM2.5, O3, NO2 at t + h (µg/m³)
```

---

## ⚡ Fast 15-Minute Reproduction Guide

The complete pipeline requires **no GPU** and runs on standard CPU hardware:

```bash
# 1. Enter the code workspace
cd MODEL/code

# 2. Install dependencies
pip install -r requirements.txt

# 3. Stage 1: Build coupled features + targets (~9 seconds)
python 01_build_coupled_features.py

# 4. Stage 2: Train all 21 models + NNLS stacking (~14 minutes on CPU)
python 02_train_coupled_ensemble.py

# 5. Stage 3: Export production bundles + golden fixtures (~12 seconds)
python 03_export_v2_bundles.py
```

---
*Certified Master Reference — Smart India Hackathon 2026 (SIH26082)*
