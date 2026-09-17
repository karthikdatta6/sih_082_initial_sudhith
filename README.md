# 🏛️ SIH 2026 (SIH26082) — Air Pollution–Weather Coupled Forecasting System

> **Official Systems Architecture, Modeling Pipeline & Research Foundation**  
> **Problem Statement ID:** SIH26082  
> **Nodal Ministry / Agency:** Ministry of Earth Sciences (MoES) / NCMRWF  
> **Target Domain:** Delhi National Capital Region (NCR), India — 10 Canonical CPCB CAAQMS Stations  
> **Temporal Foundation:** 263,040 Time Steps (3 Unbroken Years: January 1, 2023 – December 31, 2025)  
> **Forecast Horizons:** Direct Multi-Step at +1h, +3h, +6h, +12h, +24h, +48h, +72h  
> **Target Chemical Species:** Fine Particulate Matter ($PM_{2.5}$), Ground-Level Ozone ($O_3$), Nitrogen Dioxide ($NO_2$)  

---

## 📑 Repository Structure & Document Sitemap

```
sih_082_initial_sudhith/
├── MODEL/                                            # Certified SIH26082 Coupled AI/ML Engine
│   ├── README.md                                     # Directory index, architecture & quickstart
│   ├── 01_APPROACH_RESEARCH_AND_INITIAL_PLAN.md      # Research, chemistry, initial plan & methodology
│   ├── 02_RESULTS_AND_BENCHMARKS_EXPLAINED.md        # 21-model benchmark matrix, ablations & forensic analysis
│   ├── 03_OUTPUTS_INTEGRATION_AND_FUTURE_ROADMAP.md  # Outputs catalog, backend/frontend integration & future roadmap
│   ├── code/                                         # Self-contained, reproducible Python code pipeline
│   │   ├── coupled_physics.py                        # Algebraic physics kernel (ITI, Kt, corrected NW flux)
│   │   ├── 01_build_coupled_features.py              # Stage 1: Feature generation & direct targets
│   │   ├── 02_train_coupled_ensemble.py              # Stage 2: LightGBM ensemble & NNLS stacking
│   │   ├── 03_export_v2_bundles.py                   # Stage 3: Production bundle packaging
│   │   ├── 04_train_ensemble_v3.py                   # Algorithmic diversity / XGBoost-CUDA exploration
│   │   └── requirements.txt                          # Python package dependencies
│   └── results/                                      # Frozen verification metrics, schemas & logs
│       ├── metrics_by_horizon.json                   # Detailed evaluation JSON for all 21 models
│       ├── ablation_table.csv                        # Feature block ablation delta scores
│       ├── feature_catalog.csv                       # Master 78-feature tabular data dictionary
│       ├── feature_schema_v2.json                    # Canonical JSON feature schema
│       └── training_run.log                          # Verbatim stdout/stderr execution training log
│
├── MASTER_DOCUMENTATION_AND_PROJECT_JOURNEY.md       # Master Technical Bible & Complete Project Retrospective
├── walkthough.md                                     # Exhaustive Codebase Walkthrough & File-by-File Guide
└── RESEARCH PAPERS/                                  # 13 Peer-Reviewed Atmospheric & Machine Learning Literature Studies
```

---

## 🎯 Key Breakthroughs & Capabilities in Version 2.0.0

1. **Two-Way Chemical-Meteorological Coupled Physics:**
   * Replaced static one-way weather features with dynamic feedback indices:
     * **Atmospheric Inversion Trapping Index ($ITI$):** Captures nocturnal boundary layer collapse and wind stagnation ceiling.
     * **Clearness Index ($K_t$):** Instantaneous proxy of aerosol optical depth solar extinction.
     * **Corrected North-West Stubble Plume Inflow:** Fixed the legacy mathematical projection bug, accurately measuring smoke advection from Punjab/Haryana.
     * **Deterministic Astronomical Forcing:** Precise target-time solar zenith angle and diurnal harmonics evaluated at $t + h$.
2. **Brand New $PM_{2.5}$ & +72h Forecasting Capabilities:**
   * $PM_{2.5}$ achieved **$R^2 = 0.9634$** and Willmott Index $d = 0.9906$ at $+1\text{h}$ on held-out test data.
   * Full 3-day (+72h) direct forecasting capability operational across all pollutants.
3. **Significant Accuracy Leaps over Certified Legacy Baseline:**
   * Ground-Level Ozone ($O_3$) $+1\text{h}$ explained variance jumped by **$+0.0512$ $R^2$** (from $0.8689$ to $0.9201$).
   * Nitrogen Dioxide ($NO_2$) $+1\text{h}$ improved by **$+0.0187$ $R^2$** (to $0.9378$).
   * Skill vs. persistence is **positive across all 21 models** (no horizon loses to naive persistence).
4. **Zero Lookahead Leakage Certified:**
   * Split chronologically: Train ($2023\text{--}2024$), Validation (Q1-Q2 $2025$), Test (Q3-Q4 $2025$).
   * Test targets are pure un-imputed IEEE 754 `NaN`. Bit-level causality verified.

---

## ⚡ Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Sudhith/sih_082_initial_sudhith.git
cd sih_082_initial_sudhith/MODEL/code

# 2. Install requirements
pip install -r requirements.txt

# 3. Build features, train models, and export bundles (~15 minutes on CPU)
python 01_build_coupled_features.py
python 02_train_coupled_ensemble.py
python 03_export_v2_bundles.py
```

---
*Maintained by Team AIRO2 — Smart India Hackathon 2026*
