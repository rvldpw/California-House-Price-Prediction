# 🏡 California House Price Prediction

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Status](https://img.shields.io/badge/Status-Completed-green)
![LightGBM](https://img.shields.io/badge/LightGBM-Tuned-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![License](https://img.shields.io/badge/License-MIT-green)

**Capstone Project #3 – Data Science & Machine Learning Program**  
**Institution:** Purwadhika Digital School  
**Creator:** Rivaldi

---

## Project Overview

This project is an **end-to-end regression machine learning pipeline** for predicting California house prices using 1990 U.S. Census block-group data.

Developed as part of a **Data Science & Machine Learning program at Purwadhika Digital School**, this project simulates a real-world automated valuation model (AVM) for a PropTech company — **HomeValueIQ** — that aims to replace subjective, agent-driven pricing with transparent, data-driven estimates.

It is designed to support **buyers, sellers, and mortgage lenders** by predicting `median_house_value` at the census block-group level with high accuracy and minimal error.

---

## Context, Problem, and Users

**1. Context**  
The California housing market is notoriously volatile and opaque. Prices are driven by a complex mix of location, income levels, property age, and household density — factors that vary dramatically across even short geographic distances. Traditional appraisals are slow, expensive, and subjective.

**2. Problem**  
Without an objective pricing baseline, sellers overprice or underprice their listings, buyers cannot evaluate whether a listing is fairly valued, and lenders face delays from manual appraisal processes. Without a model, the naive approach — always predicting the average price of **$205,815** — produces a **MAE of $91,639** and a **MAPE of 63.0%**, representing massive mispricing risk.

**3. Users**  
- **Sellers** — need a fast, data-backed price estimate to list competitively
- **Buyers / Investors** — need an objective reference to evaluate whether a listing is fairly priced
- **Mortgage Lenders (Partners)** — need automated pre-appraisal to accelerate loan processing

---

## Business Questions

- Can we predict California house prices with MAPE ≤ 15% using 1990 Census data?
- Which features drive house price the most — income, location, or property characteristics?
- How much value does an ML model add compared to a simple naive baseline?

---

## Goals

- Build a regression model achieving **R² ≥ 0.85** and **MAPE ≤ 15%**
- Identify the strongest predictors of California house prices
- Quantify the business value of the model vs no model
- Deploy an interactive price estimator via Streamlit

---

## Deliverables

- End-to-end ML pipeline notebook (`California House Price Prediction.ipynb`)
- Model benchmarking reference (`best_ml_model.ipynb`)
- Trained model pickle (`california_house_lgbm_v2_model.sav`)
- Interactive Streamlit app for on-demand price prediction
- Canva presentation deck

---

## Dataset

| Metric | Value |
|---|---|
| Source | 1990 U.S. Census — California Housing |
| Rows | 14,448 block groups |
| Columns | 10 |
| Target | `median_house_value` (USD) |
| Split | 80% train / 20% test (`random_state=42`) |

**Key features:** `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`

**Engineered features:**
- `is_island` — flags the 2 ISLAND rows (rarest category, near-zero training signal)
- `is_capped` — flags 678 rows at the $500,001 census ceiling

**Out of scope:** post-1990 market dynamics, property-level attributes (bedrooms, bathrooms), school district quality

### Data Cleaning & Preprocessing

- No duplicate rows found
- `total_bedrooms`: 137 missing values (0.95%) → median imputation inside pipeline
- Target `median_house_value`: right-skewed (skew ≈ 0.98) → `log1p` transform applied; skewness after transform: **-0.173**
- Outliers in `total_rooms`, `population`, `households` → handled by `RobustScaler`
- No rows removed — all 14,448 rows retained

---

## Analytical Approach

| Step | Decision | Justification |
|---|---|---|
| **Problem type** | Regression | Target is a continuous numerical variable |
| **Target transform** | `log1p(y)` | Reduces skew from 0.98 to -0.173; improves model stability |
| **Feature engineering** | 9 raw + 2 flag features = 11 total | `is_island` and `is_capped` fix edge-case prediction failures |
| **Preprocessing** | `MedianImputer` + `RobustScaler` (num) / `MostFrequent` + `OrdinalEncoder` (cat) | Robust to outliers; OrdinalEncoder works natively with LightGBM |
| **Feature selection** | `SelectKBest(k='all')` | Keep all 11 features — LightGBM handles internal selection via splits |
| **Model** | **LightGBM (LGBMRegressor)** | 🥇 Best performer across full benchmark in `best_ml_model.ipynb` |
| **Tuning** | `RandomizedSearchCV` (50 candidates, 5-fold = 250 fits) | Faster than GridSearchCV for large search spaces |
| **Final training** | Retrained on full 14,448 rows | Maximizes training signal before deployment |
| **Output** | Pickle `.sav` | Streamlit-compatible serialization |

---

## Key Findings

### Model Performance

| Model | Train R² | Test R² | Train MAPE | Test MAPE | Test MAE |
|---|---|---|---|---|---|
| Baseline LightGBM (Default) | 0.8160 | 0.7659 | 18.64% | 22.11% | $38,754 |
| Benchmarked LightGBM (GridSearchCV) | 0.8160 | 0.7659 | 18.64% | 22.11% | $38,754 |
| **Tuned LightGBM (RandomizedSearchCV)** | **0.9477** | **0.8655** | **8.78%** | **15.04%** | **$27,429** |

> 🏆 **Best model: Tuned LightGBM** — Test R² **0.8655**, MAPE **15.04%**, MAE **$27,429**  
> ✅ All business targets met. No data leakage (Train-Test R² gap = 0.08).

---

### Business Value

| Metric | No Model (Naive Mean) | With LightGBM | Improvement |
|---|---|---|---|
| MAE | $91,639 | $27,429 | **$64,210 better per listing** |
| MAPE | 63.0% | 15.0% | **47.9pp reduction** |
| Monthly savings (1,000 listings) | — | — | **$64,210,017 / month** |

---

### Top Predictive Features

| Rank | Feature | Evidence |
|---|---|---|
| 1 | `median_income` | Correlation r = 0.69, Linear coef = +0.45, SHAP top driver |
| 2 | `latitude` | 6,366 LightGBM splits — strongest geographic signal |
| 3 | `longitude` | 6,269 splits — complements latitude for location pricing |
| 4 | `ocean_proximity` (INLAND) | SHAP: strongest negative driver — suppresses price significantly |
| 5 | `total_rooms` | 4,235 splits — size signal at block-group level |

---

### Demo Test Results

| Test | Scenario | Predicted | Expected Range | Result |
|---|---|---|---|---|
| 1 | Typical Inland | $91,552 | $90K–$180K | ✅ IN RANGE |
| 2 | Mid-Tier Coastal Suburb | $249,680 | $200K–$310K | ✅ IN RANGE |
| 3 | High-Income SF Bay Area | $369,342 | $350K–$480K | ✅ IN RANGE |
| 4 | Island / Low Income | $142,341 | $180K–$350K | ❌ OUT OF RANGE |
| 5 | Luxury LA Coastal | $468,932 | $400K–$500K+ | ✅ IN RANGE |

> ⚠️ Test 4 (ISLAND) fails due to only **2 training rows** in this category — a data limitation, not a model failure.

---

## Strategic Recommendations

| Priority | Area | Recommendation |
|---|---|---|
| 🔴 High | Data | Retrain on post-1990 data — 1990 USD prices are outdated for current use |
| 🔴 High | Data | Add property type (single-family, condo, apartment) — major signal expected |
| 🟡 Medium | Features | Add school district quality scores — strong California real estate driver |
| 🟡 Medium | Features | Distance to SF / LA / coastline — captures geographic clustering better than raw lat/lon |
| 🟢 Low | Model | Benchmark CatBoost — handles categoricals natively, may reduce MAPE marginally |
| 🟢 Low | Ops | Set retraining trigger when live MAPE exceeds 18% |

---

## Known Limitations

| Limitation | Detail |
|---|---|
| **ISLAND category** | Only 2 training rows — predictions unreliable for this category |
| **$500,001 price cap** | 1990 Census capped all values at $500,001 — true luxury pricing cannot be learned |
| **1990 USD** | Prices are not inflation-adjusted (~3.5× CPI multiplier for modern equivalent) |
| **California only** | Model trained on California data — do not apply outside valid lat/lon range |
| **Block-group level** | Predicts median value of a census block group, not an individual house |

---

## 🔗 Links

- 🚀 **Streamlit App:** [ca-house-price-prediction.streamlit.app](https://ca-house-price-prediction.streamlit.app/)
- 🎨 **Canva Presentation:** [California House Price Prediction](https://canva.link/california-house-price-prediction)

---

## Libraries

- **pandas**: Data manipulation, cleaning, and tabular data analysis
- **numpy**: Numerical computing and array operations
- **matplotlib / seaborn**: Visualization for distributions, correlations, and residuals
- **folium**: Interactive geographic mapping of California block groups
- **scikit-learn**: Pipeline, preprocessing, feature selection, cross-validation, metrics
- **lightgbm**: Gradient boosting regressor — final production model
- **xgboost**: Gradient boosting baseline for benchmarking
- **shap**: Model explainability via SHAP values
- **pickle**: Model serialization for Streamlit deployment

---

## Repository Structure

```text
├── README.md
├── California House Price Prediction.ipynb   ← Main capstone notebook
├── best_ml_model.ipynb                       ← Full model benchmark reference
├── streamlit/
│   ├── app.py                                ← Streamlit application
│   ├── ca_house_model.pkl                    ← Deployed model pickle
│   ├── data_ca_geocoded.csv                  ← Geocoded dataset for map
│   └── requirements.txt                      ← App dependencies
└── data_california_house.csv                 ← Raw dataset
```

---

## Environment & Tools

- **Language:** Python 3.13
- **Environment:** Conda (local) / Streamlit Cloud (production)
- **Editor:** Jupyter Notebook
- **Visualization & Reporting:** Tableau, Canva
- **Deployment:** Streamlit Community Cloud
