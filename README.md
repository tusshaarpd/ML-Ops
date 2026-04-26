# 🚀 MLOps Showcase Platform

A production-grade, end-to-end **Machine Learning Operations (MLOps)** platform built with **Python + Streamlit**. Demonstrates the full ML lifecycle across six real-world business use cases — from raw data ingestion to model governance — using realistic synthetic data, real algorithms, interactive dashboards, and enterprise-quality code.

---

## 🎯 What This Platform Demonstrates

| Capability | Details |
|---|---|
| 6 Business Use Cases | Sentiment · Spam · Churn · Fraud · Sales Forecast · HR Attrition |
| Real ML Algorithms | Logistic Regression, Random Forest, Gradient Boosting, Naive Bayes, XGBoost, Isolation Forest |
| Data Engineering | Synthetic generation with noise, missing values, outliers, duplicates |
| Training Engine | Multi-algorithm, hyperparameter config, cross-validation, auto best-model |
| Explainable AI | Global feature importance, local token attributions for text models |
| Live Predictions | Single + batch inference, REST API simulation, latency tracking |
| Monitoring | Performance trends, PSI drift detection, KS tests, data quality |
| Model Registry | SQLite-backed versioning, stage promotions, rollbacks, audit trail |
| Governance | Approval workflows, bias/fairness metrics, responsible AI report |
| Executive Dashboard | KPI cards, alerts, platform health, use-case coverage |
| MLOps Control Center | Full pipeline flow, stage status cards, SLA compliance, log viewer |

---

## 📂 Project Structure

```
ML-Ops/
├── app.py                          # Landing page & Streamlit entry point
├── config.py                       # All configuration, constants, CSS theme
├── requirements.txt
├── README.md
│
├── pages/
│   ├── 1_Home.py                   # Executive KPI dashboard
│   ├── 2_Use_Cases.py              # Dataset exploration for all 6 use cases
│   ├── 3_Training.py               # Model training engine + comparison
│   ├── 4_Predictions.py            # Live predictions, batch, API simulation, XAI
│   ├── 5_MLOps_Control_Center.py   # Pipeline flow, health, logs, SLA
│   ├── 6_Monitoring.py             # Performance, drift, data quality monitoring
│   ├── 7_Model_Registry.py         # Version management, promotions, analytics
│   └── 8_Governance.py             # Approvals, bias, responsible AI, compliance
│
├── utils/
│   ├── __init__.py
│   ├── data_generator.py           # Synthetic dataset generators (all 6 use cases)
│   ├── ml_pipeline.py              # Training, evaluation, predict_single, save/load
│   ├── metrics.py                  # Classification + regression metrics, history sims
│   ├── drift_detection.py          # PSI, KS test, mean shift, drift history
│   ├── model_registry.py           # SQLite-backed registry CRUD
│   ├── logging_utils.py            # Rotating file logger, pipeline log generator
│   └── visualizations.py          # Reusable Plotly chart functions
│
├── models/                         # Saved model artifacts (.pkl)
├── data/                           # Exported datasets
├── logs/                           # Application logs (rotating)
└── db/                             # SQLite registry database
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone <repo_url>
cd ML-Ops
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 🧭 Navigation Guide

| Page | Purpose |
|---|---|
| **Home** | Executive KPI dashboard, active alerts, production model health |
| **Use Cases** | Explore all 6 datasets: distributions, quality, segment analysis |
| **Training** | Train any algorithm on any use case; compare models; register to registry |
| **Predictions** | Live predictions, batch scoring, API simulation (with latency), XAI |
| **MLOps Control Center** | Sankey pipeline flow, stage cards, SLA board, log viewer, run simulation |
| **Monitoring** | Performance trends, PSI + KS drift detection, data quality trend |
| **Model Registry** | All model versions, stage promotions, rollbacks, version comparison |
| **Governance** | Approval workflow, bias/fairness, responsible AI, compliance report |

---

## 🤖 Business Use Cases

### 1. 💬 Sentiment Analysis
- **Data:** 1,200 customer reviews (Positive / Neutral / Negative)
- **Models:** TF-IDF + Logistic Regression, Naive Bayes
- **Features:** Text → TF-IDF bigrams (5,000 features)

### 2. 📧 Spam Detection
- **Data:** 1,500 SMS/email messages (Ham / Spam, 75/25 split)
- **Models:** Multinomial Naive Bayes, Logistic Regression
- **Features:** TF-IDF with stop-word removal

### 3. 📉 Customer Churn
- **Data:** 2,000 telecom customers (Retained / Churned, ~50% churn)
- **Models:** Random Forest, Gradient Boosting, Logistic Regression
- **Features:** Age, Region, Plan, Tenure, Bill, Complaints, Usage Drop, Payment Delays

### 4. 🔍 Fraud Detection
- **Data:** 3,000 transactions (~6% fraud, class imbalanced)
- **Models:** Random Forest, Isolation Forest, XGBoost
- **Features:** Amount, Merchant, Device/Geo mismatch, Velocity, Chargeback history

### 5. 📈 Sales Forecasting
- **Data:** 730 days (2 years) of daily sales with trend + seasonality
- **Models:** Linear Regression, Random Forest Regressor
- **Features:** Date components, Promotions, Holidays, Price changes

### 6. 👥 HR Attrition
- **Data:** 1,500 employees (Stay / Leave, ~35% attrition)
- **Models:** Gradient Boosting
- **Features:** Department, Salary Band, Overtime, Satisfaction, Tenure, Manager Changes

---

## 📊 ML Architecture

```
Raw Data → Data Validation → Feature Engineering → Training
    ↓                                                   ↓
Data Quality                                      Cross-Validation
Report                                                  ↓
                                              Model Evaluation
                                                  ↓        ↓
                                            Registry    Explainability
                                                ↓
                                         Stage: Development
                                                ↓
                                          Approval Gate
                                                ↓
                                         Stage: Staging
                                                ↓
                                          Approval Gate
                                                ↓
                                        Stage: Production
                                                ↓
                                    Monitoring + Drift Detection
                                                ↓
                                    Retraining Trigger (if needed)
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| UI Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| ML Algorithms | scikit-learn, XGBoost |
| Visualization | Plotly Express + Graph Objects |
| Persistence | SQLite (registry), joblib (models) |
| Drift Detection | SciPy (KS test), custom PSI |
| Logging | Python logging + RotatingFileHandler |

---

## 🔧 Configuration

All configuration lives in `config.py`:

- `RANDOM_SEED` — reproducibility seed (default: 42)
- `PSI_MEDIUM_THRESHOLD` — drift alert threshold (default: 0.20)
- `ALERT_ACCURACY_DROP` — performance alert threshold (default: 0.05)
- `LATENCY_RANGE_MS` — simulated inference latency range
- `STAGES` — model lifecycle stages
- `CUSTOM_CSS` — enterprise UI theme

---

## 🏗️ Engineering Quality

- **OOP where useful** — MLPipeline, ModelRegistry, DataGenerator classes
- **Reusable functions** — all chart functions in `visualizations.py`
- **Exception handling** — graceful fallbacks for optional deps (XGBoost, SMOTE)
- **Logging** — rotating file handler + structured log format
- **Deterministic** — all synthetic data uses fixed random seeds
- **PEP8** — clean, commented, well-structured code
- **Caching** — `@st.cache_data` / `@st.cache_resource` for performance
- **Modular** — config, utils, pages are cleanly separated

---

## 📝 License

MIT License — free for educational and commercial use.

---

*Built with ❤️ using Python, Streamlit, scikit-learn, and Plotly*
