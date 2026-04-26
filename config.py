"""
MLOps Showcase Application - Central Configuration
All constants, seeds, feature schemas, and defaults live here.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"
DB_DIR     = BASE_DIR / "db"

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR, DB_DIR]:
    d.mkdir(exist_ok=True)

DB_PATH    = DB_DIR / "mlops_registry.db"
LOG_FILE   = LOGS_DIR / "mlops_app.log"

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── App Meta ───────────────────────────────────────────────────────────────────
APP_TITLE   = "MLOps Showcase Platform"
APP_ICON    = "🚀"
APP_VERSION = "2.0.0"
COMPANY     = "DataOps Corp"

# ── Brand Colors ───────────────────────────────────────────────────────────────
COLOR_PRIMARY   = "#1E40AF"  # Deep blue
COLOR_SUCCESS   = "#059669"  # Emerald green
COLOR_WARNING   = "#D97706"  # Amber
COLOR_DANGER    = "#DC2626"  # Red
COLOR_INFO      = "#0891B2"  # Cyan
COLOR_NEUTRAL   = "#6B7280"  # Gray
COLOR_BG_CARD   = "#F0F4FF"
COLOR_BG_DARK   = "#1F2937"

# ── Use Cases ──────────────────────────────────────────────────────────────────
USE_CASES = {
    "sentiment": {
        "name": "Sentiment Analysis",
        "icon": "💬",
        "description": "Classify customer reviews as Positive / Neutral / Negative",
        "task": "classification",
        "n_samples": 1200,
        "target": "sentiment",
        "classes": ["Positive", "Neutral", "Negative"],
    },
    "spam": {
        "name": "Spam Detection",
        "icon": "📧",
        "description": "Classify SMS / email messages as Spam or Ham",
        "task": "classification",
        "n_samples": 1500,
        "target": "label",
        "classes": ["Ham", "Spam"],
    },
    "churn": {
        "name": "Customer Churn",
        "icon": "📉",
        "description": "Predict telecom customer churn risk",
        "task": "classification",
        "n_samples": 2000,
        "target": "churn",
        "classes": ["Retained", "Churned"],
    },
    "fraud": {
        "name": "Fraud Detection",
        "icon": "🔍",
        "description": "Flag fraudulent financial transactions",
        "task": "classification",
        "n_samples": 3000,
        "target": "is_fraud",
        "classes": ["Legit", "Fraud"],
    },
    "sales": {
        "name": "Sales Forecasting",
        "icon": "📈",
        "description": "Forecast future daily sales volumes",
        "task": "regression",
        "n_samples": 730,
        "target": "sales",
        "classes": [],
    },
    "hr": {
        "name": "HR Attrition",
        "icon": "👥",
        "description": "Predict employee likelihood of leaving the company",
        "task": "classification",
        "n_samples": 1500,
        "target": "attrition",
        "classes": ["Stay", "Leave"],
    },
}

# ── Algorithm Catalogue ────────────────────────────────────────────────────────
CLASSIFIERS = {
    "Logistic Regression":     "lr",
    "Random Forest":           "rf",
    "Gradient Boosting":       "gb",
    "Naive Bayes":             "nb",
    "XGBoost":                 "xgb",   # graceful fallback
    "Isolation Forest":        "if",    # anomaly detection
}

REGRESSORS = {
    "Linear Regression":       "linreg",
    "Random Forest Regressor": "rfr",
    "Gradient Boosting Regressor": "gbr",
}

# ── Default Hyperparameters ────────────────────────────────────────────────────
DEFAULT_HYPERPARAMS = {
    "lr":  {"C": 1.0,      "max_iter": 1000},
    "rf":  {"n_estimators": 100, "max_depth": 10},
    "gb":  {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    "nb":  {"alpha": 1.0},
    "rfr": {"n_estimators": 100, "max_depth": 10},
    "gbr": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
}

# ── Training Defaults ─────────────────────────────────────────────────────────
DEFAULT_TEST_SIZE  = 0.2
DEFAULT_CV_FOLDS   = 5
MIN_TRAIN_SAMPLES  = 50

# ── Drift Thresholds ──────────────────────────────────────────────────────────
PSI_LOW_THRESHOLD    = 0.1   # no significant change
PSI_MEDIUM_THRESHOLD = 0.2   # moderate change – investigate
# above 0.2 = significant drift – retrain

KS_ALPHA = 0.05              # p-value threshold for KS test

# ── Monitoring Defaults ───────────────────────────────────────────────────────
PERF_HISTORY_DAYS   = 30
DRIFT_CHECK_DAYS    = 7
ALERT_ACCURACY_DROP = 0.05   # trigger alert if accuracy drops by this amount

# ── Model Registry Stages ─────────────────────────────────────────────────────
STAGES = ["Development", "Staging", "Production", "Archived"]

# ── Deployment Simulation ─────────────────────────────────────────────────────
LATENCY_RANGE_MS = (12, 180)   # simulated inference latency window

# ── Governance ────────────────────────────────────────────────────────────────
APPROVAL_LEVELS = ["Data Scientist", "ML Engineer", "Risk Manager", "CTO"]
BIAS_METRICS    = ["Demographic Parity", "Equal Opportunity", "Calibration"]

# ── CSS Theme (injected into Streamlit pages) ─────────────────────────────────
CUSTOM_CSS = """
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: linear-gradient(160deg,#0f172a 0%,#1e3a5f 100%); }
    [data-testid="stSidebar"] .css-1d391kg { color: #e2e8f0; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg,#1e40af22,#0891b222);
        border: 1px solid #1e40af55;
        border-radius: 12px;
        padding: 12px 16px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg,#1e40af,#0891b2);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Headers */
    h1 { color: #1e40af; }
    h2 { color: #1e3a8a; }
    h3 { color: #1e40af; }

    /* Alert banners */
    .alert-danger  { background:#fee2e2; border-left:4px solid #dc2626; padding:10px 14px; border-radius:6px; }
    .alert-warning { background:#fef3c7; border-left:4px solid #d97706; padding:10px 14px; border-radius:6px; }
    .alert-success { background:#d1fae5; border-left:4px solid #059669; padding:10px 14px; border-radius:6px; }
    .alert-info    { background:#e0f2fe; border-left:4px solid #0891b2; padding:10px 14px; border-radius:6px; }

    /* Stage badges */
    .badge-prod    { background:#059669; color:white; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
    .badge-staging { background:#d97706; color:white; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
    .badge-dev     { background:#6b7280; color:white; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
    .badge-arch    { background:#dc2626; color:white; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }

    /* Divider */
    hr { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }

    /* Code blocks */
    code { background: #1e293b; color:#38bdf8; padding: 2px 6px; border-radius: 4px; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; padding: 8px 20px; font-weight: 600; }

    /* Scrollable table */
    .scrollable-table { overflow-x: auto; border-radius: 8px; }
</style>
"""
