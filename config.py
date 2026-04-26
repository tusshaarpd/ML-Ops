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
    /* ── Sidebar background ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(170deg, #0f172a 0%, #1e3a5f 60%, #1e40af 100%) !important;
    }

    /* ── All text inside the sidebar ── */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e2e8f0 !important;
    }

    /* ── Sidebar widget labels (selectbox, slider, etc.) ── */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stTextArea label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* ── Sidebar selectbox / input backgrounds ── */
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stMultiSelect > div > div {
        background-color: rgba(255,255,255,0.12) !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }

    /* ── Sidebar slider track ── */
    [data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
        background: rgba(255,255,255,0.2) !important;
    }

    /* ── Sidebar divider ── */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2) !important;
    }

    /* ── Sidebar nav page links ── */
    [data-testid="stSidebarNav"] a,
    [data-testid="stSidebarNav"] span {
        color: #bfdbfe !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebarNav"] a:hover span {
        color: #ffffff !important;
    }
    [data-testid="stSidebarNav"] [aria-selected="true"] span {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* ── Sidebar button ── */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.15) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.25) !important;
    }

    /* ── Main page metric cards ── */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg,#dbeafe,#e0f2fe);
        border: 1px solid #93c5fd;
        border-radius: 12px;
        padding: 14px 18px;
    }
    div[data-testid="metric-container"] label {
        color: #1e40af !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }
    div[data-testid="metric-container"] [data-testid="metric-value"] {
        color: #1e3a8a !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* ── Main buttons ── */
    .stButton > button {
        background: linear-gradient(90deg,#1e40af,#0891b2) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.45rem 1.2rem !important;
        transition: opacity 0.15s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* ── Page headers ── */
    h1 { color: #1e40af !important; font-weight: 800 !important; }
    h2 { color: #1e3a8a !important; font-weight: 700 !important; }
    h3 { color: #1e40af !important; font-weight: 700 !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.88rem;
    }

    /* ── Alert banners ── */
    .alert-danger  { background:#fee2e2; border-left:5px solid #dc2626; padding:12px 16px; border-radius:8px; margin:6px 0; }
    .alert-warning { background:#fef3c7; border-left:5px solid #d97706; padding:12px 16px; border-radius:8px; margin:6px 0; }
    .alert-success { background:#d1fae5; border-left:5px solid #059669; padding:12px 16px; border-radius:8px; margin:6px 0; }
    .alert-info    { background:#e0f2fe; border-left:5px solid #0891b2; padding:12px 16px; border-radius:8px; margin:6px 0; }

    /* ── Explanation / callout boxes ── */
    .explain-box {
        background: linear-gradient(135deg,#f0f9ff,#e0f2fe);
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0 16px;
        font-size: 0.9rem;
        color: #0c4a6e;
        line-height: 1.6;
    }
    .explain-box b { color: #0369a1; }

    /* ── Stage badges ── */
    .badge-prod    { background:#059669; color:#fff; padding:3px 12px; border-radius:12px; font-size:0.75rem; font-weight:700; }
    .badge-staging { background:#d97706; color:#fff; padding:3px 12px; border-radius:12px; font-size:0.75rem; font-weight:700; }
    .badge-dev     { background:#6b7280; color:#fff; padding:3px 12px; border-radius:12px; font-size:0.75rem; font-weight:700; }
    .badge-arch    { background:#dc2626; color:#fff; padding:3px 12px; border-radius:12px; font-size:0.75rem; font-weight:700; }

    /* ── Dividers ── */
    hr { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }

    /* ── Code ── */
    code { background:#1e293b; color:#38bdf8; padding:2px 7px; border-radius:5px; font-size:0.85em; }
</style>
"""
