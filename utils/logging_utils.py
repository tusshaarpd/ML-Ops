"""
Logging setup and in-app log utilities.
"""
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from config import LOG_FILE

_INITIALIZED = False


def setup_logging(level: int = logging.INFO):
    global _INITIALIZED
    if _INITIALIZED:
        return
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    fmt     = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    # File handler with rotation (5 MB × 3 backups)
    fh = logging.handlers.RotatingFileHandler(
        str(LOG_FILE), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    fh.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    ch.setLevel(logging.WARNING)  # only warnings+ to console to keep UI clean

    root.addHandler(fh)
    root.addHandler(ch)
    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def read_log_tail(n_lines: int = 100) -> list[str]:
    """Return the last n lines from the log file."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-n_lines:]]
    except FileNotFoundError:
        return ["Log file not found – no events logged yet."]


def generate_pipeline_logs(stage: str) -> list[str]:
    """Return synthetic pipeline execution logs for a given stage."""
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    templates = {
        "data_ingestion": [
            f"[{ts}] INFO  | Pipeline | Starting data ingestion from source",
            f"[{ts}] INFO  | Pipeline | Loaded 2000 records from synthetic generator",
            f"[{ts}] INFO  | Pipeline | Schema validation passed (12 columns)",
            f"[{ts}] INFO  | Pipeline | Data ingestion complete in 0.34s",
        ],
        "data_validation": [
            f"[{ts}] INFO  | Validator | Running null checks … 0.8% missing found",
            f"[{ts}] INFO  | Validator | Running duplicate checks … 12 duplicates found",
            f"[{ts}] WARN  | Validator | 3 outliers detected in 'amount' column",
            f"[{ts}] INFO  | Validator | Schema check passed",
            f"[{ts}] INFO  | Validator | Data quality score: 94/100",
        ],
        "feature_store": [
            f"[{ts}] INFO  | FeatureStore | Fetching feature set v3.2",
            f"[{ts}] INFO  | FeatureStore | 18 features loaded for use case 'churn'",
            f"[{ts}] INFO  | FeatureStore | Feature freshness check passed (< 24h)",
        ],
        "training": [
            f"[{ts}] INFO  | Trainer | Starting training with RandomForest (n_estimators=100)",
            f"[{ts}] INFO  | Trainer | Cross-validation fold 1/5 … F1=0.874",
            f"[{ts}] INFO  | Trainer | Cross-validation fold 2/5 … F1=0.882",
            f"[{ts}] INFO  | Trainer | Cross-validation fold 3/5 … F1=0.868",
            f"[{ts}] INFO  | Trainer | Cross-validation fold 4/5 … F1=0.877",
            f"[{ts}] INFO  | Trainer | Cross-validation fold 5/5 … F1=0.871",
            f"[{ts}] INFO  | Trainer | Mean CV F1 = 0.874 ± 0.005",
            f"[{ts}] INFO  | Trainer | Training complete in 4.21s",
        ],
        "evaluation": [
            f"[{ts}] INFO  | Evaluator | Running test set evaluation",
            f"[{ts}] INFO  | Evaluator | Accuracy=0.887, Precision=0.885, Recall=0.888, F1=0.884",
            f"[{ts}] INFO  | Evaluator | ROC-AUC = 0.951",
            f"[{ts}] INFO  | Evaluator | Model exceeds threshold (F1 > 0.80) ✓",
        ],
        "registry": [
            f"[{ts}] INFO  | Registry | Model artifact saved to models/churn_rf_v2.0.pkl",
            f"[{ts}] INFO  | Registry | Model registered: ChurnPredictor v2.0 → Development",
            f"[{ts}] INFO  | Registry | Metadata written to SQLite registry",
        ],
        "deployment": [
            f"[{ts}] INFO  | Deployer | Staging deployment triggered",
            f"[{ts}] INFO  | Deployer | Health check passed – /predict endpoint responsive",
            f"[{ts}] INFO  | Deployer | Canary traffic set to 10%",
            f"[{ts}] INFO  | Deployer | Deployment to Production approved",
        ],
        "monitoring": [
            f"[{ts}] INFO  | Monitor | Performance check: Accuracy=0.881 (baseline=0.887)",
            f"[{ts}] WARN  | Monitor | Accuracy dropped 0.6% – within tolerance",
            f"[{ts}] INFO  | Monitor | PSI score for 'age' = 0.04 (Stable)",
            f"[{ts}] WARN  | Monitor | PSI score for 'usage_drop_pct' = 0.19 (Moderate drift)",
            f"[{ts}] INFO  | Monitor | Drift report saved to logs/drift_2025-01-15.json",
        ],
        "retraining": [
            f"[{ts}] INFO  | Retrainer | Drift threshold exceeded – scheduling retraining",
            f"[{ts}] INFO  | Retrainer | New training job submitted (job_id=RT-0042)",
            f"[{ts}] INFO  | Retrainer | Awaiting approval from ML Engineer",
        ],
    }
    return templates.get(stage, [f"[{ts}] INFO | No logs for stage '{stage}'"])
