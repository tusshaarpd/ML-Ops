"""
Model Registry backed by SQLite.
Supports CRUD operations, stage transitions, and version comparison.
"""
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from config import DB_PATH, STAGES

logger = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist and seed with sample entries."""
    conn = _connect()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name  TEXT NOT NULL,
            version     TEXT NOT NULL,
            use_case    TEXT NOT NULL,
            algo_key    TEXT NOT NULL,
            stage       TEXT NOT NULL DEFAULT 'Development',
            accuracy    REAL,
            f1          REAL,
            rmse        REAL,
            metrics     TEXT,
            params      TEXT,
            owner       TEXT,
            description TEXT,
            model_path  TEXT,
            created_at  TEXT,
            updated_at  TEXT,
            UNIQUE(model_name, version)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            action      TEXT,
            model_name  TEXT,
            version     TEXT,
            from_stage  TEXT,
            to_stage    TEXT,
            actor       TEXT,
            notes       TEXT
        )
    """)
    conn.commit()

    # Seed with realistic sample models if empty
    if cur.execute("SELECT COUNT(*) FROM models").fetchone()[0] == 0:
        _seed_registry(cur)
        conn.commit()
    conn.close()


def _seed_registry(cur):
    samples = [
        ("SentimentClassifier",  "1.0", "sentiment", "lr",  "Production", 0.8920, 0.8890, None,
         '{"accuracy":0.892,"precision":0.891,"recall":0.889,"f1":0.889}',
         '{"C":1.0}', "Alice Johnson", "Baseline TF-IDF + LR model for review sentiment"),
        ("SentimentClassifier",  "2.0", "sentiment", "rf",  "Staging",    0.9120, 0.9085, None,
         '{"accuracy":0.912,"precision":0.913,"recall":0.910,"f1":0.909}',
         '{"n_estimators":200}', "Alice Johnson", "RF upgrade with more features"),
        ("SpamFilter",           "1.0", "spam",      "nb",  "Production", 0.9670, 0.9640, None,
         '{"accuracy":0.967,"precision":0.968,"recall":0.966,"f1":0.964}',
         '{"alpha":1.0}', "Bob Smith", "Multinomial NB spam classifier"),
        ("ChurnPredictor",       "1.0", "churn",     "rf",  "Production", 0.8350, 0.8310, None,
         '{"accuracy":0.835,"precision":0.840,"recall":0.830,"f1":0.831}',
         '{"n_estimators":100}', "Carol Lee", "Random Forest churn prediction"),
        ("ChurnPredictor",       "2.0", "churn",     "gb",  "Staging",    0.8690, 0.8650, None,
         '{"accuracy":0.869,"precision":0.870,"recall":0.868,"f1":0.865}',
         '{"n_estimators":200,"learning_rate":0.05}', "Carol Lee", "GB upgrade – better recall"),
        ("FraudDetector",        "1.0", "fraud",     "rf",  "Production", 0.9540, 0.9420, None,
         '{"accuracy":0.954,"precision":0.960,"recall":0.942,"f1":0.942}',
         '{"n_estimators":150}', "David Park", "High-precision fraud classifier"),
        ("SalesForecaster",      "1.0", "sales",     "rfr", "Production", None,    None, 42.30,
         '{"rmse":42.3,"mae":31.2,"r2":0.964,"mape":5.8}',
         '{"n_estimators":100}', "Eva Chen", "RF sales forecast model"),
        ("HRAttritionModel",     "1.0", "hr",        "gb",  "Production", 0.8820, 0.8790, None,
         '{"accuracy":0.882,"precision":0.884,"recall":0.879,"f1":0.879}',
         '{"n_estimators":100}', "Frank Wu", "Gradient Boosting attrition predictor"),
        ("HRAttritionModel",     "0.9", "hr",        "lr",  "Archived",   0.8140, 0.8100, None,
         '{"accuracy":0.814,"precision":0.815,"recall":0.810,"f1":0.810}',
         '{"C":1.0}', "Frank Wu", "Deprecated baseline"),
    ]
    now = datetime.utcnow().isoformat()
    for s in samples:
        cur.execute("""
            INSERT OR IGNORE INTO models
            (model_name, version, use_case, algo_key, stage, accuracy, f1, rmse,
             metrics, params, owner, description, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (*s, now, now))


# ── CRUD ──────────────────────────────────────────────────────────────────────

def register_model(
    model_name: str,
    version: str,
    use_case: str,
    algo_key: str,
    metrics: dict,
    params: dict,
    owner: str,
    description: str = "",
    model_path: str = "",
    stage: str = "Development",
) -> int:
    conn = _connect()
    cur  = conn.cursor()
    now  = datetime.utcnow().isoformat()
    acc  = metrics.get("accuracy")
    f1   = metrics.get("f1")
    rmse = metrics.get("rmse")
    cur.execute("""
        INSERT OR REPLACE INTO models
        (model_name, version, use_case, algo_key, stage, accuracy, f1, rmse,
         metrics, params, owner, description, model_path, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (model_name, version, use_case, algo_key, stage, acc, f1, rmse,
          json.dumps(metrics), json.dumps(params), owner, description, model_path, now, now))
    row_id = cur.lastrowid
    _log_action(cur, "REGISTER", model_name, version, "", stage, owner, description)
    conn.commit()
    conn.close()
    logger.info("Registered model %s v%s", model_name, version)
    return row_id


def get_all_models() -> pd.DataFrame:
    conn = _connect()
    df   = pd.read_sql("SELECT * FROM models ORDER BY updated_at DESC", conn)
    conn.close()
    return df


def get_model(model_name: str, version: str) -> dict:
    conn = _connect()
    row  = conn.execute(
        "SELECT * FROM models WHERE model_name=? AND version=?",
        (model_name, version)
    ).fetchone()
    conn.close()
    return dict(row) if row else {}


def promote_model(model_name: str, version: str, to_stage: str, actor: str, notes: str = ""):
    conn = _connect()
    cur  = conn.cursor()
    row  = cur.execute(
        "SELECT stage FROM models WHERE model_name=? AND version=?",
        (model_name, version)
    ).fetchone()
    from_stage = row["stage"] if row else "Unknown"
    cur.execute(
        "UPDATE models SET stage=?, updated_at=? WHERE model_name=? AND version=?",
        (to_stage, datetime.utcnow().isoformat(), model_name, version)
    )
    _log_action(cur, "PROMOTE", model_name, version, from_stage, to_stage, actor, notes)
    conn.commit()
    conn.close()
    logger.info("Promoted %s v%s → %s", model_name, version, to_stage)


def delete_model(model_name: str, version: str, actor: str):
    conn = _connect()
    cur  = conn.cursor()
    cur.execute(
        "DELETE FROM models WHERE model_name=? AND version=?",
        (model_name, version)
    )
    _log_action(cur, "DELETE", model_name, version, "", "Deleted", actor, "")
    conn.commit()
    conn.close()


def get_audit_log() -> pd.DataFrame:
    conn = _connect()
    df   = pd.read_sql(
        "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 200", conn
    )
    conn.close()
    return df


def _log_action(cur, action, model_name, version, from_stage, to_stage, actor, notes):
    cur.execute("""
        INSERT INTO audit_log (timestamp, action, model_name, version, from_stage, to_stage, actor, notes)
        VALUES (?,?,?,?,?,?,?,?)
    """, (datetime.utcnow().isoformat(), action, model_name, version, from_stage, to_stage, actor, notes))


def get_production_models() -> pd.DataFrame:
    conn = _connect()
    df   = pd.read_sql("SELECT * FROM models WHERE stage='Production'", conn)
    conn.close()
    return df


def compare_versions(model_name: str) -> pd.DataFrame:
    conn = _connect()
    df   = pd.read_sql(
        "SELECT version, stage, accuracy, f1, rmse, metrics, created_at FROM models WHERE model_name=? ORDER BY version",
        conn, params=(model_name,)
    )
    conn.close()
    return df
