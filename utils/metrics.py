"""
Metric computation helpers – wraps sklearn with consistent output dicts.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve,
    roc_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
)


def classification_metrics(y_true, y_pred, y_proba=None, average="weighted") -> dict:
    m = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average=average, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average=average, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, average=average, zero_division=0), 4),
    }
    if y_proba is not None:
        try:
            n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2
            if n_classes == 2:
                proba_1d = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                m["roc_auc"] = round(roc_auc_score(y_true, proba_1d), 4)
                m["avg_precision"] = round(average_precision_score(y_true, proba_1d), 4)
            else:
                m["roc_auc"] = round(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"), 4
                )
        except Exception:
            pass
    return m


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 3),
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
        "mape": round(mape, 2),
    }


def roc_data(y_true, y_proba_pos) -> dict:
    """Return fpr, tpr, thresholds for ROC curve."""
    try:
        fpr, tpr, thr = roc_curve(y_true, y_proba_pos)
        return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}
    except Exception:
        return {}


def pr_data(y_true, y_proba_pos) -> dict:
    """Return precision, recall, thresholds for PR curve."""
    try:
        prec, rec, thr = precision_recall_curve(y_true, y_proba_pos)
        return {"precision": prec.tolist(), "recall": rec.tolist(), "thresholds": thr.tolist()}
    except Exception:
        return {}


def confusion_matrix_data(y_true, y_pred, labels=None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"matrix": cm.tolist(), "labels": labels or sorted(set(y_true))}


def simulate_metric_history(
    base_acc: float = 0.88,
    days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a plausible historical metric trend for monitoring dashboards.
    Introduces slight drift downward in the last 7 days.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")
    noise = rng.normal(0, 0.008, days)
    drift = np.linspace(0, -0.04, days)   # gradual slight degradation
    acc   = np.clip(base_acc + drift + noise, 0.50, 0.99)
    f1    = np.clip(acc - rng.uniform(0.01, 0.03, days), 0.50, 0.99)
    prec  = np.clip(acc + rng.uniform(-0.02, 0.02, days), 0.50, 0.99)
    rec   = np.clip(acc + rng.uniform(-0.03, 0.01, days), 0.50, 0.99)

    return pd.DataFrame({
        "date":      dates,
        "accuracy":  acc.round(4),
        "f1":        f1.round(4),
        "precision": prec.round(4),
        "recall":    rec.round(4),
    })


def simulate_regression_history(
    base_rmse: float = 45.0,
    days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")
    noise = rng.normal(0, 2.0, days)
    drift = np.linspace(0, 8, days)
    rmse  = (base_rmse + drift + noise).clip(10)
    mae   = (rmse * 0.75 + rng.normal(0, 1, days)).clip(5)

    return pd.DataFrame({
        "date": dates,
        "rmse": rmse.round(2),
        "mae":  mae.round(2),
    })
