"""
Drift detection utilities.
Implements PSI (Population Stability Index) and KS Test.
Also generates simulated drift data for dashboards.
"""
import numpy as np
import pandas as pd
from scipy import stats
from config import PSI_LOW_THRESHOLD, PSI_MEDIUM_THRESHOLD, KS_ALPHA


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between reference (expected) and current (actual)
    distributions. PSI < 0.1: stable, 0.1–0.2: moderate, >0.2: significant drift.
    """
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual, dtype=float)

    # Remove NaNs
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    def bucket_counts(data, breaks):
        counts = np.histogram(data, bins=breaks)[0]
        counts = counts / counts.sum()
        counts = np.where(counts == 0, 1e-6, counts)
        return counts

    exp_counts = bucket_counts(expected, breakpoints)
    act_counts = bucket_counts(actual,   breakpoints)

    psi = np.sum((act_counts - exp_counts) * np.log(act_counts / exp_counts + 1e-9))
    return round(float(psi), 4)


def ks_test(expected: np.ndarray, actual: np.ndarray) -> dict:
    """Two-sample KS test. Returns statistic, p-value, and drift flag."""
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]

    stat, pval = stats.ks_2samp(expected, actual)
    return {
        "statistic": round(float(stat), 4),
        "p_value":   round(float(pval), 4),
        "drifted":   bool(pval < KS_ALPHA),
    }


def mean_shift(expected: np.ndarray, actual: np.ndarray) -> dict:
    """Absolute and relative mean shift between distributions."""
    e = float(np.nanmean(expected))
    a = float(np.nanmean(actual))
    rel = abs(a - e) / (abs(e) + 1e-9)
    return {
        "ref_mean":    round(e, 4),
        "cur_mean":    round(a, 4),
        "abs_shift":   round(abs(a - e), 4),
        "rel_shift":   round(rel, 4),
    }


def psi_label(psi: float) -> str:
    if psi < PSI_LOW_THRESHOLD:
        return "Stable"
    elif psi < PSI_MEDIUM_THRESHOLD:
        return "Moderate"
    else:
        return "High Drift"


def psi_color(psi: float) -> str:
    if psi < PSI_LOW_THRESHOLD:
        return "green"
    elif psi < PSI_MEDIUM_THRESHOLD:
        return "orange"
    else:
        return "red"


def analyze_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare numeric columns between reference and current DataFrames.
    Returns a summary DataFrame with PSI, KS stats, and drift label per column.
    """
    num_cols = ref_df.select_dtypes(include="number").columns.intersection(
        cur_df.select_dtypes(include="number").columns
    ).tolist()

    rows = []
    for col in num_cols:
        psi  = compute_psi(ref_df[col].dropna(), cur_df[col].dropna())
        ks   = ks_test(ref_df[col].dropna(), cur_df[col].dropna())
        ms   = mean_shift(ref_df[col].dropna(), cur_df[col].dropna())
        rows.append({
            "Feature":       col,
            "PSI":           psi,
            "KS Statistic":  ks["statistic"],
            "KS p-value":    ks["p_value"],
            "KS Drifted":    ks["drifted"],
            "Ref Mean":      ms["ref_mean"],
            "Cur Mean":      ms["cur_mean"],
            "Abs Shift":     ms["abs_shift"],
            "Rel Shift %":   round(ms["rel_shift"] * 100, 2),
            "Status":        psi_label(psi),
        })

    return pd.DataFrame(rows).sort_values("PSI", ascending=False)


def simulate_drift_batch(
    ref_df: pd.DataFrame,
    drift_fraction: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a simulated 'new batch' by perturbing a fraction of numeric columns.
    Used to demonstrate drift detection in the monitoring dashboard.
    """
    rng      = np.random.default_rng(seed)
    cur_df   = ref_df.copy()
    num_cols = cur_df.select_dtypes(include="number").columns.tolist()

    n_drift = max(1, int(len(num_cols) * drift_fraction))
    drift_cols = rng.choice(num_cols, size=n_drift, replace=False)

    for col in drift_cols:
        std = cur_df[col].std()
        cur_df[col] = cur_df[col] + rng.normal(1.5 * std, 0.5 * std, len(cur_df))

    return cur_df


def generate_psi_history(features: list, days: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Generate a time-series of PSI values per feature for monitoring charts.
    Introduces a spike in the last week to simulate drift event.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")
    rows  = []
    for feat in features:
        base_psi = rng.uniform(0.02, 0.08)
        psi_vals = base_psi + rng.normal(0, 0.01, days)
        # inject drift event in last 7 days for one random feature
        if rng.random() < 0.4:
            psi_vals[-7:] += rng.uniform(0.12, 0.25)
        psi_vals = np.clip(psi_vals, 0, 1)
        for dt, val in zip(dates, psi_vals):
            rows.append({"date": dt, "feature": feat, "psi": round(float(val), 4)})
    return pd.DataFrame(rows)
