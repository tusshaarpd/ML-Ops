"""
Synthetic dataset generation for all 6 MLOps use cases.
Each generator uses a fixed seed for deterministic output and introduces
realistic imperfections: missing values, outliers, duplicates, class imbalance.
"""
import numpy as np
import pandas as pd
from config import RANDOM_SEED, USE_CASES


# ── Helpers ────────────────────────────────────────────────────────────────────

def _inject_missings(df: pd.DataFrame, rate: float = 0.03, rng=None) -> pd.DataFrame:
    """Randomly set `rate` fraction of cells to NaN (numeric cols only)."""
    rng = rng or np.random.default_rng(RANDOM_SEED)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df


def _inject_outliers(df: pd.DataFrame, cols: list, rate: float = 0.02, rng=None) -> pd.DataFrame:
    """Inject extreme outliers (5× std from mean) into numeric columns."""
    rng = rng or np.random.default_rng(RANDOM_SEED + 1)
    for col in cols:
        if col not in df.columns:
            continue
        idx = rng.choice(len(df), size=max(1, int(len(df) * rate)), replace=False)
        sign = rng.choice([-1, 1], size=len(idx))
        df.loc[idx, col] = df[col].mean() + sign * 5 * df[col].std()
    return df


def _inject_duplicates(df: pd.DataFrame, rate: float = 0.02, rng=None) -> pd.DataFrame:
    """Duplicate a fraction of rows."""
    rng = rng or np.random.default_rng(RANDOM_SEED + 2)
    n_dup = max(1, int(len(df) * rate))
    dup_idx = rng.choice(len(df), size=n_dup, replace=False)
    dup_rows = df.iloc[dup_idx].copy()
    return pd.concat([df, dup_rows], ignore_index=True)


# ── 1. Sentiment Analysis ──────────────────────────────────────────────────────

_POSITIVE_SEEDS = [
    "Product quality is amazing and exceeded expectations",
    "Fast delivery and great packaging",
    "Absolutely love this product, highly recommend",
    "Excellent customer service, very responsive",
    "Best purchase I have made this year",
    "Great value for the price, totally satisfied",
    "Super happy with my order, arrived on time",
    "Outstanding build quality, exactly as described",
    "Works perfectly, setup was very easy",
    "Five stars without hesitation, brilliant product",
    "Very durable and well designed, impressed",
    "Quick shipping and product looks premium",
    "Fantastic experience from start to finish",
    "The product is exactly what was advertised",
    "Delighted with the quality, will order again",
]

_NEUTRAL_SEEDS = [
    "Product is okay, nothing special",
    "Delivery was a bit late but package intact",
    "Average quality for the price point",
    "Does the job but nothing extraordinary",
    "Decent product, meets basic requirements",
    "Works as expected, nothing more",
    "Packaging was simple but product fine",
    "Not bad but not great either",
    "Acceptable quality, nothing to complain about",
    "Moderate experience, would consider alternatives",
    "Product matches description, no issues",
    "Standard delivery time, product works",
    "Fair enough for the cost",
    "Neutral experience overall",
    "Product arrived, seems fine so far",
]

_NEGATIVE_SEEDS = [
    "Very poor customer support, no response",
    "I want a refund immediately, terrible product",
    "Broke after two days of use, very disappointed",
    "Completely different from what was described",
    "Worst purchase ever, waste of money",
    "Extremely slow delivery, took three weeks",
    "Product stopped working within a week",
    "Terrible quality, feels cheap and flimsy",
    "No response from seller after complaint",
    "Missing parts, completely unusable out of box",
    "Arrived damaged, packaging was very poor",
    "False advertising, product is nothing like photos",
    "Do not buy this, complete scam",
    "Battery dies in an hour, misleading claims",
    "Returned it immediately, awful experience",
]


def generate_sentiment_data(n: int = 1200, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice(["Positive", "Neutral", "Negative"],
                        size=n, p=[0.50, 0.25, 0.25])

    texts = []
    for lbl in labels:
        if lbl == "Positive":
            base = rng.choice(_POSITIVE_SEEDS)
        elif lbl == "Neutral":
            base = rng.choice(_NEUTRAL_SEEDS)
        else:
            base = rng.choice(_NEGATIVE_SEEDS)
        # small surface variations
        variants = ["", " really", " very", " quite", " somewhat", " honestly"]
        words = base.split()
        if len(words) > 3:
            insert_pos = rng.integers(1, len(words))
            variant = rng.choice(variants)
            if variant:
                words.insert(int(insert_pos), variant.strip())
        texts.append(" ".join(words))

    df = pd.DataFrame({"text": texts, "sentiment": labels})
    # id column
    df.insert(0, "review_id", [f"REV{i:05d}" for i in range(len(df))])
    return df


# ── 2. Spam Detection ─────────────────────────────────────────────────────────

_SPAM_MSGS = [
    "Claim your free reward now, call immediately",
    "Urgent update your bank details to avoid suspension",
    "You have won a lottery prize of 10000 dollars",
    "FREE entry in weekly competition to win FA Cup final tickets",
    "WINNER: You have been selected for cash prize",
    "Send your personal details to claim your gift",
    "Your mobile number has been awarded 500 pounds",
    "Congratulations you are chosen for exclusive offer",
    "Click here to get free iPhone today only limited",
    "Double your income working from home guaranteed",
    "Meet singles in your area tonight, join free now",
    "Buy meds online no prescription required cheapest",
    "URGENT account suspended verify now or lose access",
    "Investment opportunity returns 300 percent monthly",
    "Earn 5000 per week from home easy money method",
]

_HAM_MSGS = [
    "Team meeting at 4 PM in the conference room",
    "Invoice attached please review and approve",
    "Reminder your appointment is tomorrow at 10 AM",
    "Can you pick up milk on your way home",
    "Happy birthday hope you have a great day",
    "The report is ready, please find it attached",
    "Lunch at noon sounds good, see you there",
    "Please call me when you get a chance",
    "Your package has been dispatched, tracking below",
    "Hi just checking in, how are you doing",
    "Project deadline extended to next Friday",
    "Could you review the proposal before Thursday",
    "Great work on the presentation today well done",
    "The server maintenance window is Saturday night",
    "Please join the standup at nine tomorrow morning",
]


def generate_spam_data(n: int = 1500, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice(["Ham", "Spam"], size=n, p=[0.75, 0.25])

    messages = []
    for lbl in labels:
        if lbl == "Spam":
            msg = rng.choice(_SPAM_MSGS)
            # vary with extra tokens
            extras = ["", "Reply STOP to unsubscribe", "Limited time", "Act now"]
            msg = msg + ". " + rng.choice(extras) if rng.random() < 0.4 else msg
        else:
            msg = rng.choice(_HAM_MSGS)
        messages.append(msg)

    df = pd.DataFrame({"message": messages, "label": labels})
    df.insert(0, "msg_id", [f"MSG{i:05d}" for i in range(len(df))])
    return df


# ── 3. Customer Churn ─────────────────────────────────────────────────────────

def generate_churn_data(n: int = 2000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age          = rng.integers(18, 75, size=n)
    region       = rng.choice(["North", "South", "East", "West", "Central"], size=n)
    plan         = rng.choice(["Basic", "Standard", "Premium", "Enterprise"], size=n,
                               p=[0.30, 0.35, 0.25, 0.10])
    tenure       = rng.integers(1, 120, size=n)          # months
    monthly_bill = rng.uniform(20, 200, size=n)
    complaints   = rng.integers(0, 10, size=n)
    usage_drop   = rng.uniform(0, 1, size=n)             # 0 = no drop, 1 = full drop
    pay_delays   = rng.integers(0, 12, size=n)

    # Churn probability based on features
    churn_score = (
        0.3  * (complaints / 10)
        + 0.25 * usage_drop
        + 0.2  * (pay_delays / 12)
        + 0.15 * (1 - tenure / 120)
        + 0.1  * (monthly_bill / 200)
    )
    churn_score = np.clip(churn_score + rng.normal(0, 0.08, n), 0, 1)
    churn = (churn_score > 0.5).astype(int)

    df = pd.DataFrame({
        "customer_id":    [f"CUST{i:05d}" for i in range(n)],
        "age":            age,
        "region":         region,
        "plan_type":      plan,
        "tenure_months":  tenure,
        "monthly_bill":   monthly_bill.round(2),
        "num_complaints": complaints,
        "usage_drop_pct": (usage_drop * 100).round(1),
        "payment_delays": pay_delays,
        "churn":          churn,
    })

    df = _inject_missings(df, rate=0.02, rng=rng)
    df = _inject_outliers(df, cols=["monthly_bill", "num_complaints"], rng=rng)
    return df


# ── 4. Fraud Detection ────────────────────────────────────────────────────────

def generate_fraud_data(n: int = 3000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    amount           = rng.exponential(scale=200, size=n).clip(1, 5000).round(2)
    merchant_type    = rng.choice(["Retail", "Online", "Travel", "Restaurant", "ATM"],
                                   size=n, p=[0.30, 0.35, 0.10, 0.15, 0.10])
    device_mismatch  = rng.binomial(1, 0.12, n)
    geo_mismatch     = rng.binomial(1, 0.08, n)
    velocity_count   = rng.integers(1, 20, size=n)
    prev_chargeback  = rng.binomial(1, 0.05, n)
    night_txn        = rng.binomial(1, 0.20, n)
    card_age_months  = rng.integers(1, 120, size=n)
    is_international = rng.binomial(1, 0.15, n)

    # Fraud probability
    fraud_score = (
        0.30 * device_mismatch
        + 0.25 * geo_mismatch
        + 0.20 * prev_chargeback
        + 0.10 * (velocity_count / 20)
        + 0.08 * night_txn
        + 0.07 * is_international
        + rng.normal(0, 0.05, n)
    )
    fraud_score = np.clip(fraud_score, 0, 1)
    is_fraud = (fraud_score > 0.55).astype(int)  # ~5-8% fraud rate

    df = pd.DataFrame({
        "txn_id":           [f"TXN{i:06d}" for i in range(n)],
        "amount":           amount,
        "merchant_type":    merchant_type,
        "device_mismatch":  device_mismatch,
        "geo_mismatch":     geo_mismatch,
        "velocity_count":   velocity_count,
        "prev_chargeback":  prev_chargeback,
        "night_txn":        night_txn,
        "card_age_months":  card_age_months,
        "is_international": is_international,
        "fraud_score":      fraud_score.round(4),
        "is_fraud":         is_fraud,
    })

    df = _inject_missings(df, rate=0.01, rng=rng)
    df = _inject_outliers(df, cols=["amount", "velocity_count"], rng=rng)
    return df


# ── 5. Sales Forecasting ──────────────────────────────────────────────────────

def generate_sales_data(n: int = 730, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate ~2 years of daily sales with trend, seasonality, and noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    day_of_year  = np.arange(n)
    trend        = 500 + 0.15 * day_of_year
    seasonality  = 100 * np.sin(2 * np.pi * day_of_year / 365.25)
    weekly_cycle = 80  * np.sin(2 * np.pi * day_of_year / 7 + 1)
    noise        = rng.normal(0, 40, n)
    base_sales   = (trend + seasonality + weekly_cycle + noise).clip(0)

    # External factors
    promotion    = rng.binomial(1, 0.15, n)
    holiday_flag = np.isin(dates.day_of_year,
                           [1, 50, 100, 150, 200, 250, 300, 350]).astype(int)
    price_change = rng.choice([-1, 0, 0, 0, 1], size=n)

    sales = base_sales + 150 * promotion + 200 * holiday_flag - 50 * (price_change == 1)
    sales = sales.clip(0).round(0)

    df = pd.DataFrame({
        "date":          dates,
        "sales":         sales.astype(int),
        "promotion":     promotion,
        "holiday_flag":  holiday_flag,
        "price_change":  price_change,
        "day_of_week":   dates.day_of_week,
        "month":         dates.month,
        "quarter":       dates.quarter,
        "year":          dates.year,
        "day_of_year":   dates.day_of_year,
    })
    return df


# ── 6. HR Attrition ───────────────────────────────────────────────────────────

def generate_hr_data(n: int = 1500, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    departments      = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
    department       = rng.choice(departments, size=n,
                                   p=[0.30, 0.20, 0.15, 0.10, 0.12, 0.13])
    salary_band      = rng.choice(["Low", "Mid", "High", "Executive"],
                                   size=n, p=[0.25, 0.40, 0.28, 0.07])
    overtime         = rng.binomial(1, 0.28, n)
    satisfaction     = rng.uniform(1, 5, n).round(1)
    tenure_years     = rng.integers(0, 20, size=n)
    manager_changes  = rng.integers(0, 5, size=n)
    promotions       = rng.integers(0, 4, size=n)
    distance_km      = rng.integers(1, 60, size=n)
    training_hours   = rng.integers(0, 80, size=n)

    # salary encoding for scoring
    sal_map = {"Low": 0.9, "Mid": 0.4, "High": 0.15, "Executive": 0.05}
    sal_risk = np.array([sal_map[s] for s in salary_band])

    attrition_score = (
        0.30 * sal_risk
        + 0.25 * overtime
        + 0.20 * (1 - satisfaction / 5)
        + 0.10 * (manager_changes / 5)
        + 0.08 * (1 - tenure_years / 20)
        + 0.07 * (distance_km / 60)
        + rng.normal(0, 0.08, n)
    )
    attrition_score = np.clip(attrition_score, 0, 1)
    attrition = (attrition_score > 0.45).astype(int)

    df = pd.DataFrame({
        "emp_id":          [f"EMP{i:05d}" for i in range(n)],
        "department":      department,
        "salary_band":     salary_band,
        "overtime":        overtime,
        "satisfaction":    satisfaction,
        "tenure_years":    tenure_years,
        "manager_changes": manager_changes,
        "promotions":      promotions,
        "distance_km":     distance_km,
        "training_hours":  training_hours,
        "attrition":       attrition,
    })

    df = _inject_missings(df, rate=0.02, rng=rng)
    df = _inject_outliers(df, cols=["distance_km", "training_hours"], rng=rng)
    df = _inject_duplicates(df, rate=0.01, rng=rng)
    return df


# ── Registry ──────────────────────────────────────────────────────────────────

GENERATORS = {
    "sentiment": generate_sentiment_data,
    "spam":      generate_spam_data,
    "churn":     generate_churn_data,
    "fraud":     generate_fraud_data,
    "sales":     generate_sales_data,
    "hr":        generate_hr_data,
}


def get_dataset(use_case: str, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return the dataset for the given use case key."""
    gen = GENERATORS.get(use_case)
    if gen is None:
        raise ValueError(f"Unknown use case: {use_case}")
    n = USE_CASES[use_case]["n_samples"]
    return gen(n=n, seed=seed)


def data_quality_report(df: pd.DataFrame) -> dict:
    """Compute a data quality summary for any dataframe."""
    n = len(df)
    report = {
        "total_rows":    n,
        "total_cols":    df.shape[1],
        "missing_count": int(df.isnull().sum().sum()),
        "missing_pct":   round(df.isnull().sum().sum() / (n * df.shape[1]) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_pct":  round(df.duplicated().sum() / n * 100, 2),
        "col_missing":    df.isnull().sum().to_dict(),
        "dtypes":         df.dtypes.astype(str).to_dict(),
        "num_summary":    df.describe().round(3).to_dict() if not df.empty else {},
    }
    # Range violations: values > 3 std from mean (numeric only)
    num_df = df.select_dtypes(include="number")
    z_scores = ((num_df - num_df.mean()) / (num_df.std() + 1e-9)).abs()
    report["outlier_count"] = int((z_scores > 3).sum().sum())
    return report
