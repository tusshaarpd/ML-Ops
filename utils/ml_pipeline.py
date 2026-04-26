"""
ML Pipeline: preprocessing, training, evaluation, and persistence.
Handles both classification and regression use cases.
"""
import time
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
    IsolationForest,
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

from config import MODELS_DIR, RANDOM_SEED, DEFAULT_TEST_SIZE, DEFAULT_CV_FOLDS

# ── XGBoost optional ──────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── imbalanced-learn optional ─────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# ── Text use-case feature extraction ─────────────────────────────────────────

def build_text_pipeline(clf, max_features: int = 5000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", clf),
    ])


# ── Tabular preprocessor ──────────────────────────────────────────────────────

def build_tabular_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    transformers = []
    if num_cols:
        transformers.append(("num", num_transformer, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_transformer, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_tabular_pipeline(clf, num_cols: list, cat_cols: list) -> Pipeline:
    preprocessor = build_tabular_preprocessor(num_cols, cat_cols)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf),
    ])


# ── Model factory ─────────────────────────────────────────────────────────────

def get_classifier(algo_key: str, params: dict = None, seed: int = RANDOM_SEED):
    params = params or {}
    mapping = {
        "lr":  LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            random_state=seed,
        ),
        "rf":  RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            random_state=seed,
            n_jobs=-1,
        ),
        "gb":  GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=seed,
        ),
        "nb":  MultinomialNB(alpha=params.get("alpha", 1.0)),
        "if":  IsolationForest(
            n_estimators=params.get("n_estimators", 100),
            contamination=params.get("contamination", 0.05),
            random_state=seed,
        ),
    }
    if XGBOOST_AVAILABLE:
        mapping["xgb"] = XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
        )
    return mapping.get(algo_key)


def get_regressor(algo_key: str, params: dict = None, seed: int = RANDOM_SEED):
    params = params or {}
    mapping = {
        "linreg": LinearRegression(),
        "rfr":    RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            random_state=seed,
            n_jobs=-1,
        ),
        "gbr":    GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=seed,
        ),
    }
    if XGBOOST_AVAILABLE:
        mapping["xgbr"] = XGBRegressor(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=seed,
            verbosity=0,
        )
    return mapping.get(algo_key)


# ── Feature extraction helpers ────────────────────────────────────────────────

def _schema_for_use_case(use_case: str, df: pd.DataFrame):
    """Return (text_col, feature_cols, num_cols, cat_cols) based on use case."""
    id_cols   = [c for c in df.columns if "_id" in c.lower() or c.lower() == "id"]
    target    = _target_col(use_case)
    drop_cols = id_cols + [target, "date"]

    if use_case in ("sentiment", "spam"):
        text_col = "text" if "text" in df.columns else "message"
        return text_col, None, None, None

    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    num_cols   = feature_df.select_dtypes(include="number").columns.tolist()
    cat_cols   = feature_df.select_dtypes(include="object").columns.tolist()
    return None, feature_df.columns.tolist(), num_cols, cat_cols


def _target_col(use_case: str) -> str:
    from config import USE_CASES
    return USE_CASES[use_case]["target"]


# ── Main training function ─────────────────────────────────────────────────────

def train_model(
    use_case: str,
    df: pd.DataFrame,
    algo_key: str,
    params: dict = None,
    test_size: float = DEFAULT_TEST_SIZE,
    cv_folds: int = DEFAULT_CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Train a model for the given use case and return a results dict containing
    metrics, confusion matrix, feature names, importances, and the fitted pipeline.
    """
    t0 = time.time()
    params = params or {}
    target = _target_col(use_case)

    from config import USE_CASES
    task = USE_CASES[use_case]["task"]

    text_col, feature_cols, num_cols, cat_cols = _schema_for_use_case(use_case, df)

    # ── Prepare X, y ─────────────────────────────────────────────────────────
    df_clean = df.copy()

    if task == "classification":
        le = LabelEncoder()
        df_clean[target] = le.fit_transform(df_clean[target].astype(str))
    else:
        le = None

    if use_case in ("sentiment", "spam"):
        X = df_clean[text_col].fillna("").astype(str)
        y = df_clean[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y if task == "classification" else None
        )
        clf = get_classifier(algo_key, params, seed)
        if clf is None:
            raise ValueError(f"Algorithm {algo_key} not available")
        # NB requires non-negative, so we use LR as fallback for text
        if algo_key == "nb":
            pipeline = build_text_pipeline(
                MultinomialNB(alpha=params.get("alpha", 1.0)), max_features=5000
            )
        else:
            pipeline = build_text_pipeline(clf, max_features=5000)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        feature_names = pipeline.named_steps["tfidf"].get_feature_names_out().tolist()
        if hasattr(pipeline.named_steps["clf"], "coef_"):
            importances = np.abs(pipeline.named_steps["clf"].coef_[0])
        elif hasattr(pipeline.named_steps["clf"], "feature_importances_"):
            importances = pipeline.named_steps["clf"].feature_importances_
        else:
            importances = np.zeros(len(feature_names))
        # top 20 features
        top_idx = np.argsort(importances)[::-1][:20]
        feat_imp = dict(zip(
            np.array(feature_names)[top_idx],
            importances[top_idx].tolist()
        ))
    else:
        id_cols = [c for c in df_clean.columns if "_id" in c.lower() or c.lower() == "id"]
        drop = id_cols + [target, "date", "fraud_score"] if use_case == "fraud" else id_cols + [target, "date"]
        X = df_clean.drop(columns=[c for c in drop if c in df_clean.columns])
        y = df_clean[target]

        # impute missings
        num_X = X.select_dtypes(include="number")
        cat_X = X.select_dtypes(include="object")
        X[num_X.columns] = num_X.fillna(num_X.median())
        X[cat_X.columns] = cat_X.fillna("Unknown")

        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(include="object").columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed,
            stratify=y if task == "classification" else None
        )

        if task == "classification":
            clf = get_classifier(algo_key, params, seed)
            if clf is None:
                raise ValueError(f"Algorithm {algo_key} not available")
        else:
            clf = get_regressor(algo_key, params, seed)
            if clf is None:
                raise ValueError(f"Algorithm {algo_key} not available")

        pipeline = build_tabular_pipeline(clf, num_cols, cat_cols)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # feature importances
        prep = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["clf"]
        try:
            ohe_names = []
            if cat_cols:
                ohe = prep.named_transformers_.get("cat")
                if ohe is not None:
                    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
            all_feat_names = num_cols + ohe_names
            if hasattr(estimator, "feature_importances_"):
                imps = estimator.feature_importances_
            elif hasattr(estimator, "coef_"):
                imps = np.abs(estimator.coef_[0] if estimator.coef_.ndim > 1 else estimator.coef_)
            else:
                imps = np.ones(len(all_feat_names)) / len(all_feat_names)
            # truncate / pad
            min_len = min(len(imps), len(all_feat_names))
            feat_imp = dict(zip(all_feat_names[:min_len], imps[:min_len].tolist()))
            # sort
            feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:20])
        except Exception:
            feat_imp = {}

    # ── Compute metrics ───────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 2)

    if task == "classification":
        acc  = round(accuracy_score(y_test, y_pred), 4)
        prec = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)
        rec  = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4)
        f1   = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
        cm   = confusion_matrix(y_test, y_pred).tolist()
        try:
            if len(np.unique(y_test)) == 2:
                proba = pipeline.predict_proba(X_test)[:, 1]
            else:
                proba = pipeline.predict_proba(X_test)
            roc  = round(roc_auc_score(y_test, proba,
                                        multi_class="ovr", average="weighted"), 4)
        except Exception:
            roc = None

        # cross-val
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train,
                                         cv=StratifiedKFold(cv_folds, shuffle=True,
                                                            random_state=seed),
                                         scoring="f1_weighted", n_jobs=-1)
            cv_mean  = round(float(cv_scores.mean()), 4)
            cv_std   = round(float(cv_scores.std()), 4)
        except Exception:
            cv_mean, cv_std = f1, 0.0

        metrics = {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "roc_auc": roc,
            "cv_f1_mean": cv_mean, "cv_f1_std": cv_std,
        }
        classes = le.classes_.tolist() if le else list(range(len(cm)))

    else:  # regression
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
        mae  = round(mean_absolute_error(y_test, y_pred), 3)
        r2   = round(r2_score(y_test, y_pred), 4)
        mape = round(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100, 2)
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train,
                                         cv=KFold(cv_folds, shuffle=True, random_state=seed),
                                         scoring="neg_root_mean_squared_error", n_jobs=-1)
            cv_mean  = round(float(-cv_scores.mean()), 3)
            cv_std   = round(float(cv_scores.std()), 3)
        except Exception:
            cv_mean, cv_std = rmse, 0.0

        metrics = {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
                   "cv_rmse_mean": cv_mean, "cv_rmse_std": cv_std}
        cm      = None
        classes = []

    result = {
        "use_case":       use_case,
        "algo_key":       algo_key,
        "task":           task,
        "metrics":        metrics,
        "confusion_matrix": cm,
        "classes":        classes,
        "feature_importance": feat_imp,
        "pipeline":       pipeline,
        "label_encoder":  le,
        "n_train":        len(X_train),
        "n_test":         len(X_test),
        "training_time_s": elapsed,
        "test_size":      test_size,
        "cv_folds":       cv_folds,
        "params":         params,
        "y_test":         y_test,
        "y_pred":         y_pred,
    }
    logger.info("Trained %s / %s  acc=%s  time=%ss", use_case, algo_key,
                metrics.get("accuracy") or metrics.get("r2"), elapsed)
    return result


# ── Predict helpers ───────────────────────────────────────────────────────────

def predict_single(result: dict, input_data) -> dict:
    """
    Run inference on a single sample.
    input_data: string for text models, dict for tabular models.
    Returns prediction label and probabilities.
    """
    pipeline = result["pipeline"]
    le       = result["label_encoder"]
    task     = result["task"]

    if isinstance(input_data, str):
        X = pd.Series([input_data])
    elif isinstance(input_data, dict):
        X = pd.DataFrame([input_data])
    else:
        X = input_data

    pred = pipeline.predict(X)
    label = le.inverse_transform(pred)[0] if le else pred[0]

    proba = {}
    try:
        proba_arr = pipeline.predict_proba(X)[0]
        if le is not None:
            proba = {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba_arr)}
        else:
            proba = {str(i): round(float(p), 4) for i, p in enumerate(proba_arr)}
    except Exception:
        pass

    return {"prediction": label, "probabilities": proba}


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(result: dict, version: str) -> Path:
    fname = MODELS_DIR / f"{result['use_case']}_{result['algo_key']}_v{version}.pkl"
    joblib.dump({
        "pipeline": result["pipeline"],
        "label_encoder": result["label_encoder"],
        "metrics": result["metrics"],
        "classes": result["classes"],
        "task": result["task"],
        "algo_key": result["algo_key"],
        "use_case": result["use_case"],
        "feature_importance": result["feature_importance"],
    }, fname)
    logger.info("Model saved: %s", fname)
    return fname


def load_model(path: Path) -> dict:
    data = joblib.load(path)
    return data


def get_saved_models() -> list:
    return list(MODELS_DIR.glob("*.pkl"))
