"""model.py — baseline vs intermediate models, honest-fit design.

HONEST-FIT DECISION (per the jd-to-ml-project skill, Step 2): the LifeSci
Finance work is forecasting / valuation, not deep learning. This project
therefore uses interpretable regression + classification (Ridge / Logistic
as baselines; RandomForest / GradientBoosting as intermediates) and a
Monte Carlo budget engine — the tools a finance team would actually use.
No transformer / LLM is forced onto this problem.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_regression(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    models = {
        "Linear Regression (baseline)": LinearRegression(),
        "Ridge (baseline)": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
    }
    results, fitted = {}, {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2")
        results[name] = {
            "r2_test": round(float(r2_score(y_te, pred)), 4),
            "rmse_test_usd_m": round(float(np.sqrt(mean_squared_error(y_te, pred))), 2),
            "mae_test_usd_m": round(float(mean_absolute_error(y_te, pred)), 2),
            "r2_cv_mean": round(float(cv.mean()), 4),
            "r2_cv_std": round(float(cv.std()), 4),
        }
        fitted[name] = model
    best_name = max(results, key=lambda k: results[k]["r2_test"])
    return results, fitted, best_name, (X_tr, X_te, y_tr, y_te)


def train_classification(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    models = {
        "Logistic Regression (baseline)": Pipeline([
            ("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
    }
    results, fitted = {}, {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="roc_auc")
        results[name] = {
            "roc_auc_test": round(float(roc_auc_score(y_te, proba)), 4),
            "accuracy_test": round(float(accuracy_score(y_te, pred)), 4),
            "precision_test": round(float(precision_score(y_te, pred, zero_division=0)), 4),
            "recall_test": round(float(recall_score(y_te, pred, zero_division=0)), 4),
            "f1_test": round(float(f1_score(y_te, pred, zero_division=0)), 4),
            "roc_auc_cv_mean": round(float(cv.mean()), 4),
        }
        fitted[name] = model
    best_name = max(results, key=lambda k: results[k]["roc_auc_test"])
    return results, fitted, best_name, (X_tr, X_te, y_tr, y_te)
