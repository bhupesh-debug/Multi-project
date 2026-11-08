"""Model training and evaluation.

Primary task (binary): predict ``target_up`` — will this asset be higher
in 20 trading days? Models are compared on a strict chronological split
(train 2019–2023, test 2024–2025) so no future information leaks into
training. A 3-class regime classifier (bull/range/bear) is trained as a
secondary model with the same features.

Models:
* ``rule`` — baseline momentum rule (no learning): long if 60-day return
  is positive and price is above its 60-day average, else short.
* ``logreg`` — logistic regression on standardised features (interpretable).
* ``rf`` — random forest (feature importances, non-linear).
* ``hgb`` — histogram gradient boosting (strong tabular default).
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import FEATURE_COLUMNS

RANDOM_STATE = 42


def rule_signal(frame: pd.DataFrame) -> np.ndarray:
    """Momentum rule baseline: +1 long, -1 short. No fitting involved."""
    long = (frame["ret_60d"] > 0) & (frame["ma_ratio"] > 0)
    return np.where(long, 1, -1)


def train_binary_models(
    x_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, object]:
    """Train the binary up/down models. Returns {name: fitted model}."""
    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
    return models


def train_regime_model(x_train: pd.DataFrame, y_train: pd.Series):
    """3-class regime classifier (bull/range/bear)."""
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
    )
    model.fit(x_train, y_train)
    return model


def binary_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def evaluate_binary_models(models, x_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        proba = model.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({"model": name, **binary_metrics(y_test, pred, proba)})
    # Rule baseline has no probabilities: AUC is not defined, accuracy only.
    return pd.DataFrame(rows)


def rule_metrics(frame_test: pd.DataFrame) -> dict[str, float]:
    pred_up = (rule_signal(frame_test) == 1).astype(int)
    y = frame_test["target_up"]
    return {
        "model": "rule",
        "accuracy": round(accuracy_score(y, pred_up), 4),
        "roc_auc": float("nan"),
        "precision": round(precision_score(y, pred_up, zero_division=0), 4),
        "recall": round(recall_score(y, pred_up, zero_division=0), 4),
        "f1": round(f1_score(y, pred_up, zero_division=0), 4),
    }


def feature_importances(rf_model, feature_names=FEATURE_COLUMNS) -> pd.DataFrame:
    """Random-forest importances, sorted. Only valid for the RF model."""
    importances = rf_model.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_model(model, path: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump(model, fh)


def load_model(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    from data_pipeline import generate_synthetic_prices
    from feature_engineering import build_model_frame, time_split

    frame = build_model_frame(generate_synthetic_prices())
    train, test = time_split(frame)
    models = train_binary_models(train[FEATURE_COLUMNS], train["target_up"])
    results = evaluate_binary_models(models, test[FEATURE_COLUMNS], test["target_up"])
    print(results.to_string(index=False))
    print("\nConfusion matrix (best by F1):")
    best = results.sort_values("f1", ascending=False).iloc[0]["model"]
    proba = models[best].predict_proba(test[FEATURE_COLUMNS])[:, 1]
    print(confusion_matrix(test["target_up"], (proba >= 0.5).astype(int)))
