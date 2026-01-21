import xgboost as xgb
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def make_target(df, horizon=24, method="binary"):
    """
    Create classification targets for medium-horizon price movement.

    Args:
        df: DataFrame containing a close price column
        horizon: Prediction horizon in periods
        method: 'binary' or 'multiclass'

    Returns:
        y: Target labels
        forward_returns: Forward log returns (for analysis only)
    """
    forward_returns = np.log(
        df["close_close"].shift(-horizon) / df["close_close"]
    )

    if method == "binary":
        threshold = forward_returns.median()
        y = (forward_returns > threshold).astype(int)

        print("Target construction (binary)")
        print(f"Horizon: {horizon}")
        print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

    elif method == "multiclass":
        lower = forward_returns.quantile(0.25)
        upper = forward_returns.quantile(0.75)

        y = pd.Series(1, index=forward_returns.index)
        y[forward_returns < lower] = 0
        y[forward_returns > upper] = 2

        print("Target construction (multiclass)")
        print(f"Horizon: {horizon}")
        print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

    else:
        raise ValueError("method must be 'binary' or 'multiclass'")

    return y, forward_returns


def train_model(X_train, y_train, X_val, y_val, method="binary", verbose=True):
    """
    Train a gradient-boosted decision tree classifier.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        method: 'binary' or 'multiclass'
        verbose: Control training output

    Returns:
        model: Trained XGBoost model
        scaler: Fitted feature scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    common_params = {
        "max_depth": 6,
        "min_child_weight": 5,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "n_estimators": 1000,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    if method == "binary":
        params = {
            **common_params,
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
        }

    elif method == "multiclass":
        params = {
            **common_params,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": ["mlogloss", "merror"],
        }

    else:
        raise ValueError("method must be 'binary' or 'multiclass'")

    model = xgb.XGBClassifier(**params)

    print("Training gradient-boosted model")
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Feature count: {X_train.shape[1]}")

    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=50 if verbose else False,
    )

    print("Training complete")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best validation score: {model.best_score:.4f}")

    return model, scaler


def evaluate_model(model, scaler, X_test, y_test, method="binary"):
    """
    Evaluate trained model on a held-out test set.
    """
    X_test_scaled = scaler.transform(X_test)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
    }

    if method == "binary":
        metrics["auc_roc"] = roc_auc_score(y_test, probabilities[:, 1])

    print("Test set performance")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nClassification report")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion matrix")
    print(confusion_matrix(y_test, predictions))

    return metrics, predictions, probabilities


def get_feature_importance(model, feature_names, top_n=20):
    """
    Return ranked feature importance from the trained model.
    """
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"Top {top_n} features by importance")
    print(importance.head(top_n).to_string(index=False))

    return importance


def cross_validate_model(X, y, method="binary", n_splits=5):
    """
    Time-series cross-validation using expanding windows.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    print(f"Time-series cross-validation ({n_splits} folds)")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model, scaler = train_model(
            X_train, y_train, X_val, y_val,
            method=method,
            verbose=False,
        )

        X_val_scaled = scaler.transform(X_val)
        preds = model.predict(X_val_scaled)
        acc = accuracy_score(y_val, preds)

        scores.append(acc)
        print(f"Fold {fold}: accuracy = {acc:.4f}")

    print(f"Mean accuracy: {np.mean(scores):.4f}")
    print(f"Std deviation: {np.std(scores):.4f}")

    return scores
