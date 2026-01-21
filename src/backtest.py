import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from model import train_model


# WALK-FORWARD MODEL EVALUATION

def walk_forward_predict(X, y, initial_train_size=10000, retrain_frequency=168, method="binary"):
    predictions = np.full(len(X), np.nan)
    probabilities = np.full((len(X), 2), np.nan)

    start_idx = initial_train_size
    total_retrains = int(np.ceil((len(X) - start_idx) / retrain_frequency))

    print("\n Walk-forward evaluation")
    start_time = time.time()

    for _ in tqdm(range(total_retrains), desc="Retraining"):
        end_idx = min(start_idx + retrain_frequency, len(X))

        X_train = X.iloc[:start_idx]
        y_train = y.iloc[:start_idx]
        X_test = X.iloc[start_idx:end_idx]

        model, scaler = train_model(
            X_train, y_train,
            X_train.iloc[-len(X_train)//5:], y_train.iloc[-len(y_train)//5:],
            method=method,
            verbose=False
        )

        X_test_scaled = scaler.transform(X_test)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)

        predictions[start_idx:end_idx] = preds
        probabilities[start_idx:end_idx] = probs

        start_idx = end_idx

    elapsed = (time.time() - start_time) / 60
    print(f" Walk-forward complete in {elapsed:.1f} minutes")

    return predictions, probabilities


# SIGNAL GENERATION (ABSTRACTED)

def generate_signals(predictions, probabilities):
    """
    Convert model outputs into abstract trading signals.

    This public version uses confidence ranking only and omits
    proprietary thresholding logic.
    """
    signals = np.zeros(len(predictions))

    confidence = probabilities[:, 1]
    valid = ~np.isnan(confidence)

    high = np.percentile(confidence[valid], 90)
    low = np.percentile(confidence[valid], 10)

    signals[confidence > high] = 1
    signals[confidence < low] = -1

    return signals


# POSITION SIZING (GENERIC)

def calculate_position_sizes(signals, probabilities):
    """
    Confidence-scaled position sizing.
    """
    confidence = np.nan_to_num(probabilities[:, 1], nan=0.0)
    size = 0.5 + 0.5 * confidence
    return signals * size


# BACKTEST ENGINE (GENERIC)

def backtest_strategy(df, positions, initial_capital=1.0):
    returns = np.log(df["close_close"] / df["close_close"].shift(1)).fillna(0)
    strategy_returns = positions * returns

    equity_curve = (1 + strategy_returns).cumprod()
    equity_curve.iloc[0] = initial_capital

    sharpe = (
        strategy_returns.mean() / strategy_returns.std() * np.sqrt(365 * 24)
        if strategy_returns.std() > 0 else 0
    )

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max

    results = {
        "equity_curve": equity_curve,
        "total_return": (equity_curve.iloc[-1] - 1) * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown.min() * 100
    }

    print("\n Backtest Results")
    print(f"   Total Return: {results['total_return']:.2f}%")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")

    return results


# PLOTTING

def plot_results(df, results):
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, results["equity_curve"], label="Strategy")
    plt.axhline(1, linestyle="--", color="gray")
    plt.title("Equity Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# EXAMPLE RUN

if __name__ == "__main__":
    from utils import load_and_prepare_data
    from features import make_features
    from model import make_target

    df = load_and_prepare_data()
    X = make_features(df)
    y, _ = make_target(df)

    valid = ~y.isna()
    X, y, df = X[valid], y[valid], df.loc[X.index]

    preds, probs = walk_forward_predict(X, y)
    signals = generate_signals(preds, probs)
    positions = calculate_position_sizes(signals, probs)

    results = backtest_strategy(df, positions)
    plot_results(df, results)
