# BTC/USD Swing Trading Backtester

This repository implements a gradient-boosted decision tree model for medium-horizon BTC/USD price prediction, complete with feature generation, walk-forward backtesting, and confidence-scaled position sizing. The repository is **public-safe**: proprietary feature engineering has been replaced with placeholder stubs that preserve interface and dimensionality.

---

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Backtesting and Results](#backtesting-and-results)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/victorgbanica/btc-ml-swing-trading.git
cd btc-ml-swing-trading
```

2. Create a Virtual Environment (Recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install Dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

This repository does not include proprietary historical data. To run the scripts, you need CSVs with the following structure:

  - XBTUSD_60.csv (BTC/USD 1-hour OHLCV)
    Columns: timestamp, open, high, low, close, volume, trades

  - SPY_Historical.csv (optional market benchmark)
    Columns: Date, Close/Last, Volume, Open, High, Low

  - fng.csv (Fear & Greed index)
    Columns: date, fng_value, fng_classification

Place these files in the data/ folder.

---

## Usage

Generate Features and Targets:

```python
from features import make_features
from model import make_target
from utils import load_and_prepare_data

# Load your data
df = load_and_prepare_data(
    xbtusd_path='data/XBTUSD_60.csv',
    spy_path='data/SPY_Historical.csv',
    fng_path='data/fng.csv',
    start_date='2021-01-11'
)

# Generate features
X = make_features(df)

# Create targets
y, forward_returns = make_target(df, horizon=24, method='binary')
```
Walk-Forward Backtesting:

```python
from backtest import (
    walk_forward_predict,
    generate_signals,
    calculate_position_sizes,
    backtest_strategy,
    plot_backtest_results
)

# Walk-forward model predictions
predictions, probabilities = walk_forward_predict(
    X, y,
    initial_train_size=6000,
    retrain_frequency=168,
    method='binary'
)

# Generate trading signals
signals = generate_signals(predictions, probabilities, method='binary', strategy='quantile', threshold_percentile=95)

# Calculate position sizes
positions = calculate_position_sizes(signals, probabilities, method='binary', sizing='confidence')

# Backtest
results = backtest_strategy(df, signals, positions)

# Plot results
plot_backtest_results(df, results)
```

---

## Project Structure

```
.
├── data/                     # CSV headers or sample files
    ├──── XBTUSD_60.csv
    ├──── SPY_Historical.csv
    ├──── fng.csv
├── src/
    ├──── features.py         # Feature engineering (public stubs)
    ├──── model.py            # Model training, evaluation
    ├──── backtest.py         # Walk-forward backtesting and plotting
    ├──── utils.py            # Helper functions
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Backtesting and Results

A Backtest from Jan 11, 2021 produced the following results:

**Signal Generation (Quantile Strategy)**:
  - Long threshold (P95): 0.6070
  - Short threshold (P5): 0.3770
  - Long signals:    1513 (4.2%)
  - Short signals:   1513 (4.2%)
  - Neutral signals: 33216 (91.7%)

**Position Sizing (Confidence-based)**:
   Average position size: 0.827

**Backtesting Strategy with Worst-Case Fee Tiers**:
  - Initial capital: $2,000
  - Maker/Taker ratio: 0% maker, 100% taker

**Backtest Results**:
  - Initial Capital: $2,000.00
  - Final Value: $76,928.25
  - Total Return: 3746.41%
  - Sharpe Ratio: 1.21
  - Max Drawdown: -74.38%
  - Number of Trades: 81
  - Win Rate: 43.2%
  - Avg Win: 15.35%
  - Avg Loss: -7.33%
  - Avg Hold Time: 135.6 hours (5.6 days)


<img width="1280" height="660" alt="05-01-2025-results" src="https://github.com/user-attachments/assets/bd0cd3c3-9e7a-40da-835f-2ee898c8dcd9" />

---

## Future Improvements

This project is actively evolving. Key improvements and ongoing work include:

- **Live Testing / Paper Trading**:  
  Live testing began on **January 5, 2026**. The model is currently being simulated in a paper trading environment, and this phase is planned to last **at least six months** to evaluate robustness.

- **Weekly Model Retraining**:  
  The strategy retrains the gradient-boosted model **once per week** using the latest available data, ensuring it adapts to recent market dynamics and maintains predictive relevance.

- **Drawdown Reduction**:  
  Implementations are underway to **minimize drawdowns**, including enhanced stop-loss rules, position sizing adjustments, and volatility-aware scaling.

- **Feature Engineering Expansion**:  
  While the public repository uses stub features for safety, private versions are exploring additional predictive indicators to improve signal quality and reduce false positives.

- **Automation and Alerts**:  
  Integration with automated monitoring, notifications, and logging is planned to track model performance, flag anomalies, and facilitate rapid iteration.

- **Comprehensive Evaluation Metrics**:  
  Additional performance measures such as drawdown duration, rolling Sharpe ratios, and tail-risk metrics will be tracked to better assess strategy stability under different market conditions.

These improvements aim to make the system **more resilient, adaptive, and informative** while maintaining reproducibility in research and backtesting.

---

## Disclaimer

This code is for educational purposes only. It is not financial advice. Trading cryptocurrencies involves substantial risk, and past performance does not guarantee future results.
