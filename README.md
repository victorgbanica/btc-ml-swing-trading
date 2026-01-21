# BTC/USD Swing Trading Backtester

This repository implements a gradient-boosted decision tree model for medium-horizon BTC/USD price prediction, complete with feature generation, walk-forward backtesting, and confidence-scaled position sizing. The repository is **public-safe**: proprietary feature engineering has been replaced with placeholder stubs that preserve interface and dimensionality.

---

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Backtesting and Results](#backtesting-and-results)
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



---

## Disclaimer

This code is for educational purposes only. It is not financial advice. Trading cryptocurrencies involves substantial risk, and past performance does not guarantee future results.
