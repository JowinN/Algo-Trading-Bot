# Mudrex Algo Trading Bot (V10 + V4 ML Filter)

A premium, machine-learning-enhanced algorithmic trading system developed for the Mudrex platform. This repository implements a robust momentum continuation strategy with a direction-specific ensemble model (V4) that dynamically filters signals and optimizes stop-loss/take-profit parameters using advanced excursion analysis.

---

## 🚀 Key Features

*   **V10 Core Strategy:** Built around a 4-hour momentum continuation pattern using multi-timeframe EMA trends, volatility expansion (Squeeze), ADX trend strength, and RSI pullbacks.
*   **V4 ML Filter & Optimizer:** 
    *   **Direction-Specific Ensembles:** Separate calibrated XGBoost models for LONG and SHORT trades to capture asymmetrical market dynamics.
    *   **Calibrated Probability Thresholds:** Probability thresholding optimized for expectancy and trade frequency (`Utility = Expectancy * (Pass_Rate ** 0.5)`).
    *   **Dynamic Excursion SL/TP:** Regressors predicting optimal stop-loss (ATR-multiple) and take-profit (R-multiple) based on historical Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE).
*   **Dual Process Architecture:**
    *   `main.py`: The production live-trading loop, checking configurations, tracking current positions, scanning symbols, and placing orders.
    *   `dashboard.py`: A premium dark-mode web console (Flask-based) showcasing real-time scanner metrics, active positions, wallet balance, and stream-tailing logs.
*   **Local Simulation & Verification Tools:**
    *   `backtest.py`: Fast event-driven backtesting engine across all 32 major cryptocurrency pairs with online-learning emulation.
    *   `evaluate_model.py`: Performance evaluator for ML filters using walk-forward holdout verification.

---

## 🛠️ Tech Stack & Requirements

*   **Python 3.10+**
*   **Core Libraries:** `pandas`, `numpy`, `xgboost`, `scikit-learn`, `requests`, `python-dotenv`, `flask`, `cryptography`
*   **Exchange API Integration:** Mudrex API via custom `mudrex` package (accessing Bybit linear derivatives).

---

## 📁 Repository Structure

```
├── main.py                     # Main trading bot loop
├── dashboard.py                # Flask dashboard server & UI template
├── config.py                   # Global system parameters, symbols, and leverage
├── data.py                     # Market data interface (Bybit API wrapper)
├── strategy.py                 # Core signal generation logic (V10)
├── risk.py                     # Position sizing and portfolio risk management
├── indicators.py               # Technical indicator computations (EMA, MACD, RSI, ADX, ATR, BB)
├── ml_model.py                 # MLFilter architecture, feature engineering & serialization
├── train_ml_v4.py              # Walk-forward parameter search & V4 model training script
├── evaluate_model.py           # ML validation and utility scoring script
├── backtest.py                 # Historical performance simulation engine
├── run.sh                      # Production service runner script (systemd control wrapper)
├── requirements.txt            # System dependencies list
└── ml_filter.pkl               # Production V4 serialized ensemble model
```

---

## 📈 Configuration & Parameters (`config.py`)

Key settings govern the bot's risk limits and execution rules:
*   **Pairs Supported:** 32 major USDT pairs (BTC, ETH, SOL, AVAX, etc.).
*   **Leverage:** 20x across all pairs.
*   **Timeframe:** 4-hour entry candles (`PRIMARY_TF`), resampled from 15-minute raw feeds (`DATA_TF`), paired with daily trend bias check (`HTF_TF`).
*   **Risk per Trade:** 1.5% of total capital per setup (`RISK_PER_TRADE`).
*   **Circuit Breakers:** Maximum daily loss cap of 8% (`DAILY_LOSS_LIMIT`), 4 concurrent active positions max (`MAX_POSITIONS`), and consecutive-loss cooldowns.

---

## 💻 Operations & Commands

Use the `./run.sh` script to manage the production trading processes under `systemd` (typically named `mudrex-bot.service` and `mudrex-dash.service`):

| Command | Action |
| :--- | :--- |
| `./run.sh start` | Launch the trading bot and the Flask web dashboard |
| `./run.sh stop` | Gracefully terminate both processes |
| `./run.sh restart` | Restart both processes to load the latest changes/models |
| `./run.sh status` | View the live systemd process logs and status details |
| `./run.sh logs` | Print and tail live trading bot output |
| `./run.sh dashlogs` | Print and tail dashboard execution logs |
| `./run.sh enable` | Configure both processes to auto-start on boot |
| `./run.sh disable` | Remove both processes from the boot start queue |

---

## 🧠 ML Model V4 Details

The V4 ensemble leverages **47 selected features** representing trend, momentum stability (`macd_hist_std_20`), volatility context (`adx_mean_20`), and price-action structure.

*   **LONG Ensemble:** Optimizes threshold to `0.35` to capture high-probability pullbacks with a `83.0%` validation pass rate.
*   **SHORT Ensemble:** Optimizes threshold to `0.45` to catch sharp trend-following breakdowns, yielding a validated expectancy of `+1.448R` per trade.
*   **Holdout Performance:** On unseen holdout data, V4 achieves a combined profit factor of `1.39` with a temporal holdout expectancy of `+0.746R` per trade.
