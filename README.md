# Trading Bot Python

This repository contains a Python trading bot focused on Brazilian stocks, built around technical analysis, risk management and Discord alerts.

The project provides two main operation modes:
- **Conservative mode**: multi‑timeframe confirmation (weekly + daily) with stricter filters before sending signals.
- **Aggressive mode**: more frequent signals based mainly on daily data, suitable for faster entries.

---

## Features

- Technical analysis on Brazilian tickers (candles, moving averages, RSI, MACD, ATR, volume, risk–reward).
- Conservative engine with multi‑timeframe confirmation and rich explanations for each signal.
- Aggressive engine with faster alerts and support for short (sell) signals.
- SQLite database to store all alerts and avoid sending duplicate signals.
- Optional machine learning filter (Random Forest) to validate signals when there is enough historical data.
- Discord integration with rich embeds (colors, emojis, risk–reward, stops and targets).
- Logging system to keep track of every analysis and alert sent.

---

## Project structure

- `bot_analise_tecnica.py`  
  Core technical analysis logic: data loading, indicators, signal generation, helpers for risk/reward, etc.

- `bot_analise_tecnica_runner_conservador.py`  
  Conservative runner.  
  Runs multi‑timeframe analysis (1W + 1D), compares with the last saved alert, decides whether to send a new Discord alert and stores the result in SQLite.

- `bot_aggressivo_runner.py`  
  Aggressive runner.  
  Runs a faster daily analysis for each ticker, saves alerts and sends Discord embeds whenever the signal changes.

- `machine_learning_pipeline.py`  
  Machine learning pipeline (Random Forest) used as an additional opinion/filter on top of the technical rules.

- `ml_validator.py`  
  Small helper that checks whether there is enough labeled data to safely use the ML model.

- `logger_bot.py`  
  Central logging configuration used by the runners and helpers.

- `check_db.py`, `import_sqlite3.py`, `inicia_alerts.py`  
  Utilities to create/inspect the SQLite database and make sure the `alerts` table is ready.

---

## Requirements

The bot was built with Python 3.10+.

Main third‑party libraries:

- `pandas`
- `numpy`
- `yfinance`
- `discord.py`
- `python-dotenv`
- `pytz`
- `scikit-learn` (RandomForest, TimeSeriesSplit, metrics)
- `joblib`

Standard library modules:

- `logging`
- `sqlite3`
- `datetime`
- `math`
- `asyncio`
- `os`
- `time`

### Installation

Create and activate a virtual environment, then install the dependencies (after you create a `requirements.txt`):

python -m venv .venv

Windows
.venv\Scripts\activate

Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

To generate `requirements.txt` from your current environment:

pip freeze > requirements.txt

text

---

## Environment variables

Create a `.env` file in the project root (this file is ignored by Git) with values similar to:

DISCORD_TOKEN_CONSERVADOR=your_conservative_bot_token_here
DISCORD_CHANNEL_ID_CONSERVADOR=123456789012345678

DISCORD_TOKEN_AGRESSIVO=your_aggressive_bot_token_here
DISCORD_CHANNEL_ID_AGRESSIVO=123456789012345678

DB_FILE=alerts.db

---

## How to run

Conservative mode (multi‑timeframe, stricter signals):

python bot_analise_tecnica_runner_conservador.py


Aggressive mode (more frequent signals):

python bot_aggressivo_runner.py


Both runners will:

1. Load the list of tickers from the configuration in `bot_analise_tecnica.py`.  
2. Run the analysis for each ticker.  
3. Save alerts into the SQLite database configured in `DB_FILE`.  
4. Build a rich Discord embed and send it to the configured channel.

---

## Manual commands (Discord)

The bot also exposes manual commands in the Discord channel (for example, a command like `!analise` to trigger a forced analysis), so that you can request a fresh scan on demand.

Check the runner files for the list of supported commands and how they are handled.

---

## Disclaimer

This project was built for educational and personal use only.  
It does **not** constitute financial advice. Use it at your own risk and always validate signals before trading.
