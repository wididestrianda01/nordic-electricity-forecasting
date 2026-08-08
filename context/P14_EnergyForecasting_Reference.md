# P4 — Nordic Electricity Price Forecasting

> **Goal:** Build a multi-model day-ahead electricity price forecasting system
> for Swedish bidding zones (SE1–SE4) using ENTSO-E data, weather features, and
> cross-commodity LNG/gas variables — and deploy an interactive forecast
> dashboard. This project leverages your 4 years of LNG industry experience to
> create a portfolio piece no other KTH Financial Mathematics student can
> replicate.

- - -
## 📋 Project Overview


|Item|Detail|
|-|-|
|**Priority**|\#4 — build in Summer 2026 (July), after P2|
|**Duration**|Summer 2026, ~60–80 hours total|
|**Deliverable**|Multi-model forecast comparison + Streamlit dashboard + GitHub repo + LaTeX report|
|**Unique angle**|Spark spread analysis (gas-to-power) using your LNG background|
|**Languages**|Python|
|**Key output files**|`energy_engine/data_pipeline.py`, `energy_engine/features.py`, `energy_engine/models.py`, `energy_engine/evaluation.py`|
|**Target companies**|Vattenfall, Statkraft, Aurora Energy Research, Nord Pool, Uniper, Modity Energy Trading, OX2|

> **Your competitive advantage:** You understand LNG supply dynamics, gas
> pricing, and commodity risk from 4 years at PT Badak NGL. No other KTH
> Financial Mathematics student has this. The spark spread analysis in this
> project directly bridges your LNG experience to Nordic power markets.

- - -
## 📚 Books — Read in This Order

### Primary (hands-on, code every chapter)


|Book|Author|Chapters for P4|
|-|-|-|
|**Python for Finance Cookbook** *(owned)*|Lewinson|**Ch. 6 ALL** (decomposition · stationarity · ARIMA · auto-ARIMA) · **Ch. 7 ALL** (time-series CV · feature engineering · ML as regression · Prophet · PyCaret) · Ch. 9 GARCH volatility|
|**Streamlit for Data Science** *(owned)*|Richards|Ch. 6–9 live data + multi-page + deploy|

### Theory background (selective reading)


|Book|Author|Chapters for P4|
|-|-|-|
|**Introductory Econometrics for Finance** (4th ed.)|Brooks|Ch. 5 ARIMA · Ch. 8 Markov switching *(also used in P7)*|

> **Lewinson Ch. 6 and Ch. 7 back-to-back** before writing any forecasting code.
> Ch. 6 handles the statistical baseline (ARIMA), Ch. 7 handles the ML approach
> (feature engineering + cross-validation for time series).

- - -
## 📄 Papers — All Free


|Paper|Author|Year|What It Gives You|Find It|
|-|-|-|-|-|
|**Probabilistic Forecasting of Electricity Prices: A Review**|Nowotarski & Weron|2018|The canonical survey — cite as literature review foundation|Google Scholar → `Nowotarski Weron 2018 electricity price forecasting review`|
|**Forecasting Day-Ahead Electricity Prices: A Review of State-of-the-Art Algorithms**|Lago et al.|2021|LEAR model — your primary benchmark|Google Scholar → `Lago 2021 LEAR electricity price forecasting` · also GitHub epftoolbox|
|**Forecasting Volatility of the Nordic Electricity Market**|Dahl et al.|2025|MSGARCH for Nordic power — directly extends your GARCH module to P4|[mdpi.com/2227-9091/13/3/58](https://www.mdpi.com/2227-9091/13/3/58)|
|**Day-Ahead Electricity Price Prediction Applying Hybrid Model**|Multiple|2021|LSTM benchmarks for electricity markets|[arxiv.org/abs/2101.05249](https://arxiv.org/abs/2101.05249)|
|**Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting**|Lim et al.|2021|Your advanced extension model|Google Scholar → `Lim 2021 temporal fusion transformer`|

- - -
## 📜 Regulation — All Free


|Document|Issuer|Relevance|Link|
|-|-|-|-|
|**REMIT — Regulation 1227/2011**|EU|European energy market integrity and transparency regulation|[eur-lex.europa.eu — CELEX:32011R1227](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32011R1227)|
|**ENTSO-E Network Codes**|ENTSO-E|Rules governing day-ahead and intraday markets|[entsoe.eu/network_codes](https://www.entsoe.eu/network_codes/)|
|**Nord Pool Market Rules**|Nord Pool|Bidding zone structure, auction rules, price formation|[nordpoolgroup.com/en/trading/rules-and-regulations](https://www.nordpoolgroup.com/en/trading/rules-and-regulations)|

> **Regulatory framing for your report:** Frame your forecasting model in the
> context of REMIT compliance — market participants must not use inside
> information to trade. A systematic algorithmic price forecast based only on
> public data (ENTSO-E + weather) is REMIT-compliant. This gives your project a
> practical industry angle.

- - -
## 🌐 Free Online Resources


|Resource|URL|Used For|
|-|-|-|
|**epftoolbox**|[github.com/jeslago/epftoolbox](https://github.com/jeslago/epftoolbox)|LEAR model implementation — start here before building your own|
|**entsoe-py**|[github.com/EnergieID/entsoe-py](https://github.com/EnergieID/entsoe-py)|Python client for ENTSO-E Transparency Platform API|
|**Open-Meteo API**|[open-meteo.com](https://open-meteo.com)|Free weather data — no API key needed for basic use|
|ENTSO-E Transparency Platform|[transparency.entsoe.eu](https://transparency.entsoe.eu)|Register for free API key|
|Nord Pool market data|[nordpoolgroup.com/en/market-data1](https://www.nordpoolgroup.com/en/market-data1)|Historical spot prices (also accessible via ENTSO-E)|
|EEX TTF gas prices|[eex.com/en/market-data/natural-gas](https://www.eex.com/en/market-data/natural-gas)|TTF gas price for spark spread calculation|
|Statkraft algorithmic trading blog|[statkraft.com/newsroom/2020/physical-algorithmic-trading](https://www.statkraft.com/newsroom/news-and-stories/2020/physical-algorithmic-trading-quick-decisions/)|How your target employer actually uses these models|
|darts library|[unit8co.github.io/darts](https://unit8co.github.io/darts/)|Unified time series forecasting library (ARIMA → TFT)|

- - -
## 🗄️ Data Sources


| Source                               | What You Get                                                                             | How to Access                                                                                                                            |
| ------------------------------------ | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **ENTSO-E Transparency Platform**    | Day-ahead prices (SE1–SE4) · load · wind/solar forecasts · capacity · cross-border flows | Register at transparency.entsoe.eu → email [transparency@entsoe.eu](mailto:transparency@entsoe.eu) for API key → `pip install entsoe-py` |
| REDACTED_API_KEY |                                                                                          |                                                                                                                                          |
| **Open-Meteo**                       | Temperature · wind speed · solar radiation for Sweden                                    | Free, no key needed: `pip install openmeteo-requests`                                                                                    |
| **EEX TTF gas futures**              | Natural gas price for spark spread                                                       | Download CSV from eex.com/market-data                                                                                                    |
| **Nord Pool**                        | Historical spot prices by bidding zone                                                   | nordpoolgroup.com/en/market-data1 (free CSV download)                                                                                    |
| **Riksbank**                         | SEK/EUR exchange rate (for cross-commodity analysis)                                     | riksbank.se/en-gb/statistics                                                                                                             |

```python
# ENTSO-E data in a few lines
from entsoe import EntsoePandasClient

client = EntsoePandasClient(api_key="YOUR_KEY")

start = pd.Timestamp("2018-01-01", tz="Europe/Stockholm")
end   = pd.Timestamp("2024-12-31", tz="Europe/Stockholm")

# Day-ahead prices for SE3 (Stockholm area)
prices = client.query_day_ahead_prices("SE3", start=start, end=end)

# Wind generation forecast
wind = client.query_wind_and_solar_forecast("SE3", start=start, end=end)

# Load forecast
load  = client.query_load_forecast("SE", start=start, end=end)
```
- - -
## 🗓️ Phase Timeline


|Week|Phase|Topic|Key Task|
|-|-|-|-|
|1|Foundation|Data pipeline + EDA|ENTSO-E setup · price series for SE1–SE4 · missing data|
|1|Foundation|**MINI: Nordic price EDA**|Seasonality · negative prices · 2022 crisis · bidding zone spreads|
|2|Phase 1|Weather + gas feature engineering|Wind speed · temperature · TTF gas → spark spread|
|2|Phase 1|**MINI: Spark spread analyser**|Power price − gas × heat rate = spark spread time series|
|3|Phase 2|Statistical baseline|Lewinson Ch. 6 · ADF test · auto-ARIMA on SE3|
|4|Phase 2|LEAR model|epftoolbox · LASSO-estimated autoregressive features|
|4|Phase 2|GARCH volatility|Lewinson Ch. 9 · volatility of day-ahead prices|
|5|Phase 3|ML models|XGBoost + weather features · Lewinson Ch. 7|
|5|Phase 3|**MINI: Model comparison table**|MAPE · MAE · RMSE · Winkler score per model|
|6|Phase 4|Regime detection extension|MSGARCH or HMM for price volatility regimes|
|7|Phase 4|Report + REMIT framing|LaTeX report · regulatory section|
|8|Deploy|**Streamlit forecast dashboard**|Live price display · model toggle · deploy|

- - -
## 🐍 Python Stack

```python
# Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ENTSO-E data
from entsoe import EntsoePandasClient

# Weather data
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Statistical models
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# GARCH
from arch import arch_model

# ML forecasting
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import shap

# Unified forecasting (optional)
from darts import TimeSeries
from darts.models import NBEATSModel, TFTModel

# Electricity-specific
from epftoolbox.evaluation import MAE, MAPE, RMSE

# App
import streamlit as st
```
- - -
## 📁 Folder Structure

```
nordic-energy-forecast/
├── CLAUDE.md
├── README.md                 ← Key result: "XGBoost reduced MAPE by 22% vs ARIMA"
├── requirements.txt
├── energy_engine/
│   ├── __init__.py
│   ├── data_pipeline.py      ← ENTSO-E fetch, weather merge, gas price join
│   ├── features.py           ← spark spread, lag features, calendar vars
│   ├── models.py             ← ARIMA, LEAR, XGBoost, GARCH wrappers
│   └── evaluation.py         ← MAPE, MAE, RMSE, Winkler, Diebold-Mariano
├── notebooks/
│   ├── 01_data_pipeline_eda.ipynb
│   ├── 02_feature_engineering_spark_spread.ipynb
│   ├── 03_statistical_baseline_arima_lear.ipynb
│   └── 04_ml_models_comparison.ipynb
├── app/
│   └── streamlit_app.py      ← live forecast dashboard
├── data/
│   └── README.md             ← data provenance, download instructions
├── reports/
│   └── p4_energy_report.tex
└── tests/
    └── test_features.py
```
- - -
## 🔑 Key Concepts to Know Cold

- **Nordic bidding zones (SE1–SE4):** Sweden is divided into 4 price areas. SE3
  (Stockholm/Gothenburg) typically has the highest consumption. Cross-border
  flows with Norway (NO1–NO5), Denmark, and Germany drive price formation.
- **Merit-order effect:** Electricity price is set by the most expensive marginal
  plant needed to meet demand. When renewables produce heavily, gas plants move
  up the merit order — critical for understanding when gas prices drive
  electricity prices.
- **Spark spread:** Power price − (gas price × heat rate). Measures profitability
  of gas-fired generation. When spark spread narrows, gas plants shut down and
  prices decouple from gas. Your LNG background makes this analysis credible.
- **Negative electricity prices:** Occur when must-run renewables exceed demand.
  Nordic prices go negative frequently — your model must handle this without
  taking log transforms.
- **Day-ahead auction:** Prices are set daily at 12:00 CET for the following day.
  This is the primary market you're forecasting. Intraday is separate.
- **MAPE vs RMSE for electricity:** MAPE is problematic when prices approach zero
  (division by zero). Use Winkler score for probabilistic forecasts or sMAPE as
  a fallback.
- **LEAR model:** Lasso-Estimated AutoRegressive model. Uses prices from the
  previous 7 days + load forecast + renewable forecast as features. Simple but
  hard to beat — use as your baseline.
- **Diebold-Mariano test:** Statistical test for whether one forecast is
  significantly better than another. Essential for publication-quality results
  and thesis extension.

- - -
## 🔑 The Spark Spread — Your Unique Contribution

```python
# Spark spread calculation
# Formula: Spark Spread = Electricity Price - (Gas Price × Heat Rate)
# Heat rate for CCGT: ~7.5 GJ/MWh (or ~2.08 MWh gas per MWh electricity)

def compute_spark_spread(
    power_price_eur_mwh: pd.Series,
    gas_price_eur_mwh: pd.Series,   # convert from EUR/MWh or EUR/MMBtu
    heat_rate: float = 7.5           # GJ/MWh for a combined cycle gas turbine
) -> pd.Series:
    """
    Compute the clean spark spread (ignoring carbon cost for simplicity).
    Positive spread = gas generation is profitable = gas price drives power price.
    """
    return power_price_eur_mwh - (gas_price_eur_mwh * heat_rate / 3.6)

# In your cover letter to Vattenfall/Statkraft:
# "My project includes a spark spread analysis for Swedish bidding zones,
# connecting gas price dynamics to power price formation — a perspective
# informed by my four years working in LNG production at PT Badak NGL."
```
- - -
## ✍️ Report Structure (LaTeX / Overleaf)

1.  Abstract — best MAPE result + key finding upfront
2.  Nordic Power Market — bidding zone structure, merit order, gas-power nexus
3.  Data & Feature Engineering — ENTSO-E pipeline, weather features, spark
    spread construction
4.  Methodology — ARIMA baseline, LEAR, GARCH volatility, XGBoost with feature
    importance
5.  Results — MAPE/MAE/RMSE comparison table, SHAP feature importance,
    Diebold-Mariano tests
6.  Spark Spread Analysis — gas-power relationship, regime shifts, 2022 energy
    crisis
7.  **Regulatory Context** — REMIT compliance, use of public data only
8.  Limitations & Extensions — probabilistic forecasting, regime switching,
    real-time intraday

- - -
## 💬 Interview Talking Points

- *"Why are Nordic electricity prices hard to forecast?"* → Nordic prices are
  driven by hydro reservoir levels (stochastic), wind intermittency,
  cross-border flows with Germany and Norway, and periodic negative prices. Gas
  price dynamics create regime shifts. No single model dominates across all
  market conditions.
- *"What is the merit-order effect and why does it matter for your model?"* →
  Electricity price is set by the marginal cost of the last unit dispatched.
  When solar/wind generation is high, gas plants set the price — so TTF gas
  prices become a key feature. My spark spread analysis quantifies this
  relationship explicitly.
- *"How does your LNG background inform this project?"* → LNG supply disruptions
  directly feed into European gas prices, which then drive Nordic electricity
  prices through the merit order. I modeled this gas-to-power transmission
  channel using a spark spread time series derived from my understanding of gas
  commodity markets.
- *"What is the LEAR model and why is it your baseline?"* → LEAR
  (Lasso-Estimated AutoRegressive) uses lagged prices plus load and renewable
  forecasts as features, with LASSO regularization for feature selection. It
  consistently outperforms naive ARIMA in electricity price forecasting and is
  the de facto benchmark in the EPF literature (Lago et al. 2021).
- *"How do you evaluate forecast quality when prices go negative?"* → MAPE fails
  near zero because of division problems. I use MAE (robust to zero), RMSE
  (penalises large errors — important for risk management), and the
  Diebold-Mariano test to assess whether model A is statistically better than
  model B.
