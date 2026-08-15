# Nordic day-ahead electricity price forecasting — benchmark report

A ten-model comparison of day-ahead price forecasts for the Swedish bidding
zone SE3, scored on proper probabilistic metrics and ranked on a Pareto
accuracy-versus-compute basis. This document reports the full evaluation; the
`README.md` carries the one-page summary. All figures are reproducible from
the committed snapshot (`data/snapshot/`), with no API key.

## 1. Problem and scoring rules

The task is a day-ahead point-in-time forecast: at the close of the day-ahead
auction, produce the distribution of prices for each market time unit (MTU)
of the next delivery day. The forecast is a three-quantile grid (P10, P50,
P90), not a single point, because a downstream decision (bidding, storage
sizing) needs the spread and the tail risk, not just the median.

The primary metric is the **continuous ranked probability score (CRPS)**, a
proper scoring rule for the full predictive distribution. A proper rule
reaches its minimum only at the true distribution, so it cannot be gamed by a
forecaster that hedges toward the center. CRPS is evaluated from the three
quantiles by the quantile-weighted average of the pinball loss, which is how
the literature approximates CRPS from a finite quantile grid.

The three metrics reported per model are:

- **CRPS** — the probabilistic accuracy score (lower is better).
- **Pinball loss** — the quantile-level loss that decomposes CRPS; reported at
  P10, P50, and P90 so the reader can see where a model's calibration and
  sharpness fail.
- **MAE** — the mean absolute error of the P50 (median) forecast, the standard
  point-forecast accuracy measure.

A fourth, derived measure is the **seasonal-naive skill score**, `1 − CRPS /
CRPS_baseline`, where the baseline repeats the price from the same hour one
week earlier. A negative skill score means the model is worse than the
seasonal baseline.

## 2. Data

Four sources feed the pipeline, joined at a single `assemble_data` seam:

- **ENTSO-E Transparency** — day-ahead prices, load and wind forecasts, net
  positions, scheduled exchanges, and neighbour prices; weekly hydro
  reservoir storage (document A72).
- **Open-Meteo** — six weather variables per zone centroid (temperature, wind
  at two heights, shortwave radiation, precipitation, snowfall).
- **ECB** — the SEK/EUR reference rate.
- **EEX** — the EUA carbon auction price.

The window is 2021-01-01 to 2026-08-13. Every covariate obeys an as-of timing
rule: a forecast made at time `t` uses only information published before `t`.
Day-ahead load and wind forecasts are excluded entirely because their values
are not known when the price is set; using them would leak the future. The
as-of rule and the exclusion serve one purpose: a model that sees a value
published after its forecast time looks good in a backtest and fails in
production, where that value is not yet available.

## 3. Models

Ten models across four families, all pinned to seed 42:

| family | models | feature set |
|---|---|---|
| classical | SARIMA, ETS | price-only |
| gradient-boosted | LightGBM, XGBoost, CatBoost | full-features |
| deep learning | N-BEATS (price-only), TFT (full-features) | per model |
| foundation | Chronos-2 (full-features), TimesFM 2.5 (price-only) | per model |
| linear baseline | LEAR | price-only |

Each model exposes the same `fit`/`predict_quantiles` contract and returns a
P10/P50/P90 grid per MTU. Foundation models are zero-shot (weights pretrained,
no training); the deep models run a pinned, two-epoch budget.

The four families were chosen to span the range of forecasting technique —
statistical, gradient-boosted, deep, and pretrained — so the comparison
measures where each approach earns its place rather than ranking variants of
one method.

## 4. Evaluation metrics

The three metrics answer different questions. CRPS judges the whole predictive
distribution, pinball loss shows where in the distribution a model is sharp or
miscalibrated, and MAE judges the median point forecast. The skill score gives
all three a common yardstick: is the model better than a trivial baseline?

For a delivery-day forecast with realised price `y_t` and quantile forecast
`Q_q(t)` at level `q`:

**Pinball loss** at level $q$:

$$PL_q = \frac{1}{n} \sum_{t=1}^{n} \begin{cases} q\,(y_t - Q_q(t)) & \text{if } y_t \ge Q_q(t) \\ (q-1)\,(y_t - Q_q(t)) & \text{otherwise} \end{cases}$$

**CRPS** from the three quantiles:

$$\text{CRPS} = \frac{2}{3} \sum_{q \in \{0.10,\,0.50,\,0.90\}} PL_q$$

**MAE** of the median:

$$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} \left| y_t - Q_{0.50}(t) \right|$$

**Skill score**:

$$\text{skill} = 1 - \frac{\text{CRPS}}{\text{CRPS}_\text{seasonal-naive}}$$

All metrics are computed per timestamp, then averaged within a fold and
across folds.

## 5. Experimental setup

The evaluation is an expanding-window walk-forward backtest, which is the only
setup that reproduces how a forecaster would be used day to day: each fold
trains only on data that would have been available at the time, then forecasts
the next day. Anything else (a random split, a single train/test cut) either
leaks the future or measures one lucky period instead of a track record.

- **Folds** — yearly refit. The hourly regime has three folds (forecast dates
  2023-01-01, 2024-01-01, 2025-01-01); the 15-minute regime has one
  (2026-01-01). Yearly refit was chosen over monthly because a month is too
  short to estimate the seasonal structure the models need, and over a single
  fit because one refit cannot show whether a model holds up across years.
- **Purge** — a 7-day purge around each structural boundary (flow-based market
  coupling on 2024-11-04, the MTU switch on 2025-10-01), so no fold straddles
  a break. The purge exists because a model trained on both sides of a market
  redesign learns a relationship that no longer holds on the other side.
- **Regimes** — the market moved from 60-minute to 15-minute settlement on
  2025-10-01. The two regimes are assembled and backtested as separate
  single-frequency windows because a 60-minute and a 15-minute price series are
  different processes, and mixing them into one frame would force a model to
  forecast an ill-defined frequency.
- **Seed** — every model is pinned to seed 42, because a benchmark that cannot
  be reproduced is not a benchmark.

## 6. Results

### 6.1 Headline ranking

Mean across the three hourly folds and one 15-minute fold, lower is better
for every accuracy column:

| model | family | CRPS | MAE | pinball P10 | pinball P50 | pinball P90 | skill | compute (s) |
|---|---|---|---|---|---|---|---|---|
| chronos2 | foundation | 9.61 | 15.45 | 3.75 | 7.73 | 2.94 | 0.514 | 3061.6 |
| catboost | gbdt | 9.64 | 15.50 | 2.34 | 7.75 | 4.38 | 0.411 | 1.8 |
| lgbm | gbdt | 9.72 | 15.37 | 3.56 | 7.68 | 3.34 | 0.453 | 0.5 |
| xgboost | gbdt | 10.63 | 15.70 | 2.16 | 7.85 | 5.93 | 0.317 | 1.8 |
| nbeats | deep | 14.39 | 18.40 | 3.18 | 9.20 | 9.21 | 0.256 | 38.6 |
| timesfm | foundation | 14.72 | 21.33 | 7.40 | 10.66 | 4.01 | 0.434 | 17.5 |
| sarima | classical | 14.99 | 21.69 | 4.24 | 10.85 | 7.40 | −0.114 | 166.6 |
| tft | deep | 20.12 | 30.20 | 9.80 | 15.10 | 5.27 | −0.707 | 640.6 |
| lear | ml | 26.29 | 26.29 | 21.72 | 13.15 | 4.57 | −0.283 | 2.5 |
| ets | classical | 33.40 | 52.81 | 11.21 | 26.41 | 12.48 | −1.331 | 147.8 |

The trees win because they fit a separate quantile regressor for P10, P50, and
P90 on the full feature matrix, so they model the price distribution directly
and use every exogenous group without leaking the future. Chronos-2 leads
without any training on Nordic data, which is the concrete demonstration that
a pretrained backbone transfers to a new market; it is also the only model
whose accuracy is bought with thousands of seconds of inference. The deep
models (N-BEATS, TFT) trail the trees not because deep learning is unsuited to
the task but because they run a pinned two-epoch budget sized for CPU smoke
tests, so this comparison understates a longer training run. LEAR collapses
its three quantiles to one point forecast — it carries no spread — and ETS is
a single-exponential smoother that cannot represent the regime structure.
Four models (SARIMA, TFT, LEAR, ETS) score below the seasonal-naive baseline,
so they are worse than repeating last week's price.

### 6.2 Per-regime

The ranking differs sharply between regimes, which the pooled headline hides:

| model | hourly CRPS | 15-min CRPS |
|---|---|---|
| chronos2 | 6.46 | 19.06 |
| timesfm | 6.58 | 39.13 |
| lgbm | 7.67 | 15.87 |
| catboost | 8.52 | 12.99 |
| xgboost | 10.05 | 12.36 |
| nbeats | 9.81 | 28.12 |
| sarima | 17.03 | 8.88 |
| tft | 25.27 | 4.66 |
| lear | 17.09 | 53.90 |
| ets | 35.45 | 27.24 |

Chronos-2 and TimesFM lead the hourly regime but degrade on the single
15-minute fold, where TFT and SARIMA rank first and second. Two forces explain
this. First, the foundation models are pretrained mostly on hourly data, so
their inductive bias fits the hourly structure better than a 96-slot day.
Second, the 15-minute regime has one fold, so its column is one day-ahead
forecast and carries far more variance than the hourly column; a single day's
ranking should be read as suggestive, not conclusive.

### 6.3 Pareto frontier

The frontier flags the models that are not dominated on both accuracy and
compute, computed within each regime so a model cheap in one regime does not
dominate a model in the other. The gradient-boosted trees sit on the frontier:
they reach CRPS within 0.1–1.0 of Chronos-2 at roughly 0.5–1.8 seconds of
compute, against Chronos-2's 3062 seconds. Chronos-2 is the accuracy anchor
only if inference cost is ignored.

### 6.4 Significance

Pairwise two-sided tests, on the per-fold mean hourly CRPS (three yearly
folds, treated as independent blocks). A negative statistic favours the first
model:

| model A | model B | DM statistic | p-value |
|---|---|---|---|
| chronos2 | catboost | −0.87 | 0.385 |
| chronos2 | lgbm | −0.97 | 0.332 |
| chronos2 | xgboost | −2.63 | 0.009 |
| catboost | lgbm | 0.73 | 0.466 |
| catboost | xgboost | −0.81 | 0.418 |
| lgbm | xgboost | −2.52 | 0.012 |

Two differences reach significance at the 5% level: XGBoost is significantly
worse than both Chronos-2 and LightGBM. The headline gap between Chronos-2 and
CatBoost (0.5% CRPS) is not significant (p = 0.385), and the gap between
Chronos-2 and LightGBM is not significant either (p = 0.332). With three
folds the test has low power, so a non-significant result means the evidence
is insufficient, not that the models are equal; the cross-zone result in 6.7
is the stronger signal for Chronos-2's accuracy edge.

### 6.5 Feature selection

Seven feature groups feed the full-features models. Group 1 (autoregressive
lags, rolling statistics, calendar features, regime label) is the base;
groups 3–7 are cross-border flows, weather, hydro, carbon, and FX. Group 2
(day-ahead load and wind forecasts) is excluded for leakage.

Forward selection adds cross-border flow first, cutting CRPS by 1.68; dropping
it from the full set costs 1.16 CRPS. Weather and hydro add less, and carbon
does not help. The efficient set (within 1% of the full-features CRPS) is the
base group plus cross-border. A transfer check confirms the set earns its cost
on the other full-features arms: the CRPS delta is negative for XGBoost
(−1.56), TFT (−0.89), CatBoost (−0.65), and Chronos-2 (−0.18).

The dominance of cross-border flow is what one would expect for a bidding zone
whose price is set at the margin by imports and exports: the net position and
the scheduled exchanges on the interconnectors carry the price pressure that a
purely local history misses. Carbon does not help because the EUA price is a
slow, market-wide series with little intra-day variation relative to a Nordic
system that is mostly hydro, wind, and nuclear, so it adds no signal beyond
what the autoregressive base already captures.
### 6.6 Tuning

A secondary comparison tunes the best model per family against its pinned
default; the default table stays the headline. SARIMA is tuned by an AIC/BIC
order grid, CatBoost and N-BEATS by bounded Optuna, and Chronos-2 by a
context-length sweep (its pretrained weights are never tuned). None of the
tuned configurations beat the default on the hourly regime: deltas are +0.28
(SARIMA), +1.08 (CatBoost), +26.40 (N-BEATS), +0.68 (Chronos-2). The N-BEATS
search overfit the single validation day.

### 6.7 Cross-zone robustness

The headline runs on SE3. A reduced run takes the frontier (the three trees)
plus Chronos-2 across SE1–SE4 at default configs, hourly regime. Chronos-2
leads every zone (mean CRPS 6.23, 6.00, 6.46, 7.14 for SE1–SE4), and the three
trees hold ranks two through four in each, so both the accuracy anchor and
the frontier generalise beyond SE3.

## 7. Discussion and limitations

The efficient choice for day-ahead SE3 price forecasting is a gradient-boosted
tree: it reaches near-frontier accuracy at a fraction of the compute of the
foundation models, and its ranking is stable across zones. Chronos-2 is the
accuracy anchor but only when inference cost is ignored; on the single
15-minute fold its advantage does not hold.

Three caveats bound these conclusions. First, the 15-minute regime has one
fold, so any ranking there is a single forecast and carries more uncertainty
than the hourly column. Second, the pooled headline weights the hourly regime
three-to-one over the 15-minute regime, so the "Chronos-2 leads" claim is
driven by the hourly regime. Third, the models are not hyperparameter-tuned by
default; the tuning pass shows the pinned defaults are already at or near the
frontier on the hourly regime.
