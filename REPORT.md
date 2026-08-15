# Nordic day-ahead electricity price forecasting: a complete study

This document is a full account of the project, written to be read end to end.
It explains what was done, how, and why at every step. It covers the data
pipeline, the preprocessing, the data split, the training and validation
scheme, the evaluation metrics, the ten models, the results, the challenges,
and the design decisions. The `README.md` carries a one-page summary; this
document carries the reasoning.

## 1. The problem

A day-ahead electricity market clears once a day. Every hour (and, since
October 2025, every 15 minutes) of the next day, a price is set for each
bidding zone. A buyer or seller who can forecast that price before the auction
closes can bid better.

This project forecasts day-ahead prices for the Swedish bidding zone SE3. The
forecast is a distribution, not a single number. It returns three quantiles:
the 10th, 50th, and 90th percentiles (P10, P50, P90). A distribution matters
because a downstream decision needs the spread and the tail risk, not just the
median. A storage operator wants to know how low the price might fall, not only
its average.

The goal of the project is narrower than "predict well". It compares ten
forecasting models from four families and ranks them on accuracy against
compute cost. The answer to "which model" depends on both axes. A model that is
slightly more accurate but a thousand times more expensive is not always the
right choice.

## 2. The data pipeline

### 2.1 The sources

Four external sources feed the pipeline:

- **ENTSO-E Transparency**. This is the European grid operator's data platform.
  It provides day-ahead prices, load and wind forecasts, net positions,
  scheduled exchanges, neighbour-zone prices, and weekly hydro reservoir
  storage.
- **Open-Meteo**. This is a free weather API. It provides six weather variables
  per zone: temperature, wind speed at two heights, shortwave radiation,
  precipitation, and snowfall.
- **ECB**. This is the European Central Bank. It provides the SEK/EUR exchange
  rate.
- **EEX**. This is the European Energy Exchange. It provides the EUA carbon
  auction price.

Each source answers a different question about the price. The price of
electricity in Sweden is set by supply and demand. Demand follows load and
weather. Supply follows wind, hydro levels, and imports and exports through the
interconnectors. Carbon and the exchange rate shift the cost of generation.
Each source is therefore a candidate driver of the price.

### 2.2 The assembly seam

Every source is fetched in one place: the `assemble_data` function. It joins
the six fetchers (market, weather, hydro, cross-border, FX, carbon) into one
dataframe indexed by market time unit (MTU). The result is a single table where
each row is one MTU and each column is one feature or the target price.

This single seam has one job: it is the only place the code talks to the
network. Everything downstream reads from memory. This makes the pipeline easy
to reason about. There is one point where data enters, and one point where a
cache can sit.

### 2.3 The cache

`assemble_data` is backed by a Parquet cache. The cache stores the joined
frame, keyed by the zone list and the date range. A second run with the same
key reads from disk instead of the network.

The cache exists because the fetch is slow and the sources can be flaky. The
first fetch takes minutes. A re-run should not pay that cost again. The cache
is an optimisation, never a dependency. A failed read falls back to a fresh
fetch. A failed write is ignored, so a fetch is never thrown away because the
cache was full or locked.

The window is fixed. Historical slices are immutable. There is no incremental
fetch. If the end date moves, that is a new key and a new fetch. This is
deliberate: the benchmark window is fixed, so a partial-update mechanism would
add complexity for no benefit.

## 3. Preprocessing

The raw data is not ready for a model. Several steps turn it into features.

### 3.1 The as-of rule (why leakage matters)

Every feature obeys one rule: a forecast made at time `t` may use only
information published before `t`. This is the as-of rule. It exists because of
leakage.

Leakage is the single most common way a forecasting model is silently broken.
If a model sees a value that was not available at forecast time, it looks
excellent in a backtest and fails in production. The classic case is using a
day-ahead load forecast as an input to a day-ahead price forecast. The load
forecast is published at the same time as the price is set. It cannot be known
before the price forecast is made. Using it leaks the answer.

This project excludes the day-ahead load and wind forecasts for exactly that
reason. They are day-ahead values. They are not known when the price forecast
is produced. The price-only autoregressive features (lags of the price itself)
are safe, because a lag uses a past price, which was published in the past.

### 3.2 Resolution and forward-fill

The sources have different resolutions. Prices are hourly (and 15-minute after
October 2025). Hydro storage is weekly. Carbon and FX are daily. Weather is
hourly.

Coarser series are forward-filled to the MTU grid. A weekly hydro value is
repeated across every MTU of the week it covers. This is correct for a slow
state variable: the reservoir level changes slowly, so the last published value
is the best available estimate for each hour. Forward-fill is the standard way
to align a slow covariate onto a fine grid.

### 3.3 Feature groups

Seven feature groups feed the full-features models:

1. **Autoregressive, calendar, regime**. Price lags at 1, 2, 3, 7, 14, and 28
   days. Rolling 7-day mean and standard deviation. Cyclical hour, day-of-week,
   and month (sine and cosine, so the circular structure is preserved). A
   Swedish holiday flag. A regime label from a hidden Markov model.
2. **System fundamentals**. Day-ahead load and wind forecasts. Excluded for
   leakage, as explained above.
3. **Cross-border**. Net positions, scheduled exchanges, and neighbour prices.
   These capture the import/export pressure on the zone.
4. **Weather**. The six variables per zone.
5. **Hydro**. Weekly reservoir storage.
6. **Commodities**. The EUA carbon price.
7. **FX**. The SEK/EUR rate.

Group 1 is the always-present base. Groups 3 through 7 are the exogenous
groups. Group 2 is excluded by design.

### 3.4 Why cyclical encoding

Hour, day-of-week, and month are circular. Hour 23 is one step from hour 0, but
their raw numbers (23 and 0) are far apart. A model that reads raw hours would
see a false gap. The sine and cosine encoding maps each circular value onto two
coordinates on a circle, so hour 23 and hour 0 are adjacent. This preserves the
structure a linear or tree model would otherwise miss.

### 3.5 Boundary masking

The market changed structure twice: flow-based coupling on 4 November 2024, and
the move to 15-minute settlement on 1 October 2025. A lag or rolling window
that crosses one of these boundaries mixes two different price regimes. The
value is not a clean feature.

Boundary masking sets such cells to NaN. A lag that reaches across a boundary
is dropped. The tree models treat NaN natively. The neural and foundation
models receive a zero-fill for NaN, because they require clean float inputs.
The masking prevents a model from learning a relationship that held on one side
of the break but not the other.

## 4. Data split, training, validation

### 4.1 Why walk-forward, not a random split

A random train/test split is wrong for time series. It shuffles the future into
the training set. A model trained on shuffled data has already seen prices from
after its test period. That is leakage, in a different form.

The correct scheme for a forecaster is a walk-forward (or rolling-origin)
evaluation. Each fold trains only on data before its test period, forecasts
the next day, and scores against the realised price. This reproduces how the
model would be used in production: fit on the past, predict the next day,
repeat.

### 4.2 The folds

The evaluation uses an expanding window. Each fold trains on all data before
the fold's forecast date, and the window grows as time moves forward. The
hourly regime has three folds, forecast on 1 January 2023, 1 January 2024, and
1 January 2025. The 15-minute regime has one fold, forecast on 1 January 2026.

Refit is yearly. A yearly refit was chosen over a monthly one because a month
is too short to estimate the seasonal structure the models need. It was chosen
over a single fit because one fit cannot show whether a model holds up across
years. The trade-off is that three folds is a small sample, which limits how
firmly the results can be stated. That limitation is discussed in section 9.

### 4.3 The purge

Each structural boundary carries a 7-day purge. No fold's forecast window or
training window crosses a boundary. The purge exists because a model trained on
both sides of a market redesign learns a relationship that does not hold on the
other side.

### 4.4 Regimes

The market moved from hourly to 15-minute settlement on 1 October 2025. The two
periods are different processes. An hourly price series and a 15-minute series
cannot be mixed into one frame, because the target's frequency changes. The two
regimes are therefore assembled and evaluated as separate single-frequency
windows. The headline table averages the results across both.

### 4.5 Train and validation

There is no separate held-out validation set in the classic sense. The
walk-forward folds themselves are the validation: each fold is a genuine
out-of-sample test. The tuning pass (section 8.6) uses the folds the same way.
This is the correct structure for time series, where a single static validation
split would either leak or measure one lucky period.

### 4.6 The seed

Every model is pinned to seed 42. A benchmark that cannot be reproduced is not
a benchmark. The seed removes one source of run-to-run variance, so two people
running the same code get the same numbers.

## 5. Methodology, step by step

Here is the full procedure, in order, with the reason for each step.

1. **Fetch the data.** Call `assemble_data` for the window 2021-01-01 to
   2026-08-13. This hits the four sources and joins them into one MTU-indexed
   frame. Reason: one entry point, one cache, one place to reason about the
   network.

2. **Split into regimes.** Separate the frame into the hourly window (before
   1 October 2025) and the 15-minute window (after). Reason: the two regimes
   are different processes with different target frequencies.

3. **Generate folds.** For each regime, compute the yearly forecast dates, skip
   any within 7 days of a boundary. Reason: a walk-forward with a purge is the
   only setup that avoids leakage and structural breaks.

4. **Build features.** For each fold, build the price-only and full-features
   matrices from the training window, and the horizon matrix for the forecast
   day. Reason: the model must see history as of the forecast time, and the
   horizon must contain no value published after that time.

5. **Fit.** Train the model on the training window. Price-only models get the
   price target alone. Full-features models get the feature matrix. Reason:
   each family consumes the matrix that matches its capability.

6. **Predict.** Produce the P10/P50/P90 grid for the forecast day. Reason: the
   deliverable is a distribution, not a point.

7. **Score.** Compute CRPS, pinball loss, and MAE against the realised price.
   Reason: three metrics answer different questions about accuracy.

8. **Repeat.** Move to the next fold. The window expands. Reason: each fold is
   an independent out-of-sample test.

9. **Aggregate.** Average each metric across folds. Reason: one fold is a
   single day; the average is the track record.

10. **Rank.** Sort models by CRPS, and flag the Pareto frontier on accuracy
    against compute. Reason: the answer depends on both axes.

## 6. Evaluation metrics

### 6.1 The three metrics

The three metrics answer three different questions.

- **CRPS** judges the whole predictive distribution.
- **Pinball loss** shows where in the distribution a model is sharp or
  miscalibrated.
- **MAE** judges the median point forecast.

For a forecast of price $y_t$ with quantile forecast $Q_q(t)$ at level $q$:

**Pinball loss** at level $q$:

$$PL_q = \frac{1}{n} \sum_{t=1}^{n} \begin{cases} q\,(y_t - Q_q(t)) & \text{if } y_t \ge Q_q(t) \\ (q-1)\,(y_t - Q_q(t)) & \text{otherwise} \end{cases}$$

**CRPS** from the three quantiles:

$$\text{CRPS} = \frac{2}{3} \sum_{q \in \{0.10,\,0.50,\,0.90\}} PL_q$$

**MAE** of the median:

$$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} \left| y_t - Q_{0.50}(t) \right|$$

**Skill score**:

$$\text{skill} = 1 - \frac{\text{CRPS}}{\text{CRPS}_\text{seasonal-naive}}$$

All metrics are computed per timestamp, then averaged within a fold and across
folds.

### 6.2 What CRPS means

CRPS is a proper scoring rule for a distribution. It measures the distance
between the predicted cumulative distribution and a step function at the
observed value. The formula above computes it from three quantiles, which is
the standard approximation when only a few quantiles are available.

"Proper" has a precise meaning. A proper scoring rule reaches its lowest
expected value at the true distribution. A forecaster who hedges toward the
centre cannot game it, because the centre is penalised when the truth is in the
tails. This is why CRPS is used for probabilistic forecasting and a simple
accuracy measure is not.

Lower CRPS is better. The units are the units of the price, so a CRPS of 9.6
means roughly a 9.6 EUR/MWh average distance between the forecast distribution
and the truth.

### 6.3 What pinball loss means

Pinball loss is the loss at a single quantile. At the 90th percentile, it
penalises an over-forecast lightly and an under-forecast heavily, because the
90th percentile should be exceeded only 10% of the time. This asymmetry is what
makes it a quantile loss.

The three pinball values (P10, P50, P90) show where a model fails. A model with
a high P10 pinball is poor at the lower tail. A high P90 pinball means a poor
upper tail. Reading the three together shows whether the errors are in the
tails or the centre.

### 6.4 What MAE means

MAE is the mean absolute error of the median forecast. It answers a simple
question: how far is the central forecast from the truth, on average. It is
less sensitive to outliers than root mean squared error, which squares the
errors. For a price series with occasional spikes, MAE is the more honest
central measure.

### 6.5 Calibration and sharpness

Two properties of a probabilistic forecast matter, and they pull against each
other.

- **Calibration** asks: does the P90 interval contain the truth about 90% of
  the time? A calibrated model is honest about its uncertainty.
- **Sharpness** asks: how narrow are the intervals? A sharp model is confident.

A trivial forecast can be perfectly calibrated by being very wide, but that is
not useful. A useful forecast is both calibrated and sharp. CRPS rewards both:
a wide forecast has a high CRPS even if calibrated, and a narrow but wrong
forecast is penalised heavily. This is why CRPS, not interval width alone, is
the primary metric.

### 6.6 The skill score and its baseline

The skill score compares a model to a baseline. The baseline here is
seasonal-naive: repeat the price from the same hour one week earlier. A skill
score of 0 means the model ties the baseline. A negative score means the model
is worse than simply repeating last week.

The baseline matters because it is the floor a model must beat to be worth
anything. Four of the ten models (SARIMA, TFT, LEAR, ETS) score below it. That
is a useful finding: a model can be sophisticated and still lose to a trivial
rule.

## 7. The ten models

Each model is described by what it is, how it works, its strengths, its
weaknesses, and when to use it.

### 7.1 SARIMA

Seasonal ARIMA fits an autoregressive-moving-average model with differencing to
a single price series. The version here is the "airline" model, SARIMA(0,1,1)
with a one-day seasonal period (24 steps hourly, 96 steps at 15 minutes).

It captures three things. The autoregressive part uses past prices. The moving
average part smooths shocks. The differencing removes trend and seasonality so
the remaining series is stationary, which the model can then fit.

Strengths: interpretable, few parameters, good for a series with clear trend
and seasonality, a strong classical baseline. Weaknesses: linear, produces a
point forecast that needs a separate step to become a distribution, and does
not use exogenous features. It also fits slowly at a 96-step seasonal period.
Use it when the series is well-behaved and you want a transparent baseline.

### 7.2 ETS

Exponential smoothing (ETS) decomposes the series into level, trend, and
seasonal components and smooths each with a weighted average of past values.
The version here is additive Holt-Winters: additive trend plus additive
seasonality.

Strengths: simple, fast, interpretable, and a useful benchmark. Weaknesses:
linear, no exogenous features, a point forecast, and poor when the series has
structural breaks, because a single smoothed level cannot represent two regimes.
Use it as a cheap baseline against which everything else is measured.

### 7.3 LEAR

LEAR (Lasso Estimated AutoRegressive) is the benchmark of Lago et al. for
electricity price forecasting. It fits a regularised linear regression on a
large set of lagged prices. The Lasso (L1) penalty drives most lag weights to
zero, so the model keeps only the lags that matter.

Strengths: parsimonious, fast, interpretable, and state-of-the-art among linear
electricity-price models. Weaknesses: linear, so it cannot capture nonlinear
price formation. In this repository its three quantiles collapse to a single
point, so it carries no spread. Use it as the linear reference.

### 7.4 LightGBM, XGBoost, CatBoost (gradient-boosted trees)

These three are gradient-boosted decision trees. Each trains an ensemble of
shallow trees that correct the errors of the previous trees. For probabilistic
forecasting, each fits three separate models, one per quantile (P10, P50, P90),
using the quantile (pinball) loss.

The three differ in implementation detail. LightGBM is fast and memory-light
and handles missing values natively. XGBoost uses a more regularised objective.
CatBoost handles categorical features natively, which suits the regime label.

They share the same strengths: nonlinear, able to use all exogenous features,
native handling of missing values (the NaN from lags and boundary masking),
fast to train, and very accurate on tabular data. Their shared weakness is
interpretability: a forest of trees is harder to explain than a linear model.
They can also overfit if the depth or the number of trees is too high.

Use them when you have rich tabular features and want strong accuracy without
hand-tuning. They are the workhorse of this comparison.

### 7.5 N-BEATS

N-BEATS is a deep learning model for univariate time series. It is a pure deep
architecture: no hand-crafted features, no explicit seasonality. Its blocks
learn trend and seasonality from the data through a basis expansion.

Strengths: strong on univariate series, and it removes the need for manual
feature engineering. Weaknesses: data-hungry, expensive to train, and a black
box. In this project it runs a deliberately small budget (two epochs), so its
result understates what a fully trained N-BEATS can do. Use it when you have a
lot of data and want to avoid feature engineering.

### 7.6 TFT

The Temporal Fusion Transformer is a deep model built for forecasting with
covariates. It combines an LSTM for local temporal patterns with a transformer
for long-range dependencies. Its attention mechanism is interpretable: you can
see which past steps and which covariates the model attends to.

Strengths: uses static, past, and future covariates, and its attention gives
interpretability. Weaknesses: complex, expensive, and data-hungry. Here it also
runs a small two-epoch budget. Use it when you have rich covariates and want to
explain the forecast.

### 7.7 Chronos-2

Chronos-2 is a pretrained foundation model from Amazon. It treats a time series
as a sequence of tokens and is trained on many series from many domains. It
forecasts zero-shot: no training on the target data. It returns native
quantiles.

Strengths: no training, and it transfers knowledge across domains. This project
shows it leading the hourly regime without ever seeing Nordic data. Weaknesses:
very expensive at inference (thousands of seconds), a black box, and its
pretraining is mostly hourly, so it fits the 15-minute regime less well. Use it
when you have no or little training data, or as a strong zero-shot baseline.

### 7.8 TimesFM

TimesFM is a pretrained foundation model from Google. It is univariate and
decoder-only. It also forecasts zero-shot and returns native quantiles.

Strengths: no training, fast enough inference, strong on univariate series.
Weaknesses: no covariates, large, and a black box. Use it for a univariate
zero-shot forecast.

## 8. Results and their meaning

### 8.1 The headline

Mean across the three hourly folds and one 15-minute fold, lower is better on
every accuracy column:

| model | family | CRPS | MAE | pinball P50 | skill | compute (s) |
|---|---|---|---|---|---|---|
| chronos2 | foundation | 9.61 | 15.45 | 7.73 | 0.514 | 3061.6 |
| catboost | gbdt | 9.64 | 15.50 | 7.75 | 0.411 | 1.8 |
| lgbm | gbdt | 9.72 | 15.37 | 7.68 | 0.453 | 0.5 |
| xgboost | gbdt | 10.63 | 15.70 | 7.85 | 0.317 | 1.8 |
| nbeats | deep | 14.39 | 18.40 | 9.20 | 0.256 | 38.6 |
| timesfm | foundation | 14.72 | 21.33 | 10.66 | 0.434 | 17.5 |
| sarima | classical | 14.99 | 21.69 | 10.85 | −0.114 | 166.6 |
| tft | deep | 20.12 | 30.20 | 15.10 | −0.707 | 640.6 |
| lear | ml | 26.29 | 26.29 | 13.15 | −0.283 | 2.5 |
| ets | classical | 33.40 | 52.81 | 26.41 | −1.331 | 147.8 |

The trees win because they fit a separate quantile regressor for each of P10,
P50, and P90 on the full feature matrix. They model the distribution directly
and use every exogenous group without leaking the future. Chronos-2 leads
without any training on Nordic data, which is the concrete demonstration that a
pretrained backbone transfers to a new market. The deep models trail because
they run a two-epoch budget, so the comparison understates a longer training
run. LEAR collapses its quantiles to one point, so it carries no spread. ETS is
a single smoother that cannot represent the regime structure.

Four models score below the seasonal-naive baseline. They are worse than
repeating last week's price.

### 8.2 The two regimes

The ranking differs sharply between regimes:

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

Two forces explain this. First, the foundation models are pretrained mostly on
hourly data, so their inductive bias fits the hourly structure better than a
96-slot day. Second, the 15-minute regime has one fold. Its column is one
day-ahead forecast and carries far more variance than the hourly column. A
single day's ranking is suggestive, not conclusive.

### 8.3 The Pareto frontier

The frontier flags models not dominated on both accuracy and compute. The
gradient-boosted trees sit on it: they reach CRPS within 0.1 to 1.0 of Chronos-2
at roughly 0.5 to 1.8 seconds of compute, against Chronos-2's 3062 seconds.
Chronos-2 is the accuracy anchor only if inference cost is ignored. This is the
core trade-off of the project: the trees give near-frontier accuracy at a
thousandth of the cost.

### 8.4 Statistical significance

The headline gap between Chronos-2 and CatBoost is 0.5% of CRPS. Is that gap
real or noise? A Diebold-Mariano test answers this. It compares the per-fold
loss of two models and asks whether the difference is larger than the noise.

Pairwise two-sided tests, on the per-fold mean hourly CRPS (three folds), a
negative statistic favouring the first model:

| model A | model B | DM statistic | p-value |
|---|---|---|---|
| chronos2 | catboost | −0.87 | 0.385 |
| chronos2 | lgbm | −0.97 | 0.332 |
| chronos2 | xgboost | −2.63 | 0.009 |
| catboost | lgbm | 0.73 | 0.466 |
| catboost | xgboost | −0.81 | 0.418 |
| lgbm | xgboost | −2.52 | 0.012 |

Two differences reach significance at 5%: XGBoost is significantly worse than
both Chronos-2 and LightGBM. The headline gap between Chronos-2 and CatBoost is
not significant, and neither is the gap between Chronos-2 and LightGBM. With
three folds the test has low power, so a non-significant result means the
evidence is insufficient, not that the models are equal. The cross-zone result
below is the stronger signal for Chronos-2's accuracy edge.

### 8.5 Feature selection

Forward selection adds the feature group that most reduces CRPS, then repeats
until no group helps. Cross-border flow is added first and cuts CRPS by 1.68.
Dropping it from the full set costs 1.16. Weather and hydro add less, and
carbon does not help.

The dominance of cross-border flow is expected for a bidding zone whose price
is set at the margin by imports and exports. The net position and the scheduled
exchanges carry the price pressure a purely local history misses. Carbon does
not help because the EUA price is a slow, market-wide series with little
intra-day variation for a system that is mostly hydro, wind, and nuclear. It
adds no signal beyond what the autoregressive base already captures.

The efficient set is the base group plus cross-border. A transfer check re-runs
each full-features arm on that set. The CRPS delta is negative for XGBoost,
TFT, CatBoost, and Chronos-2, so the set earns its cost everywhere.

### 8.6 Tuning

A secondary pass tunes the best model per family against its pinned default.
None of the tuned configurations beat the default on the hourly regime. The
deltas are +0.28 for SARIMA, +1.08 for CatBoost, +26.40 for N-BEATS, and +0.68
for Chronos-2.

The N-BEATS search overfit the single validation day. That is the clearest
lesson of the tuning pass: with one validation day, a search can pick a
configuration that is good for that day and bad elsewhere. The pinned default,
chosen once and never tuned to the folds, is more honest.

### 8.7 Cross-zone robustness

The headline runs on SE3. A reduced run takes the frontier (the three trees)
plus Chronos-2 across SE1 through SE4. Chronos-2 leads every zone, with mean
CRPS of 6.23, 6.00, 6.46, and 7.14. The three trees hold ranks two through four
in each zone. Both the accuracy anchor and the frontier generalise beyond SE3.

## 9. Challenges and limitations

Four limitations bound these conclusions.

**Fold count.** Three hourly folds is a small sample. It limits the power of
the significance test and means the 15-minute regime is a single day. The
results are directional, not final.

**The deep-model budget.** N-BEATS and TFT run a two-epoch budget sized for
smoke tests. Their results are a lower bound on what a longer training run
would reach. This is a deliberate choice to keep the comparison runnable, not a
claim about deep learning.

**The foundation-model cost.** Chronos-2 needs thousands of seconds of
inference. That cost is real in production. A model that is slightly more
accurate but a thousand times more expensive is not automatically better.

**Regime imbalance.** The pooled headline weights the hourly regime three to
one over the 15-minute regime. The "Chronos-2 leads" claim is driven by the
hourly regime.

## 10. Design decisions

These are the decisions that shaped the project, with the reason for each. The
full record is in `docs/adr/`.

**Standalone benchmark, not a feeder.** The project was first scoped as the
forecast input for another project (P16). It was recast as a standalone,
recruiter-facing benchmark. The reason: a benchmark that compares ten models on
a defensible methodology is more useful on its own than a narrowly scoped
feeder. This supersedes ADR-0001.

**Ten models across four families.** The roster spans classical, gradient-boosted,
deep, and pretrained models. The reason: to measure where each approach earns
its place, not to rank variants of one method. This supersedes ADR-0005.

**darts as the host library.** The classical, deep, and foundation arms run on
darts. The reason: darts covers the full spectrum behind one interface, so ten
models share one `fit`/`predict_quantiles` contract. This supersedes ADR-0006.

**CRPS as the primary metric.** The reason: it is a proper scoring rule for a
distribution, so it cannot be gamed and it rewards both calibration and
sharpness. Pinball and MAE are reported alongside because they answer different
questions.

**Yearly walk-forward with a purge.** The reason: a walk-forward is the only
setup that reproduces production use, yearly refit balances data against folds,
and the purge keeps structural breaks out of the training window.

**Exclude day-ahead load and wind.** The reason: leakage. Those values are not
known when the price forecast is made.

**The disk cache.** The reason: the fetch is slow and flaky, and a re-run should
not pay for it again. The cache is an optimisation, never a dependency.

## 11. What this teaches

A few ideas recur throughout this project and are worth stating plainly, because
they are the ideas that separate a working forecast from a broken one.

Leakage is the first thing to check. A model that sees the future looks great
and fails in production. The as-of rule, the walk-forward split, and the
exclusion of day-ahead load all exist to prevent it.

A proper scoring rule is not optional for probabilistic work. A metric that a
model can game gives a false ranking. CRPS is proper; a naive accuracy measure
is not.

A baseline is the floor. Four of ten models lose to a rule that repeats last
week. Every model should be measured against the trivial rule before it is
taken seriously.

Statistical significance matters when the gaps are small. A 0.5% difference in
CRPS may be noise. A significance test, even underpowered, forces the claim to
be defended.

The accuracy-compute trade-off is a first-class result, not an afterthought. A
model that is slightly better and far more expensive is a different answer than
a model that is slightly worse and far cheaper. The Pareto frontier makes that
trade-off explicit.

Overfitting appears in disguise. A tuning search that wins one validation day
and loses everywhere else is overfitting, even when the word is not used. The
pinned default is more honest than a search tuned to a single day.
