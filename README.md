# Nordic electricity price forecasting

A ten-model comparison for day-ahead Nord Pool electricity prices, ranked on
a Pareto accuracy-versus-compute basis. The repository assembles the data,
engineers a leakage-free feature matrix, runs an expanding-window backtest
across the 60-minute and 15-minute market regimes, and reports which models
sit on the accuracy-per-compute frontier.

## Result

Mean CRPS across the hourly (2023-01-01 to 2025-09-30, three folds) and
15-minute (2025-10-01 onward, one fold) regimes, lower is better:

| model | family | mean CRPS | total compute |
|---|---|---|---|
| chronos2 | foundation | 9.61 | 3061.6 s |
| catboost | gbdt | 9.64 | 1.8 s |
| lgbm | gbdt | 9.72 | 0.5 s |
| xgboost | gbdt | 10.63 | 1.8 s |
| nbeats | deep | 14.39 | 38.6 s |
| timesfm | foundation | 14.72 | 17.5 s |
| sarima | classical | 14.99 | 166.6 s |
| tft | deep | 20.12 | 640.6 s |
| lear | ml | 26.29 | 2.5 s |
| ets | classical | 33.40 | 147.8 s |

The gradient-boosted trees (CatBoost, LightGBM, XGBoost) dominate the
accuracy-per-compute frontier. Chronos-2 edges the trees on mean CRPS by about
0.5% but costs roughly three orders of magnitude more compute (3061.6 s versus
0.5 to 1.8 s). The deep models (N-BEATS, TFT) and the classical models
(SARIMA, ETS) are dominated on both axes. LEAR, the regularized linear
baseline, is fast but the least accurate of the price-driven models.

The frontier flag is computed within each regime, not pooled, so a model that
is cheap and accurate in one regime does not dominate a model in the other.

## Tuning

A secondary comparison tunes the best model per family and reports the delta
against the pinned default; the default table above stays the headline. SARIMA
is tuned by an AIC/BIC order grid, CatBoost and N-BEATS by bounded Optuna, and
Chronos-2 by a context-length sweep (its weights are pretrained and are never
tuned). None of the tuned configs beat the pinned default on the hourly
regime: the deltas are +0.28 (SARIMA), +1.08 (CatBoost), +26.40 (N-BEATS), and
+0.68 (Chronos-2). The N-BEATS search overfit the single validation day, which
is the clearest case for keeping the bounded default as the headline. The
tuned table lives in the snapshot under `tuned_table.csv`.

## Cross-zone robustness

The headline runs on SE3. A reduced run takes the frontier (the three trees)
plus Chronos-2 across SE1 through SE4 at default configs, hourly regime only,
to check whether the frontier generalises and whether Chronos-2's accuracy
edge survives outside SE3. Chronos-2 leads every zone (mean CRPS 6.23, 6.00,
6.46, and 7.14 for SE1 through SE4), and the three trees hold ranks 2 through
4 in each zone, so both the accuracy anchor and the frontier generalise. The
per-zone ranking is in `cross_zone_summary.csv`.

## Feature selection

Seven feature groups feed the full-features models. Group 1 (autoregressive
lags, rolling statistics, calendar features, regime label) is the
always-present base. Groups 3 through 7 are cross-border, weather, hydro,
commodities, and foreign exchange. Group 2 (day-ahead load and wind forecasts)
is excluded because those values are known only after the price is set.

Forward selection finds cross-border flow to be the dominant exogenous group:
it is added first and cuts CRPS by 1.68, and dropping it from the full set
loses 1.16 CRPS. Weather and hydro add less; the carbon price does not help.
The efficient set (within 1% of the full-features CRPS) is the base group plus
cross-border. A transfer check re-runs each full-features arm on that set and
confirms it earns its cost everywhere: the CRPS delta is negative for XGBoost
(-1.56), TFT (-0.89), CatBoost (-0.65), and Chronos-2 (-0.18).

## Methodology

- **Data**: ENTSO-E (cross-border, hydro), Open-Meteo (weather), ECB (EUR/SEK),
  EEX (EUA carbon). Joined at a single `assemble_data` seam backed by a
  Parquet cache keyed on `(source, start, end, params)`.
- **Features**: every covariate follows an as-of timing rule, so no feature
  leaks future information. Boundary masking zeroes lags and rolling windows
  that cross a regime switch; calendar features are cyclical.
- **Regimes**: the market moved from 60-minute to 15-minute settlement on
  2025-10-01, with an earlier boundary detected on 2024-11-04. Each regime is
  assembled and backtested as its own single-frequency window.
- **Backtest**: expanding walk-forward, yearly refit, 7-day purge. Primary
  metric is mean CRPS from the P10/P50/P90 quantiles, with pinball loss and
  MAE reported alongside and a seasonal-naive skill score.
- **Roster**: four families (classical, gradient-boosted, deep, foundation),
  ten models, every arm pinned to seed 42.

See the ADRs in `docs/adr/` for the full decision record.

## Replication

The snapshot holds every matrix and result table needed to reproduce the
figures offline, with no API key.

```bash
uv sync
uv run python -c "from forecast_pipeline.snapshot import load_results, headline_ranking; \
print(headline_ranking(load_results()['results']).to_string(index=False))"
```

To rebuild the snapshot from the cached data window (no API key):

```bash
uv run python -m forecast_pipeline.snapshot --build
```

To run the full comparison from scratch, an ENTSO-E API key is required in
`.env`:

```bash
set -a && source .env && set +a
uv run python -m forecast_pipeline.bakeoff --zones SE3 --start 2021-01-01 --end 2026-08-13
```

## Full narrative

- `notebooks/01_data_features_regimes.ipynb`: data pipeline, features, regime
  detection, feature selection.
- `notebooks/02_backtest_comparison_pareto.ipynb`: model comparison, tuning,
  cross-zone, Pareto frontier.
- `app.py`: a single-page Streamlit explorer (`uv run streamlit run app.py`).
