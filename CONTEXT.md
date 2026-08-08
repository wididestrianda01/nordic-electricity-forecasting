# P14 — Nordic Electricity Forecasting

Day-ahead price forecasting for Swedish bidding zones (SE1–SE4), built as a portfolio piece for energy-quant and power-trading recruiters, and as the forecast-input layer for P16 (BESS Revenue Optimizer).

## Language

**P14 (this project)**:
The forecast-layer scope defined by the Portfolio Master Index — a ~25-30 hour build producing day-ahead price forecasts and regime diagnostics, sized to feed P16 rather than stand alone as a flagship deliverable.
_Avoid_: "the reference doc scope" (the 60-80h flagship version in `context/P14_EnergyForecasting_Reference.md` — superseded by the Master Index framing)

**MTU (Market Time Unit)**:
The Nordic day-ahead auction's clearing interval. 15 minutes (96 MTUs/day) since 1 Oct 2025; hourly (24/day) before that date.
_Avoid_: "hourly price" as a project-wide default — only correct for the pre-Oct-2025 period

**Regime**:
A structurally distinct period in the price-formation process, driven by a market-design change rather than ordinary demand/supply variation. Confirmed regime boundaries: flow-based market coupling (4 Nov 2024) and the MTU switch to 15-minute (1 Oct 2025).
_Avoid_: "outlier" or "anomaly" for these periods — they are permanent structural shifts, not transient noise

**P14→P16 handoff**:
The interface P14 exposes to P16 (BESS Revenue Optimizer): a quantile forecast grid (P10/P50/P90) per MTU, plus a regime label per period. Not a point forecast — P16's dispatch decisions need the distribution and the regime context to size risk correctly.

**LEAR**:
LASSO-Estimated AutoRegressive model — the baseline forecaster (Lago et al.). Not the regime-conditional model; see below.

**Regime-conditional model**:
The primary forecaster: an HMM assigns a regime label per period, which conditions a LightGBM quantile-regression model producing the P14→P16 handoff's quantile grid. This is the project's headline model, distinct from the LEAR baseline.

**Spark spread (as used in P14)**:
The gas-to-power price differential, used here as a diagnostic for cross-border transmission pressure through the SE4↔DE/DK/PL interconnectors — not as a claim that gas sets the marginal price in Sweden's hydro/wind/nuclear-dominated mix.
_Avoid_: framing spark spread as a direct price driver for SE1-SE3
