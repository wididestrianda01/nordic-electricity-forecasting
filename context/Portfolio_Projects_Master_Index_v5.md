# Portfolio Projects — Master Index (Revised v6)

> Last updated: 2026-08-04
> Revision v6: **Equinor track added.** New concrete target — Equinor (Stavanger) Market Analysis & Trading Graduate Programme 2027, applications opening 7 August 2026. New project **P17 — Energy Cross-Commodity Trading Analytics Platform** added as the Equinor-specific deliverable: multi-commodity data pipeline (Brent crude, TTF natural gas, EUA carbon, German/Nordic power) with spark/dark/crack spread economics, DCC-GARCH correlation modeling, t-copula tail dependence, multi-commodity VaR/ES engine (extending P1), stress scenario engine (gas crisis, recession, energy transition), and Streamlit trading dashboard. ~40 hrs, deliverable within one week. P17 closes the crude oil and natural gas gap in the portfolio — P13/P14/P16 cover Nordic power/BESS; P17 covers the hydrocarbon and cross-commodity desk that is Equinor's core MMP business. P17 also extends P0's Black-Scholes engine into commodity options territory (Black-76 on energy futures), making the options knowledge transferable to energy derivatives desks. Build strategy: data pipeline → spread economics → correlation engine → VaR → scenario tool → dashboard. P16 (BESS) remains the Ingrid flagship; P17 is the Equinor play.

> Revision v5 (retained): Energy quant track elevated — Ingrid Quantitative Power Trader, P16 BESS Optimizer added.
> Revision v4 (retained): SQL competency promoted from a token keyword to a defensible, interview-grade skill. The thin DuckDB patch on P2 is deepened into a full multi-table SQL feature pipeline (joins, GROUP BY aggregations, window functions over the 7-table Home Credit schema). SQL data/reporting layers added to P6 (EL aggregation + regulatory backtest reconciliation), P8 (attribution decomposition as relational GROUP BY), and P13/P14 (ENTSO-E area-price fact tables). New standalone project **P15 — Financial-Data SQL Analytics Layer** added as a dedicated SQL showcase (star schema + documented query library with window functions, CTEs, drawdown and VaR-breach logic) so SQL is independently visible in a repo rather than buried in module code. Skills-line phrasing upgraded.
>
> Revision v3 (retained): Dual-track restructure — SEB IRB primary track, Vattenfall BA Markets secondary track. P6 (LGD/EAD) moved forward to October as the highest-priority new build. P13 (Nordic Electricity VaR/ES) promoted from optional to scheduled for Vattenfall thesis outreach. P0 (Black-Scholes Options Pricing Notebook) added as a lightweight pre-CFA deliverable to unlock SEB Front Office Model Validation and RVS thesis angles, grounded in KTH DiVA evidence that SEB co-supervises derivatives/options thesis projects. Hull-White extension added within P3 scope. SQL layer added to P2. Energy projects P13/P14 reclassified from optional to conditional-active for Vattenfall track.

---
## Track Architecture


This portfolio serves four targets, with the energy quant track now split into a power/BESS stream (Ingrid) and a cross-commodity stream (Equinor).

| Track | Primary Target | Secondary Target | Key Projects |
|---|---|---|---|
| **Energy quant — power/BESS** | Ingrid (Stockholm) — Quantitative Power Trader | Vattenfall BA Markets (Solna) · Statkraft · Modity | P1, P13, P14, **P16** |
| **Energy quant — cross-commodity** | Equinor (Stavanger) — Market Analysis & Trading Graduate | Danske Commodities · Centrica Energy Trading · Gunvor | P0, P1, P15, **P17** |
| **Bank IRB (primary thesis)** | SEB NRCRM · SEB RVS · SEB Front Office Model Validation | Handelsbanken · Nordea · Swedbank | P0, P2+SQL, P6, P10, P15 |
| **Asset management / AP funds** | AP3/AP4 · Alecta · AMF | Swedbank Robur · SEB IM | P3+HW, P4, P5, P7, P8, P12 |
> **Decision rule (updated):** P17 (Cross-Commodity Trading Analytics) is the #1 priority this week — deliver before Equinor graduate applications open (7 Aug 2026). The 40-hour build sprint runs Aug 4–10: data pipeline → spread economics → correlation engine → VaR → scenario tool → dashboard, all reusing P0 (options) and P1 (VaR/ES) heavily. P16 (BESS Optimizer) remains the Ingrid flagship and resumes after P17 ships. If Equinor application yields an interview, P17 serves as the primary talking point; if not, P17 still fills the crude/gas gap and strengthens any energy trading application (Centrica, Danske Commodities, Gunvor). P13 and P14 hours fold into P16 reinforcement as originally planned.

---
## Quick Reference Table

| ID      | Project                                           | Domain                           | Status                         | Timeline                    | Builds On                      |
| ------- | ------------------------------------------------- | -------------------------------- | ------------------------------ | --------------------------- | ------------------------------ |
| **P0**  | Black-Scholes Options Pricing Notebook            | Derivatives / Market Risk        | ✅ Completed                    | Aug 2026 (pre-CFA, ~12 hrs) | Base                           |
| **P1**  | VaR & Expected Shortfall Engine                   | Market Risk                      | ✅ Completed                    | May 2026                    | Base                           |
| **P2**  | Credit Risk Scoring Pipeline + SQL Feature Layer  | Credit Risk / ML                 | ✅ + SQL deepened               | Jun 2026 + Aug patch        | Base                           |
| **P3**  | Fixed Income & Yield Curve Engine + Hull-White    | Fixed Income / Rates             | ✅ Completed                    | Sep 2026 (W3–5)             | Base                           |
| **P4**  | Factor Model & Portfolio Optimisation             | Asset Management                 | Prepared                       | Sep 2026 (W1–5)             | —                              |
| **P5**  | Mean-CVaR Portfolio Optimization                  | Asset Mgmt / Risk                | Planned                        | Sep 2026 (W1–3)             | P1, P4                         |
| **P6**  | LGD & EAD Modelling (IRB)                         | Credit Risk — IRB                | ✅ Completed                    | Oct 2026 (W1–4)             | P2                             |
| **P7**  | ESG-Integrated Portfolio Optimization             | Asset Mgmt / Sustainable Finance | NEW                            | Oct 2026 (W4–6)             | P4, P5                         |
| **P8**  | Brinson Performance & Risk Attribution            | Asset Management                 | Planned                        | Oct 2026 (W5–8)             | P4                             |
| **P9**  | HMM Regime Detection                              | Market Risk / ML                 | Planned                        | Nov–Dec 2026                | P1                             |
| **P10** | IFRS 9 / Expected Credit Loss Pipeline            | Credit Risk / Regulatory         | NEW                            | Nov–Dec 2026                | P2, P6                         |
| **P11** | ML Return Forecasting + Transaction-Cost Backtest | Buy-Side / ML                    | NEW                            | Jan–Feb 2027                | P4, P9                         |
| **P12** | ALM / Liability-Driven Investment Simulation      | Pension / ALM                    | NEW                            | Feb–Mar 2027                | P1, P3, P4                     |
| **P13** | Nordic Electricity VaR & ES Engine                | Energy Risk                      | **SCHEDULED (supports P16)**   | Sep 2026 (W1–2)             | P1                             |
| **P14** | Nordic Electricity Price Forecasting              | Energy / Forecasting             | Active-optional (supports P16) | Post-thesis                 | P1                             |
| **P15** | Financial-Data SQL Analytics Layer                | Data Engineering / SQL           | **Prepared**                   | Jun–Aug 2026 (~6–8 hrs)     | P1                             |
| **P16** | Nordic BESS Revenue Optimizer & Intraday Trader   | Energy Trading / Optimization    | **PRIORITY ↑↑ — Aug–Sep 2026** | Aug–Sep 2026 (~60–80 hrs)   | P1, P14 (forecast), P13 (risk) |
| **P17** | Energy Cross-Commodity Trading Analytics          | Energy Trading / Cross-Commodity | ✅ Completed                    | Aug 4–10, 2026              | P0, P1, P15                    |

---
## Major Phase of the Project

- Phase 1: Build the model, excluding presentation (jupyter notebook, report), using synthetic data
- Phase 2: Build the model, using the real data, including full data pipeline.
- Phase 2*: Improve the model, with configuration management such as Hydra + MLFlow, to track all trial. Choose the best model.
- Phase 3: Build project presentation: jupyter notebooks, report
- Phase 4: Deploy streamlit app
- Phase 5: Project Repo finalization: cleanup, verification, clean dead code, unused function, final review of objective, README, etc.


---
## P0 — Black-Scholes Options Pricing Notebook ✅

| | |
|---|---|
| **Domain** | Derivatives / Market Risk |
| **Status** | ✅ Completed — Aug 2026 |
| **Timeline** | Completed (pre-CFA, ~12 hrs) |

**Why this exists.** KTH DiVA records SEB-hosted theses on derivatives pricing (DeepONet/PONet for geometric Asian and FX options, OMXS30 dispersion trading using BS straddles). The SEB RVS junior quant role explicitly lists derivatives as an instrument class alongside fixed-income, equities, and FX. Absent any options pricing work, the portfolio has a gap that appears immediately to anyone at SEB Front Office Model Validation or RVS. This notebook closes it at minimal time cost — no stochastic calculus from scratch, just clean Python implementation with documented derivation.

**Scope of work.**
- Black-Scholes closed-form pricer: European call/put, put-call parity verification.
- Greeks: Delta, Gamma, Vega, Theta, Rho — computed analytically and by finite difference for cross-check.
- Implied volatility solver via Brent's method (or Newton-Raphson with Vega guard).
- Volatility smile: plot IV surface across strikes and maturities using real OMXS30 option data or synthetic input.
- Delta-hedging simulation: discrete rebalancing P&L vs theoretical BS price — shows hedging error as a function of rebalancing frequency.
- Binomial tree pricer as a conceptual cross-check (CRR model, convergence to BS as steps increase).

**Deliverables.** Clean Jupyter notebook (`options_engine/bs_notebook.ipynb`) · `options_engine/pricer.py` (BS + Greeks + IV solver) · `options_engine/binomial.py` (CRR tree) · `options_engine/smile.py` (IV surface plotting) · GitHub repo · brief README documenting the derivation assumptions.

**Key concepts demonstrated.** Black-Scholes formula and assumptions, Greeks (Delta, Gamma, Vega, Theta, Rho), put-call parity, implied volatility, volatility smile, delta-hedging mechanics, binomial tree convergence — the full entry-level derivatives interview toolkit, applicable directly to SEB model validation screening.

**Target companies.** SEB Front Office Model Validation · SEB Risk & Valuations Services (RVS) · SEB (dispersion/volatility thesis angle) · Nordea Markets · Oliver Wyman.

---
## P1 — VaR & Expected Shortfall Engine ✅

| | |
| **Domain** | Market Risk |
| **Status** | ✅ Completed — May 2026 |
| **Methods** | Historical / Parametric / Monte Carlo VaR + ES · Kupiec/Christoffersen backtesting · GARCH(1,1) · EGARCH · COVID-2020 stress |
| **Stack** | `var_engine/garch.py`, `var_engine/var_methods.py`, `var_engine/backtest.py` |
| **Reused by** | P5, P9, P12, P13, P14, P16, P17 |

---
## P2 — Credit Risk Scoring Pipeline ✅ + SQL feature layer

| | |
|---|---|
| **Domain** | Credit Risk / Machine Learning |
| **Status** | ✅ Completed Jun 2026 · SQL feature layer pending (~5 hrs, Jun–Aug) |
| **Methods** | Multi-table feature engineering · XGBoost/LightGBM/CatBoost · SHAP · EU AI Act fairness · Basel III IRB OOT validation |
| **Stack** | `credit_engine/features.py`, `credit_engine/model.py`, `credit_engine/explain.py`, `app/api.py` |
| **SQL feature layer** | The Home Credit dataset is 7 joined tables — the best SQL showcase in the portfolio. Move the multi-table feature engineering *into* SQL: joins across application/bureau/previous-loan tables, GROUP BY aggregations (count of prior defaults, avg days-past-due, credit-utilisation ratios), and window functions for time-since-last-event features. Converts the SQL claim from a `run_query()` token into "built a 155-feature pipeline in SQL with joins, GROUP BY aggregations, and window functions over a 7-table relational schema." ~5 hrs. |
| **Reused by** | P6, P10, P15 |

---
## P3 — Fixed Income & Yield Curve Engine + Hull-White Extension ✅

| | |
|---|---|
| **Domain** | Fixed Income / Rates |
| **Status** | Planned — Sep 2026, Weeks 3–5 |
| **Timeline** | ~28–33 hrs (base) + ~3 hrs (Hull-White extension) |

**Project goals.** Build a practical fixed income toolkit covering bond pricing, yield curve construction, interest rate risk measurement, and stochastic rate-path simulation. The `rates_engine/` package feeds directly into P12 (ALM) for liability discounting. The Hull-White extension replaces the synthetic rate-path shortcut in P12 with a proper one-factor short-rate model, which is the minimum required for an AP fund or pension-house thesis topic involving rate-path simulation.

**Scope of work.**
- Bootstrap a par-coupon yield curve using swap/government bond data (Nelson-Siegel parametric form as fallback).
- Price fixed-rate and floating-rate bonds: dirty/clean price, accrued interest, yield-to-maturity.
- Compute DV01, modified duration, convexity, and key-rate durations (KRDs) at standard tenor buckets.
- Parallel and non-parallel rate shock scenarios (±25bp, ±50bp, ±100bp); P&L attribution by duration bucket.
- Build a small bond portfolio tracker: aggregate DV01, portfolio KRD profile, and funding spread analysis.
- Use QuantLib-Python for pricing validation (cross-check against manual implementations).
- **Hull-White extension:** one-factor HW model calibrated to the bootstrapped yield curve; simulate short-rate paths under risk-neutral measure; produce term-structure scenarios for use in P12 ALM simulation.

**Deliverables.** `rates_engine/` Python package · Streamlit bond analyser and portfolio tracker · rate-shock scenario report (LaTeX) · HW simulation module (`rates_engine/hw_model.py`) · GitHub repo with unit tests.

**Key concepts demonstrated.** Yield curve bootstrapping (Nelson-Siegel), bond pricing, YTM, DV01, modified duration, convexity, key-rate duration, rate-shock P&L attribution, QuantLib integration, Hull-White one-factor short-rate model, mean reversion, stochastic rate-path simulation.

**Target companies.** AP2/AP3/AP4 · AMF · Alecta · Folksam · SEB · Handelsbanken · Nordea · Kidbrooke · zeb · Skandia · SPP/Storebrand.

---
## P4 — Factor Model & Portfolio Optimisation

| | |
|---|---|
| **Domain** | Asset Management / Quantitative Finance |
| **Status** | Planned — Sep 2026 |
| **Timeline** | Sep 2026, Weeks 1–5, ~35–45 hrs |

**Project goals.** Build a reusable factor-model and portfolio-construction engine that produces optimal Nordic-equity portfolios from estimated factor exposures and a robust covariance matrix, validated by walk-forward backtesting. Foundation for P5, P7, and P8.

**Scope of work.**
- Build the Fama-French 5-factor + momentum model; estimate factor loadings per asset.
- Estimate the covariance matrix with Ledoit-Wolf shrinkage (vs sample covariance baseline).
- Implement three allocators: mean-variance, risk-parity, Black-Litterman.
- Walk-forward backtest (rolling estimation/holding windows) with turnover and Sharpe/Sortino/max-drawdown reporting.
- Streamlit explorer to inspect exposures, weights, and backtest paths.

**Deliverables.** Streamlit portfolio explorer · walk-forward backtest report (PDF/LaTeX) · GitHub repo · `factor_engine/` package.

**Key concepts demonstrated.** Factor models, covariance shrinkage, convex portfolio optimization, Bayesian (Black-Litterman) views, out-of-sample backtesting discipline.

**Target companies.** AP2/AP3/AP4 · Swedbank Robur · SEB Investment Management · Lynx · Brummer · Coeli · Lannebo · Robeco.

---
## P5 — Mean-CVaR Portfolio Optimization

| | |
|---|---|
| **Domain** | Asset Management / Market Risk |
| **Status** | Planned — Sep 2026, Weeks 1–3 |
| **Timeline** | ~20–25 hrs |

**Project goals.** Replace variance with CVaR (Expected Shortfall) as the portfolio risk objective, connecting the completed market-risk work (P1) to portfolio construction. Demonstrates tail-risk optimization — the framing regulators (FRTB) and pension funds care about.

**Scope of work.**
- Implement Rockafellar-Uryasev (2000) CVaR linear-programming formulation.
- Build the CVaR efficient frontier; compare against the mean-variance frontier from P4.
- Confidence-level sensitivity analysis (95% / 97.5% / 99%).
- Reuse the ES estimation logic from P1 as the risk input.

**Deliverables.** Streamlit CVaR optimizer · LaTeX write-up · GitHub repo · `cvar_engine/` package.

**Key concepts demonstrated.** CVaR/ES as a coherent risk measure, Rockafellar-Uryasev LP, efficient-frontier comparison, FRTB risk framing.

**Target companies.** AP3 · AP4 · Lynx · Brummer · Alecta · AMF · Robeco.

---
## P6 — LGD & EAD Modelling (IRB) ✅

|              |                                              |
| ------------ | -------------------------------------------- |
| **Domain**   | Credit Risk — Advanced IRB                   |
| **Status**   | **PRIORITY ↑ — Oct 2026, Weeks 1–4**         |
| **Timeline** | ~55–65 hrs |

**Why this is the priority.** SEB NRCRM (Non-Retail Credit Risk Modelling) builds and maintains IRB models (PD, LGD, EAD) for regulatory capital under CRR/Basel III. KTH DiVA records a 2023 financial mathematics thesis specifically on PD estimation for low-default portfolios (EL = EAD × PD × LGD framing confirmed). P6 completes the IRB triangle started in P2, making the portfolio the most complete student demonstration of IRB methodology available. This is the single project most likely to generate a positive signal from SEB NRCRM thesis outreach.

**Scope of work.**
- Two-stage LGD model (probability of cure + loss-severity regression).
- CCF regression for EAD.
- Downturn LGD adjustment per regulatory guidance.
- EL calculator combining the P2 PD output with new LGD/EAD.
- Backtesting against realized losses.
| **Deliverables.** LGD + EAD models · EL calculator app · backtesting report · GitHub repo · LaTeX report mapping each component to CRR articles.

| **Key concepts demonstrated.** Full IRB framework (CRR Art. 161–162, 166), downturn adjustment, two-stage modelling, regulatory backtesting — the complete PD + LGD + EAD = EL pipeline.

**Target companies.** SEB NRCRM · SEB (any credit risk team) · Nordea · Handelsbanken · Swedbank · Hoist Finance · zeb · Deloitte/KPMG risk modelling.

---
## P7 — ESG-Integrated Portfolio Optimization *(NEW)*

| | |
|---|---|
| **Domain** | Asset Management / Sustainable Finance |
| **Status** | Planned — Oct 2026, Weeks 4–6 |
| **Timeline** | ~10–15 hrs (extension of P4 + P5) |

**Project goals.** Add ESG constraints and tilts to the portfolio engines from P4/P5. An explicit requirement for AP funds and Swedbank Robur thesis topics given their statutory sustainability mandates.

**Scope of work.**
- Ingest ESG scores (and/or carbon-intensity data) per asset.
- Add ESG-constraint and ESG-tilt layers to the existing optimizers (hard floor on portfolio ESG score; carbon-budget cap).
- Quantify the risk/return cost of ESG constraints vs the unconstrained frontier.
- Optional: Scope-1/2 carbon-intensity reduction target with tracking-error budget.

**Deliverables.** ESG-constrained optimizer (extends `factor_engine/` + `cvar_engine/`) · comparison report (constrained vs unconstrained frontier) · GitHub update.

**Key concepts demonstrated.** SFDR/sustainable-finance integration, constrained optimization, ESG-vs-return trade-off analysis, carbon budgeting.

**Target companies.** AP2/AP3/AP4 · Folksam · Swedbank Robur · E. Öhman · Robeco · Alecta.

---
## P8 — Brinson Performance & Risk Attribution

| | |
|---|---|
| **Domain** | Asset Management |
| **Status** | Planned — Oct 2026, Weeks 5–8 |
| **Timeline** | ~20–25 hrs |

**Project goals.** Decompose portfolio performance into allocation, selection, and interaction effects against a benchmark — the institutional attribution that AP funds and asset managers expect candidates to understand.

**Scope of work.**
- Brinson-Hood-Beebower and Brinson-Fachler attribution.
- Carino multi-period geometric linking.
- Walk-forward 5-year attribution vs MSCI Sweden benchmark.
- Factor-level attribution extension (tie back to P4 exposures).

**Deliverables.** Streamlit attribution dashboard · multi-period backtest report · GitHub repo · LaTeX write-up · `attribution_engine/`.

**Key concepts demonstrated.** BHB/Brinson-Fachler attribution, multi-period geometric linking, benchmark-relative performance analysis, GIPS alignment.

**Target companies.** AP3 · AP4 · Alecta · AMF · Swedbank Robur · SEB Investment Management.

---
## P9 — HMM Regime Detection

| | |
|---|---|
| **Domain** | Market Risk / Unsupervised ML |
| **Status** | Planned — Nov–Dec 2026 |
| **Timeline** | ~Weeks 14–26 of Year 2 |

**Project goals.** Detect market regimes (calm/crisis) with Hidden Markov Models and produce regime-conditional VaR, demonstrating ML on time series and a risk application that outperforms unconditional VaR.

**Scope of work.**
- Fit Gaussian HMM (forward-backward, Viterbi, Baum-Welch); GMM baseline.
- Map states to regimes; produce regime-conditional VaR.
- Backtest regime-conditional VaR vs unconditional VaR from P1.

**Deliverables.** Live regime-monitor Streamlit app · LaTeX report · GitHub repo · `hmm_engine/`.

**Key concepts demonstrated.** HMMs (forward-backward, Viterbi, Baum-Welch), unsupervised regime classification, regime-conditional risk, model comparison.

**Target companies.** SEB · Nordea · Brummer · Lynx · zeb.

---
## P10 — IFRS 9 / Expected Credit Loss Pipeline *(NEW)*

| | |
|---|---|
| **Domain** | Credit Risk / Regulatory |
| **Status** | Planned — Nov–Dec 2026 |
| **Timeline** | ~15–20 hrs (capstone over P2 + P6) |

**Project goals.** Combine the PD model (P2) and LGD/EAD models (P6) into an IFRS 9 expected-credit-loss engine with stage allocation and forward-looking macro scenarios. KTH DiVA records at least two financial mathematics theses directly on IFRS 9 / ECL methodology (one on macro-ECL via VAR approach, one on PD modelling under IFRS 9), confirming this is an active KTH–bank thesis topic.

**Scope of work.**
- Stage allocation (Stage 1 / 2 / 3) using significant-increase-in-credit-risk triggers.
- Lifetime-PD term structure for Stage 2/3 exposures.
- 12-month vs lifetime ECL computation.
- Forward-looking macroeconomic scenario overlay (baseline/adverse/severe) with probability weighting.

**Deliverables.** ECL engine (imports `credit_engine/` + `lgd_engine/`) · scenario-weighted ECL report · GitHub update · short LaTeX note.

**Key concepts demonstrated.** IFRS 9 three-stage model, SICR triggers, lifetime PD term structure, macro scenario weighting, point-in-time vs through-the-cycle.

**Target companies.** SEB · Swedbank · Handelsbanken · Nordea · Danske Bank · zeb · PwC · KPMG · Deloitte.

---
## P11 — ML Return Forecasting + Transaction-Cost Backtest *(NEW)*

| | |
|---|---|
| **Domain** | Buy-Side / Machine Learning |
| **Status** | Planned — Jan–Feb 2027 |
| **Timeline** | ~20 hrs |

**Project goals.** Build a realistic ML return-forecasting signal and backtest it net of transaction costs, mirroring how systematic funds evaluate alpha.

**Scope of work.**
- Engineer cross-sectional features (momentum, value, volatility, regime probability from P9).
- Train ML models (gradient boosting / regularized regression) to forecast next-period returns.
- Convert forecasts to portfolio weights; backtest with a transaction-cost and turnover model.
- Compare net-of-cost performance vs a naive momentum benchmark.

**Deliverables.** Forecasting + backtest framework · net-of-cost performance report · GitHub repo.

**Key concepts demonstrated.** Cross-sectional return prediction, feature engineering, transaction-cost modelling, turnover-aware backtesting, signal decay.

**Target companies.** Lynx · Brummer · Robeco · Tidan · Coeli.

---
## P12 — ALM / Liability-Driven Investment Simulation *(NEW)*

| | |
|---|---|
| **Domain** | Pension / Asset-Liability Management |
| **Status** | Planned — Feb–Mar 2027 |
| **Timeline** | ~25–30 hrs (layered on P1 + P3 + P4) |

**Project goals.** Simulate a pension fund's assets and liabilities jointly, track the funding ratio under stochastic scenarios, and test LDI hedging strategies. The `rates_engine/hw_model.py` from P3 replaces any synthetic rate-path shortcut — liability cash flows are properly discounted using HW-simulated rate paths and QuantLib bond math.

**Scope of work.**
- Project liability cash flows and discount them under HW stochastic rate paths using the P3 yield curve engine.
- Apply QuantLib-Python for swap valuation and duration computation of the liability portfolio.
- Simulate asset returns (reuse P1 Monte Carlo + P4 portfolios).
- Track funding-ratio dynamics across scenarios.
- Test an LDI overlay (duration-matching hedge) and report funding-ratio-at-risk reduction.

**Deliverables.** ALM simulation engine · funding-ratio scenario report · LDI strategy comparison · GitHub repo · LaTeX write-up.

**Key concepts demonstrated.** Asset-liability matching, funding-ratio dynamics, liability discounting with a proper term structure, LDI / duration hedging, scenario analysis, QuantLib swap valuation.

**Target companies.** Kidbrooke · Alecta · AMF · AP2/AP3/AP4 · Folksam · Skandia · SPP/Storebrand.

---
## P13 — Nordic Electricity VaR & ES Engine *(Supports P16)*

| | |
|---|---|
| **Domain** | Energy Risk |
| **Status** | **Scheduled — Sep 2026, Weeks 1–2 (supports P16 risk overlay)** |
| **Timeline** | ~15 hrs (reuses P1 var_engine/ entirely) |

**Vattenfall context.** Vattenfall BA Markets (Solna) confirms English-only requirement on all intern and thesis postings. The Compliance & Market Analysis internship (summer 2026) requires a quantitative degree, Python, and strong English — no Swedish. The Diversity Challenge programme at Vattenfall explicitly targets academics with multicultural experience. The careers.vattenfall.com/thesis-project page is active. The International Trainee Programme is paused until 2027 (no 2026 intake) — thesis and internship are the primary routes. A KTH DiVA thesis records a Vattenfall/Forsmark-hosted project on stochastic portfolio optimization using CVaR, confirming Vattenfall hosts KTH financial mathematics students. Security clearance applies to some but not all positions.

**Project goals.** Apply the P1 VaR/ES framework to Nordic electricity prices, adding area-price dependence and 2025 negative-price stress events. This is the entry-level energy-quant credential. Now also serves as the risk measurement layer for P16 (BESS trading strategy risk limits).

**Scope of work.**
- Historical/Parametric/Monte Carlo VaR + ES on SE1–SE4 prices.
- Copula-based area dependence.
- GARCH volatility.
- 2025 negative-price stress scenario.
- **P16 integration:** Export VaR/ES computations as a risk-limits module consumable by the P16 backtest engine.

**Deliverables.** Energy VaR/ES engine · backtest report · GitHub repo.

**Key concepts demonstrated.** Risk methodology transfer to a new asset class, copula dependence, domain-specific stress design, Nordic electricity market structure.

**Target companies.** Vattenfall BA Markets · Statkraft · Fortum · Ingrid · Nordic trading desks.

**SQL layer (~2 hrs):** Load ENTSO-E SE1–SE4 price/volume data as a DuckDB fact table (keys: area, delivery_hour). Do the resampling, cross-area price joins (SE1–SE4 side-by-side per hour), and `>100 EUR/MWh` / negative-price spike-flagging in SQL before the Python risk layer consumes it. Matches how energy-trading analytics desks store and query market data.

---
## P14 — Nordic Electricity Price Forecasting *(Supports P16)*

| | |
|---|---|
| **Domain** | Energy / Time-Series Forecasting |
| **Status** | Active-optional — now also serves as the forecast input layer for P16 |
| **Timeline** | ~25–30 hrs |

**Note.** The Austrian electricity load forecasting and ARMA-GARCH repos are now published on GitHub. The SE3 day-ahead price forecasting project (SARIMA, Prophet, LightGBM on 35,000+ ENTSO-E observations) is completed and listed in the CV. P14 extends this into probabilistic intraday price forecasting with exogenous features (wind/solar forecast error, system imbalance, hydro reservoir levels), directly feeding the P16 optimization engine.

**Scope of work.**
- Probabilistic intraday price forecasting (quantile regression via LightGBM, not just point forecasts).
- Exogenous features: wind/solar forecast errors, system imbalance, hydro reservoir levels, Nordic flow data.
- GARCH volatility overlay from P1 for confidence bands.
- Walk-forward validation with realistic execution timing (forecast at t, trade at t+1).

**Deliverables.** Multi-model forecast comparison · Streamlit dashboard · GitHub repo · report · forecast module consumable by P16.

**Key concepts demonstrated.** Probabilistic time-series forecasting, exogenous-feature engineering, intraday market dynamics, forecast calibration.

**Target companies.** Vattenfall BA Markets · Statkraft · Nord Pool · Modity · OX2 · Ingrid.

**SQL layer (~2 hrs):** Load ENTSO-E SE3 hourly prices, weather, and gas/LNG series as DuckDB tables; build the intraday feature matrix (calendar features, lagged prices via window functions, spark-spread as a joined gas-price column) in SQL before model training. Reuses the same fact-table pattern as P13.

---
## P15 — Financial-Data SQL Analytics Layer *(NEW)*

| | |
|---|---|
| **Domain** | Data Engineering / SQL |
| **Status** | **Prepared — Jun–Aug 2026, ~6–8 hrs (fits a quiet pre-CFA week)** |
| **Timeline** | No finance theory beyond what P1 already delivered |

**Why this exists.** The SQL layers in P2/P6/P8/P13 are realistic — SQL embedded inside larger projects — but that means SQL is never the *headline* of any repo. P15 is the one deliverable that makes SQL independently visible: a standalone repo where SQL is the product, so "show me your SQL" is answered with a repo rather than a buried module. It also doubles as SQL-interview prep, drilling exactly the constructs interviewers probe (JOINs, GROUP BY, window functions, CTEs). SEB RVS and Klarna both list SQL; this turns the skills-line claim into something demonstrable.

**Scope of work.**
- Build a small **star schema**: a market-data fact table (date, instrument_id, price, volume, return) plus `dim_instrument` and `dim_date` dimension tables, loaded from existing yfinance / ENTSO-E pulls already used in P1.
- Write a **documented query library**:
  - Rolling returns and rolling volatility via window functions (`AVG/STDDEV OVER (PARTITION BY instrument_id ORDER BY date ROWS BETWEEN ...)`).
  - Maximum drawdown per instrument (running peak via window function, then drawdown).
  - A **VaR-breach counter**: join a realised-P&L table against a VaR-limit table and count/flag breaches per period.
  - Portfolio P&L aggregation: join holdings × instrument returns, GROUP BY portfolio and date.
  - At least one multi-step **CTE** chain (e.g., daily returns → rolling vol → annualised vol ranking).
- README documenting each query, the schema diagram, and the SQL constructs demonstrated.

**Deliverables.** `sql_analytics/` repo · DuckDB/SQLite schema + load script · documented `.sql` query library · schema diagram · README · GitHub repo.

**Key concepts demonstrated.** Star-schema / dimensional modelling, JOINs, GROUP BY aggregation, window functions (`OVER`, `PARTITION BY`, `LAG`, running aggregates), CTEs, financial-data query patterns (rolling vol, drawdown, VaR-breach detection, P&L aggregation).

**Target companies.** SEB RVS · Klarna · Nordea · Swedbank · any credit-risk or financial-data-scientist role listing SQL.

---
## P16 — Nordic BESS Revenue Optimizer & Intraday Trading Simulator ← ENERGY TRADING FLAGSHIP

| | |
|---|---|
| **Domain** | Energy Trading / Optimization / BESS |
| **Status** | **PRIORITY ↑↑ — Aug–Sep 2026, ~60–80 hrs** |
| **Timeline** | Deliver before Ingrid Quantitative Power Trader application |
| **Builds on** | P1 (VaR/ES engine), P13 (Nordic electricity risk), P14 (intraday price forecasting), existing SE3 forecasting work |

**Why this is the #1 energy priority.** The Ingrid Quantitative Power Trader role requires: *"Design, backtest, and continuously improve intraday trading algorithms — arbitrage and proprietary strategies — applying revenue-optimization thinking across ancillary, day-ahead, and intraday markets."* This project is the single deliverable that maps to every bullet in the job description: it demonstrates the full quant-trader pipeline from forecast through optimization through backtest through live performance monitoring — the exact stack Ingrid operates. The existing SE3 price forecasting project (SARIMA, Prophet, LightGBM on ENTSO-E data) and grid frequency anomaly detection (7.8M rows Fingrid) establish energy-domain credibility; P16 elevates it from "can forecast" to "can trade."

**Scope of work.**

*1. Multi-market BESS Revenue Stack Model.*
- Day-ahead spread capture: charge during low-price hours, discharge during high-price hours.
- Intraday continuous trading: arbitrage signals informed by probabilistic price forecasts.
- FCR-D and aFRR ancillary service revenues with Nordic-specific market rules (Svenska Kraftnät capacity markets).
- Opportunity-cost framework: allocate limited battery capacity across competing revenue streams.

*2. Probabilistic Intraday Price Forecasting.*
- Extends the existing SE3 forecasting work to shorter horizons (15-min, 30-min, 1-hour intraday products).
- LightGBM quantile regression for probabilistic forecasts (5th/50th/95th percentiles).
- Exogenous features: wind/solar forecast error, system imbalance, hydro reservoir levels, cross-border flow data.
- GARCH volatility overlay for dynamic confidence bands, reusing P1 `var_engine/garch.py`.

*3. Optimization Engine (MILP).*
- Mixed-integer linear programming via `cvxpy` or `pyomo`.
- Decision variables: charge/discharge power, reserve capacity allocation (FCR-D up/down), state of charge.
- Objective: maximize expected revenue across FCR-D + day-ahead spread + intraday arbitrage.
- Constraints: power rating (MW), energy capacity (MWh), round-trip efficiency, min/max SoC, cycle degradation cost, risk limits on open intraday position (imported from P13 VaR framework).
- Rolling-horizon re-optimization: re-solve every 15 minutes with updated forecasts.

*4. Degradation-Adjusted Cycling Economics.*
- Cycle aging as a function of depth-of-discharge (literature-based: Xu et al. 2018, or equivalent).
- Calendar aging component (time-at-temperature).
- Marginal degradation cost per MWh cycled — fed into the optimization as a penalty term so the optimizer avoids uneconomical shallow cycles.
- Replacement-cost breakeven analysis: "At what spread does cycling become profitable net of degradation?"

*5. Walk-Forward Backtest Engine.*
- Realistic execution: trade decisions use *forecast* prices; P&L is settled at *realized* prices — no look-ahead.
- Bid-ask spread and market impact model (slippage as function of volume).
- Performance metrics: total P&L by market, Sharpe ratio, max drawdown, capacity utilization rate, capture rate vs. theoretical optimum (the "efficiency gap").
- Benchmark comparisons: greedy strategy (charge when price < daily avg), day-ahead-only, FCR-only.

*6. P&L Attribution Module.*
- Decompose deviation between realized P&L and theoretical optimum into: forecast error contribution, constraint-binding cost, execution slippage, degradation cost, and reserve-availability impact.
- Attribution waterfall chart per trading day.

*7. Streamlit Dashboard.*
- Live strategy monitor: current SoC, open positions, forecast vs. realized prices, P&L by market.
- What-if scenario explorer: adjust battery size, efficiency, degradation cost, risk limits — see impact on expected revenue.

**Data sources (all free/public).**

| Data | Source |
|---|---|
| Day-ahead prices (SE1–SE4) | ENTSO-E Transparency Platform / Nord Pool |
| Intraday prices (SE3 15-min) | Nord Pool intraday API |
| FCR-D, aFRR prices & volumes | Svenska Kraftnät open data |
| Wind/solar generation (actual vs. forecast) | ENTSO-E / SMHI |
| System imbalance | Svenska Kraftnät |
| Cross-border physical flows | ENTSO-E |
| Load data | ENTSO-E |

**Deliverables.** `bess_trader/` Python package: `forecast.py` (probabilistic intraday price model), `optimizer.py` (MILP charge/discharge/reserve allocation), `degradation.py` (cycle-life cost model), `backtest.py` (walk-forward P&L engine with realistic execution), `attribution.py` (P&L explain with waterfall charts), `dashboard.py` (Streamlit app). Clean Jupyter notebooks: `01_data_exploration.ipynb`, `02_price_model.ipynb`, `03_backtest_results.ipynb`. LaTeX report documenting methodology, market assumptions, and results. GitHub repo with unit tests, type hints, and README.

**Key concepts demonstrated.** BESS revenue stacking (FCR-D + day-ahead + intraday), mixed-integer linear programming for optimal dispatch, degradation-adjusted cycling economics, probabilistic intraday price forecasting, walk-forward backtesting with realistic execution, P&L attribution and performance monitoring, Nordic electricity market design (price areas, ancillary service products), risk-aware trading under VaR limits — the full quant power trader toolkit.

**Target companies.** Ingrid (Stockholm) ← primary target · Vattenfall BA Markets · Statkraft · Fortum · Modity · Centrica Energy Trading · Danske Commodities · any Nordic BESS/energy trading desk.

**Interview narrative.** This project provides specific answers to every question an Ingrid interviewer will ask:
- *"Walk me through a trading strategy you built."* → The MILP optimizer architecture and the three-market revenue stack.
- *"How do you handle model uncertainty in price forecasts?"* → Probabilistic quantile regression + scenario testing under the backtest engine.
- *"What do you do when live P&L deviates from expected?"* → The attribution module decomposing deviation into forecast error, constraint cost, and slippage.
- *"How do you think about battery degradation in trading decisions?"* → The marginal-degradation-cost model and how it feeds the optimizer's penalty term.
- *"Have you worked with production code?"* → Clean Python architecture, `cvxpy`/`pyomo` optimization, Streamlit monitoring dashboard, typed interfaces.

**Build strategy (phased).**

| Phase | Weeks | Effort | Deliverable |
|---|---|---|---|
| Phase 1 — Synthetic data prototype | Aug W1–2 | ~15 hrs | MILP optimizer + backtest engine running on synthetic price paths. Verify the optimization logic and backtest harness are correct before real data. |
| Phase 2 — Real data pipeline | Aug W3–4 | ~20 hrs | ENTSO-E + Nord Pool + SvK data ingestion. Probabilistic forecast model trained on real SE3 intraday data. Full walk-forward backtest with real prices. |
| Phase 2* — Model refinement | Sep W1 | ~10 hrs | Hydra + MLFlow config management. Hyperparameter sweep on forecast model. Degradation model calibration against literature. |
| Phase 3 — Presentation | Sep W2 | ~12 hrs | Jupyter notebooks, LaTeX report, Streamlit dashboard. Clean up figures, write methodology, prepare interview talking points. |
| Phase 4 — Streamlit deployment | Sep W3 | ~8 hrs | Polish dashboard. What-if scenario tool. |
| Phase 5 — Repo finalization | Sep W3–4 | ~5 hrs | Cleanup, verification, dead code removal, README, GitHub release. |


---
## P17 — Energy Cross-Commodity Trading Analytics Platform ✅

| | |
|---|---|
| **Domain** | Energy Trading / Cross-Commodity Analytics |
| **Status** | **PRIORITY ↑ — Aug 4–10, 2026, ~40 hrs** |
| **Timeline** | Deliver before Equinor graduate applications open (7 Aug 2026) |
| **Builds on** | P0 (Black-Scholes/Black-76 options pricing), P1 (VaR/ES engine), P15 (SQL analytics) |

**Why this is the #1 priority this week.** Equinor's 2027 Market Analysis & Trading Graduate Programme opens 7 August 2026. Their MMP (Marketing, Midstream & Processing) desk in Stavanger trades crude oil, natural gas, LNG, power, and carbon — a multi-commodity physical + derivatives book. The existing portfolio has power/BESS coverage (P13, P14, P16) but zero crude oil and zero natural gas projects. P17 closes that gap: it demonstrates cross-commodity market understanding — the fuel-switching logic, carbon pass-through, and correlation dynamics that a physical energy trader thinks about daily. Built on P0 (options extended to Black-76 on commodity futures) and P1 (VaR/ES extended to multi-commodity with copula dependence), P17 reuses ~50% of existing code and ships a working trading analytics dashboard in 40 hours. For a graduate rotation programme where you could land on crude, gas, or power, breadth across all four commodities is more valuable than depth in one.

**Scope of work.**

*1. Multi-Commodity Data Pipeline (~5 hrs).*
- Pull daily settlement prices: Brent crude futures (ICE), TTF natural gas (ICE/EEX), EUA carbon allowances (EEX), German baseload power (EEX), Nord Pool system price (ENTSO-E).
- Normalize all prices to EUR/MWh for cross-commodity comparability.
- DuckDB fact table: `fact_energy_prices` (date, commodity, price_eur_mwh, volume) + `dim_commodity`.
- Query library: rolling volatility, year-on-year returns, max drawdown per commodity — reusing P15 SQL patterns.

*2. Spread Economics Engine (~8 hrs).* ← **The deep module — the one you talk about for 10 minutes in an interview.**
- **Clean spark spread:** German power price − (TTF gas price / plant efficiency) − (EUA carbon price × emission factor). Measures the profitability of a gas-fired power plant. When spark spread > 0, gas plants run; when < 0, they idle.
- **Dark spread:** German power − (coal price / efficiency) − carbon cost. The coal-plant equivalent.
- **Fuel-switching indicator:** When clean spark > clean dark, gas outcompetes coal in the merit order. Track fuel-switching dynamics 2019–2025, including the August 2022 gas crisis where spark spreads went deeply negative and coal plants ramped.
- **3-2-1 Crack spread:** (2 × RBOB gasoline + 1 × Heating Oil − 3 × Brent crude) / 3. The refinery margin — core to Equinor's CPL (Crude, Products & Liquids) desk.
- Historical distribution, seasonality decomposition, and regime-shift detection for each spread.

*3. Correlation & Dependence Modeling (~6 hrs).*
- Rolling 60-day correlation matrix: Brent↔TTF, TTF↔power, carbon↔power, Brent↔carbon.
- DCC-GARCH for time-varying conditional correlations — captures how dependence tightens during crises.
- t-copula fit for tail dependence: "When TTF spikes to the 99th percentile, what happens to German power?"
- Visualize correlation regime shifts: pre-2022 (modest gas-power link) vs. 2022 crisis (near-perfect correlation) vs. post-2023 (new equilibrium with structural gas-power decoupling via renewables).

*4. Multi-Commodity VaR Engine (~5 hrs).*
- Build a representative MMP portfolio: long Brent (+$10M), short RBOB/HO crack spread (−$5M), long TTF (+$8M), short German power spark spread (−$4M), long EUA carbon (+$3M).
- Historical VaR/ES at 95%/99% using P1's `var_engine/var_methods.py`, extended to multi-commodity with t-copula aggregation.
- Risk decomposition: "Which commodity drives my VaR?" — stacked bar by commodity.
- Delta-normal VaR for any options overlay (using P0's Greek calculations adapted to Black-76).

*5. Stress Scenario Engine (~5 hrs).*
- **Scenario A — Gas Crisis (Nord Stream zero):** TTF +300%, power +200%, carbon +50%, Brent +30%. P&L impact on the MMP book.
- **Scenario B — Global Recession:** Brent −40%, TTF −30%, power −25%, carbon −20%. All correlations → 1 in risk-off.
- **Scenario C — Energy Transition Accelerates:** Carbon +200% (€150/t), coal destroyed, spark spreads structurally wider as gas becomes the marginal fuel. Renewables cannibalize power prices.
- Each scenario shows: P&L by position, VaR shift, correlation matrix change.

*6. Streamlit Trading Dashboard (~8 hrs).*
- **Tab 1 — Market Monitor:** Multi-commodity price heatmap (daily returns colored by magnitude). Spark/dark/crack spread time series with rolling z-score and regime highlighting.
- **Tab 2 — Correlation Lab:** Rolling correlation matrix heatmap with date slider. DCC-GARCH conditional correlation overlay. t-copula tail-dependence scatter plot (Brent vs. TTF in the upper tail).
- **Tab 3 — Risk Command:** VaR decomposition waterfall by commodity. Historical VaR breaches highlighted. Stress scenario P&L waterfall (select scenario → see impact).
- **Tab 4 — Fuel Switch:** German merit order visualization: spark vs. dark spread, fuel-switching signal, carbon price pass-through rate.

*7. Cleanup & Documentation (~3 hrs).*
- Jupyter notebooks: `01_multi_commodity_eda.ipynb`, `02_spread_economics.ipynb`, `03_correlation_crisis.ipynb`, `04_portfolio_var.ipynb`.
- README with architecture diagram (data pipeline → spread engine → correlation → VaR → scenario → dashboard), methodology notes, and data sources.
- GitHub repo: `energy_cross_commodity/` with clean module structure, type hints, and unit tests on the VaR and spread engines.

**Data sources (all free/public).**

| Data | Source |
|---|---|
| Brent crude futures | ICE (via yfinance `BZ=F` or EIA) |
| TTF natural gas | ICE / EEX (via public reports or yfinance proxy) |
| EUA carbon | EEX (via public auction results or yfinance) |
| German baseload power | EEX / ENTSO-E Transparency |
| Nord Pool system price | ENTSO-E / Nord Pool public data |
| Coal (API2) | ICE (yfinance proxy) |

**Deliverables.** `energy_cross_commodity/` Python package: `data/pipeline.py` (multi-commodity → DuckDB), `spreads/spark_spread.py` (clean spark/dark spread + fuel switch), `spreads/crack_spread.py` (3-2-1 crack spread), `risk/correlation.py` (DCC-GARCH + rolling corr + copula), `risk/var_engine.py` (extends P1 to multi-commodity), `risk/scenarios.py` (stress scenario definitions + P&L impact), `dashboard/app.py` (4-tab Streamlit). Jupyter notebooks (×4). README with architecture diagram. GitHub repo.

**Key concepts demonstrated.** Cross-commodity energy market economics (spark spread, dark spread, crack spread, fuel switching), EU ETS carbon market mechanics and merit-order effects, DCC-GARCH time-varying correlation, t-copula tail dependence, multi-asset VaR/ES with copula aggregation, stress testing and scenario analysis, Black-76 commodity options pricing (extending P0), energy-commodity data modeling in SQL (DuckDB star schema), interactive trading analytics dashboard — the full cross-commodity analytics toolkit that an MMP trade desk analyst uses daily.

**Target companies.** Equinor (Stavanger) ← primary target · Danske Commodities (Aarhus) · Centrica Energy Trading (Aalborg) · Gunvor (Geneva/Singapore) · Vitol · Trafigura · Mercuria · any multi-commodity energy trading desk.

**Interview narrative.** This project provides specific answers to every question an Equinor graduate panel will ask:
- *"Tell us about a project you're proud of."* → "I built a cross-commodity energy trading analytics platform covering Brent crude, TTF gas, EUA carbon, and European power. The core insight is that these markets don't move independently — I modeled how carbon prices push coal out of the German merit order, widening spark spreads."
- *"How do you think about risk across different commodities?"* → "My VaR engine uses t-copula aggregation because energy commodities have strong tail dependence — when TTF spikes, power and carbon follow. The 2022 crisis is in the training data."
- *"What happens to our portfolio if there's another gas crisis?"* → Walk them through Scenario A: the P&L waterfall, the correlation regime shift, the VaR impact. "Your long TTF position saves the book; your short spark spread kills it — the net depends on relative sizing."
- *"Do you understand how carbon markets work?"* → "I model the EU ETS pass-through explicitly in the fuel-switching engine. At €100/t carbon, a 40%-efficient coal plant pays €85/MWh in carbon costs alone — gas at 55% efficiency pays €62/MWh. That €23 gap is why carbon prices drive gas-to-power economics."
- *"Have you worked with production code?"* → Clean Python architecture, DuckDB/SQL, Streamlit dashboard, typed interfaces, reusable modules — the same patterns from P0 and P1, extended to a new asset class.

**Build strategy (one-week sprint).**

| Day | Hours | Module | Deliverable |
|---|---|---|---|
| Mon | 5 | Multi-commodity data pipeline | DuckDB schema + Brent/TTF/carbon/power data loaded. Price normalization in EUR/MWh. |
| Tue | 8 | Spread economics engine | Spark, dark, and crack spread computations. Fuel-switching indicator. Historical analysis. |
| Wed | 6 | Correlation & dependence | DCC-GARCH, rolling correlation matrix, t-copula fit. Crisis regime visualization. |
| Thu | 5 | Multi-commodity VaR | MMP portfolio VaR/ES with copula aggregation. Risk decomposition. |
| Fri | 8 | Stress scenarios + Streamlit dashboard | Scenario P&L engine. 4-tab dashboard wired to all modules. |
| Sat | 5 | Scenario engine polish + dashboard integration | Complete scenario waterfall charts. Dashboard polish. |
## Role Type — Project Mapping

| Role Type                                    | Primary Projects            |
| -------------------------------------------- | --------------------------- |
| **Energy Quant — Power/BESS (Ingrid)**       | **P16 ↑, P14, P13, P1 ✅**   |
| **Energy Quant — Cross-Commodity (Equinor)** | **P17 ✅, P0 ✅, P1 ✅, P15**  |
| **Energy / Commodity Trading (General)**     | P16, P17 ✅, P1 ✅, P13, P14  |
| **Risk Analyst (Market Risk)**               | P1 ✅, P0 ✅, P5, P9          |
| **Risk Analyst (Credit Risk / IRB)**         | P2 ✅, P6 ✅, P10, P15        |
| **Model Validation (SEB Front Office)**      | P0 ✅, P1 ✅, P3 ✅            |
| **Financial Data Scientist**                 | P2 ✅, P10, P11, P9, P15     |
| **Asset Management / Buy-Side Quant**        | P4, P5, P7, P8, P11         |
| **Portfolio Management**                     | P4, P7, P8, P12             |
| **Pension / ALM**                            | P3 ✅, P12, P7, P8           |
| **Fixed Income / Rates**                     | P3 ✅, P12                   |
| **Energy / Commodity Risk (Vattenfall)**     | P1 ✅, P13, P14, P16, P17 ✅  |
| **SEB RVS Junior Quant (part-time)**         | P0 ✅, P1 ✅, P2 ✅, P3 ✅, P15 |
