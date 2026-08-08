# Forecast at 15-minute MTU granularity, not hourly

Nordic day-ahead auctions cleared hourly (24 MTU/day) until 30 Sep 2025, then switched to 15-minute MTU (96/day) from 1 Oct 2025 onward. We chose to model the target granularity as the current 15-minute MTU rather than aggregating back to hourly for simplicity. This is more work — 4x the series length, and the pre-Oct-2025 history needs explicit disaggregation or exclusion — but an hourly-only model would misrepresent the market a 2026 employer is actually trading in.
