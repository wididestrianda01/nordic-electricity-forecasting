"""Ticket 02: HMM-based regime detection (ADR-0004).

A Gaussian HMM is fit on price levels to label each historical period with a
structurally distinct regime, rather than assuming one global price-formation
process across known breaks (Nov 2024 flow-based coupling, Oct 2025 MTU
switch). Labels are unordered state indices, not tied to specific dates --
the HMM discovers the breaks from the data itself.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# Mirrors forecast_pipeline.pipeline.MIN_HISTORY_DAYS at hourly MTU -- an HMM
# needs enough transitions to estimate a stable transition matrix.
MIN_HISTORY_ROWS = 7 * 24


def detect_regimes(historical_data: pd.DataFrame, n_states: int = 2) -> pd.Series:
    if len(historical_data) < MIN_HISTORY_ROWS:
        raise ValueError(
            f"historical_data has {len(historical_data)} rows; "
            f"need at least {MIN_HISTORY_ROWS} for stable regime detection"
        )

    observations = historical_data[["price"]].to_numpy()
    model = GaussianHMM(n_components=n_states, covariance_type="diag", random_state=0)
    try:
        model.fit(observations)
    except Exception as e:
        raise ValueError(
            f"HMM fit failed (possible convergence or singular covariance): {type(e).__name__}: {e}"
        ) from e
    states = model.predict(observations)

    order = np.argsort(model.means_[:, 0])
    rank_by_state = {state: rank for rank, state in enumerate(order)}
    labels = [f"regime_{rank_by_state[state]}" for state in states]

    return pd.Series(labels, index=historical_data.index, name="regime")
