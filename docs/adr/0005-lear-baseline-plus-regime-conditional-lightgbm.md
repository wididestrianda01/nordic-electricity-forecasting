# LEAR baseline + HMM-regime-conditional LightGBM quantile model, not deep learning

**Status:** superseded by ADR-0009

Current EPF practice offers a choice between interpretable statistical/ML pipelines (LEAR, gradient-boosted quantile models) and deep learning (TFT, N-BEATS, distributional NNs). We chose LEAR (LASSO-AR) as the baseline and an HMM-regime-labeled LightGBM quantile model as the primary forecaster, producing quantile grids directly via LightGBM's quantile objective. Deep learning is deliberately out of scope for v1. The trade-off: a tuned transformer could plausibly beat this on raw accuracy, but 25-30 hours doesn't cover tuning one properly, and an interpretable pipeline makes the regime-conditioning logic (ADR-0004) visible and explainable to a recruiter — the point of this project — rather than buried in a black box.
