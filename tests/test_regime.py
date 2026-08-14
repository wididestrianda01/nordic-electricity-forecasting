import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from forecast_pipeline.regime import detect_regimes


def test_labels_are_causal_not_viterbi(two_regime_scenario, monkeypatch) -> None:
    """detect_regimes decodes from the forward-filtered posterior (observations
    up to each row only), never the full-sample Viterbi path (future-looking)."""

    def _no_viterbi(self, X, lengths=None):
        raise AssertionError("detect_regimes must not call model.predict (Viterbi)")

    monkeypatch.setattr(GaussianHMM, "predict", _no_viterbi)
    _, history = two_regime_scenario
    regimes = detect_regimes(history, n_states=2)
    assert set(regimes.unique()) <= {"regime_0", "regime_1"}



def test_returns_series_aligned_to_history_index(pre_nov_2024_scenario) -> None:
    _, history = pre_nov_2024_scenario
    regimes = detect_regimes(history)
    assert isinstance(regimes, pd.Series)
    assert regimes.index.equals(history.index)


def test_regime_labels_come_from_fixed_small_set(pre_nov_2024_scenario) -> None:
    _, history = pre_nov_2024_scenario
    regimes = detect_regimes(history, n_states=2)
    assert set(regimes.unique()) <= {"regime_0", "regime_1"}


def test_separates_two_genuinely_distinct_regimes(two_regime_scenario) -> None:
    _, history = two_regime_scenario
    regimes = detect_regimes(history, n_states=2)

    half = len(history) // 2
    first_half_label = regimes.iloc[:half].mode().iloc[0]
    second_half_label = regimes.iloc[half:].mode().iloc[0]

    assert first_half_label != second_half_label
    # each half should be dominated by one label, not a coin-flip mix
    assert (regimes.iloc[:half] == first_half_label).mean() > 0.9
    assert (regimes.iloc[half:] == second_half_label).mean() > 0.9


def test_rejects_too_short_history(malformed_input_scenario) -> None:
    _, history = malformed_input_scenario
    with pytest.raises(ValueError):
        detect_regimes(history)
