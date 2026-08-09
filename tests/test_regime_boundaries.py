"""Test regime boundary detection and crossing logic (ticket 05)."""

from datetime import date

import pytest

from forecast_pipeline.regime_boundaries import REGIME_BOUNDARIES, crosses_boundary


class TestRegimeBoundaries:
    """REGIME_BOUNDARIES tuple is well-defined."""

    def test_boundaries_exist(self):
        """REGIME_BOUNDARIES is a non-empty tuple."""
        assert isinstance(REGIME_BOUNDARIES, tuple)
        assert len(REGIME_BOUNDARIES) > 0

    def test_boundaries_are_dates(self):
        """All boundaries are date objects."""
        for boundary in REGIME_BOUNDARIES:
            assert isinstance(boundary, date)

    def test_boundaries_are_sorted(self):
        """Boundaries are in ascending order (optional but sensible)."""
        assert list(REGIME_BOUNDARIES) == sorted(REGIME_BOUNDARIES)


class TestCrossesBoundary:
    """crosses_boundary correctly detects regime-boundary crossing."""

    def test_both_dates_before_first_boundary(self):
        """Two dates well before Nov 4 2024: no crossing."""
        # e.g., June 2024 — well before Nov 4 2024
        date_a = date(2024, 6, 10)
        date_b = date(2024, 6, 15)
        assert crosses_boundary(date_a, date_b) is False
        assert crosses_boundary(date_b, date_a) is False

    def test_both_dates_between_boundaries(self):
        """Two dates between Nov 4 2024 and Oct 1 2025: no crossing."""
        date_a = date(2024, 11, 15)
        date_b = date(2024, 12, 1)
        assert crosses_boundary(date_a, date_b) is False
        assert crosses_boundary(date_b, date_a) is False

    def test_both_dates_after_second_boundary(self):
        """Two dates well after Oct 1 2025: no crossing."""
        date_a = date(2025, 11, 1)
        date_b = date(2025, 11, 15)
        assert crosses_boundary(date_a, date_b) is False
        assert crosses_boundary(date_b, date_a) is False

    def test_straddle_first_boundary_nov_4_2024(self):
        """Dates on opposite sides of Nov 4 2024: crossing detected."""
        date_a = date(2024, 11, 3)
        date_b = date(2024, 11, 5)
        assert crosses_boundary(date_a, date_b) is True
        assert crosses_boundary(date_b, date_a) is True

    def test_straddle_second_boundary_oct_1_2025(self):
        """Dates on opposite sides of Oct 1 2025: crossing detected."""
        date_a = date(2025, 9, 30)
        date_b = date(2025, 10, 2)
        assert crosses_boundary(date_a, date_b) is True
        assert crosses_boundary(date_b, date_a) is True

    def test_boundary_date_itself_as_lower_bound(self):
        """Boundary date as the later date: crossing NOT detected (boundary is "after")."""
        # lo=Nov3, hi=Nov4 (boundary) -> True (boundary falls between, at hi)
        # per spec: "lo < boundary <= hi" so Nov3 < Nov4 <= Nov4 is True
        date_a = date(2024, 11, 3)
        date_b = date(2024, 11, 4)
        assert crosses_boundary(date_a, date_b) is True

    def test_boundary_date_itself_as_upper_bound(self):
        """Boundary date as the earlier date: crossing NOT detected (boundary is "after")."""
        # lo=Nov4, hi=Nov5 -> Nov4 < Nov4 is False, so no crossing
        date_a = date(2024, 11, 4)
        date_b = date(2024, 11, 5)
        assert crosses_boundary(date_a, date_b) is False

    def test_one_date_at_boundary(self):
        """One date exactly at the boundary, the other after."""
        # lo=Nov4, hi=Nov5 -> Nov4 < Nov4 <= Nov5 is False
        date_a = date(2024, 11, 4)
        date_b = date(2024, 11, 10)
        assert crosses_boundary(date_a, date_b) is False

    def test_both_dates_at_boundary(self):
        """Both dates exactly at the boundary: no crossing (no interval)."""
        date_a = date(2024, 11, 4)
        date_b = date(2024, 11, 4)
        assert crosses_boundary(date_a, date_b) is False

    def test_same_date_before_boundary(self):
        """Same date for both: no crossing."""
        date_a = date(2024, 6, 15)
        date_b = date(2024, 6, 15)
        assert crosses_boundary(date_a, date_b) is False

    def test_wide_span_crossing_both_boundaries(self):
        """Span from before first boundary to after second: both should cross."""
        # This is a mega-span: June 2024 to Nov 2025.
        # lo=June 10, hi=Nov 15, 2025
        # lo < Nov 4 2024 <= hi? Yes.
        # lo < Oct 1 2025 <= hi? Yes.
        date_a = date(2024, 6, 10)
        date_b = date(2025, 11, 15)
        assert crosses_boundary(date_a, date_b) is True
