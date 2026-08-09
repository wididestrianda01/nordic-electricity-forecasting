"""Shared regime-boundary dates (ADR-0007) so lag lookback windows never blend
data across a structural price-formation break.
"""

from datetime import date


# Empirically determined price-regime boundaries where market structure changed.
# See ADR-0007 for the analytical basis.
REGIME_BOUNDARIES: tuple[date, ...] = (
    date(2024, 11, 4),  # Flow-based market coupling introduced
    date(2025, 10, 1),  # MTU resolution switched from 60 to 15 minutes
)


def crosses_boundary(date_a: date, date_b: date) -> bool:
    """True if any regime boundary falls strictly between date_a and date_b.

    Boundary is considered "after" the date it marks; i.e., if the boundary
    falls between the two dates (lo < boundary <= hi), a crossing is detected.

    Args:
        date_a: First date.
        date_b: Second date.

    Returns:
        True if any boundary from REGIME_BOUNDARIES lies in the range
        (min(date_a, date_b), max(date_a, date_b)], False otherwise.
    """
    lo, hi = sorted((date_a, date_b))
    return any(lo < boundary <= hi for boundary in REGIME_BOUNDARIES)
