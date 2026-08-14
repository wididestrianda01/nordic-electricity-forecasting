"""Feature-group classification for the ablation and transfer check (ticket 11).

The seven feature groups (ticket 08) partition the full-features matrix:
group 1 (AR/calendar/regime) is the price-only feature block; groups 2-7 are
the exogenous groups joined by ``assemble_data``. Group 2 (system
fundamentals -- the day-ahead load/wind forecast) is excluded from every
full-features arm by design (train/serve skew, ticket 08), so it never enters
a model; the ablation documents this instead of running no-op experiments.

``classify_columns`` maps a full-features column list onto the seven groups.
Group 1 is the remainder -- every column not matched to an exogenous group --
so no column belongs to two groups.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: Canonical group order (ticket 08); group 1 is the price-only block.
GROUP_ORDER: tuple[str, ...] = (
    "ar_calendar_regime",
    "system_fundamentals",
    "cross_border",
    "weather",
    "hydro",
    "commodities",
    "fx",
)

#: The group excluded from every full-features arm (day-ahead values, ticket 08).
EXCLUDED_GROUP = "system_fundamentals"

#: Exogenous columns with fixed names, per group (groups 2, 3, 5, 6, 7).
_FIXED_COLUMNS: dict[str, frozenset[str]] = {
    "system_fundamentals": frozenset({"load_forecast", "wind_forecast"}),
    "cross_border": frozenset(
        {"net_position_mwh", "scheduled_exchange_mwh", "neighbour_price_eur_mwh"}
    ),
    "hydro": frozenset({"hydro_storage_mwh"}),
    "commodities": frozenset({"carbon_eua"}),
    "fx": frozenset({"fx_sek_eur"}),
}

#: Weather (group 4) columns are flattened to ``ZONE_variable`` by
#: ``assemble_data``; the zone is one of SE1-SE4.
_WEATHER_ZONE_RE = re.compile(r"^(?:SE1|SE2|SE3|SE4)_")


def classify_columns(columns: Iterable[str]) -> dict[str, list[str]]:
    """Partition full-features column names into the seven feature groups.

    Group 1 (``ar_calendar_regime``) is the remainder: every column not matched
    to an exogenous group. Weather (group 4) is matched by the ``SE<n>_`` zone
    prefix. Returns one entry per group in ``GROUP_ORDER``; a group with no
    matching column gets an empty list.
    """
    names = list(columns)
    by_group: dict[str, list[str]] = {group: [] for group in GROUP_ORDER}
    matched: set[str] = set()

    for group, fixed in _FIXED_COLUMNS.items():
        hits = [c for c in names if c in fixed]
        by_group[group] = hits
        matched.update(hits)

    by_group["weather"] = [c for c in names if _WEATHER_ZONE_RE.match(c)]
    matched.update(by_group["weather"])

    by_group["ar_calendar_regime"] = [c for c in names if c not in matched]
    return by_group
