"""Tests for rule-based timing-pool classifier."""
from __future__ import annotations

import pandas as pd
import pytest

from hz_ab_testing.segment import classify_customer, classify_all


def _mk_events(customer_id: str, timestamps: list[str], types: list[str] | None = None):
    if types is None:
        types = ["open"] * len(timestamps)
    return pd.DataFrame(
        {
            "event_id": [f"E{i}" for i in range(len(timestamps))],
            "customer_id": [customer_id] * len(timestamps),
            "event_type": types,
            "occurred_at": timestamps,
        }
    )


def test_empty_events_unknown():
    res = classify_customer("C1", _mk_events("C1", []))
    assert res.derived_pool == "unknown"
    assert res.n_events == 0


def test_too_few_events_unknown():
    # 2 opens (weight 2) < MIN_EVENTS_FOR_CLASSIFICATION=3
    events = _mk_events(
        "C1",
        ["2026-01-05T19:00:00+00:00", "2026-01-06T20:00:00+00:00"],
    )
    res = classify_customer("C1", events)
    assert res.derived_pool == "unknown"


def test_strong_evening_signal():
    # 6 weekday-evening opens → clear evening_workday
    events = _mk_events(
        "C1",
        [
            "2026-01-05T18:00:00+00:00",  # Mon
            "2026-01-06T19:30:00+00:00",  # Tue
            "2026-01-07T20:15:00+00:00",  # Wed
            "2026-01-08T17:45:00+00:00",  # Thu
            "2026-01-12T21:00:00+00:00",  # Mon
            "2026-01-13T19:00:00+00:00",  # Tue
        ],
    )
    res = classify_customer("C1", events)
    assert res.derived_pool == "evening_workday"
    assert res.evening_share == 1.0


def test_strong_weekend_signal():
    events = _mk_events(
        "C1",
        [
            "2026-01-03T11:00:00+00:00",  # Sat
            "2026-01-04T12:30:00+00:00",  # Sun
            "2026-01-10T13:00:00+00:00",  # Sat
            "2026-01-11T10:30:00+00:00",  # Sun
            "2026-01-17T11:45:00+00:00",  # Sat
        ],
    )
    res = classify_customer("C1", events)
    assert res.derived_pool == "weekend_noon"


def test_ambiguous_signal_unknown():
    # Mix of evening and weekend with no clear majority
    events = _mk_events(
        "C1",
        [
            "2026-01-05T19:00:00+00:00",  # Mon evening
            "2026-01-06T20:00:00+00:00",  # Tue evening
            "2026-01-03T11:00:00+00:00",  # Sat noon
            "2026-01-04T12:00:00+00:00",  # Sun noon
            "2026-01-07T09:00:00+00:00",  # Wed morning (other)
        ],
    )
    res = classify_customer("C1", events)
    assert res.derived_pool == "unknown"


def test_clicks_weighted_higher_than_opens():
    # 3 opens (weight 3) outside any window + 2 clicks (weight 4) in evening
    # → evening should dominate.
    events = _mk_events(
        "C1",
        [
            "2026-01-05T09:00:00+00:00",  # Mon morning — other
            "2026-01-06T09:00:00+00:00",  # Tue morning — other
            "2026-01-07T09:00:00+00:00",  # Wed morning — other
            "2026-01-08T19:00:00+00:00",  # Thu evening — click
            "2026-01-12T20:00:00+00:00",  # Mon evening — click
        ],
        types=["open", "open", "open", "click", "click"],
    )
    res = classify_customer("C1", events)
    assert res.weighted_events == 3 * 1.0 + 2 * 2.0  # 7.0
    assert res.derived_pool == "evening_workday"


def test_classify_all_returns_all_customers():
    customers = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
        }
    )
    events = pd.concat(
        [
            _mk_events("C1", ["2026-01-05T19:00:00+00:00"] * 5),
            _mk_events("C2", []),
            _mk_events("C3", ["2026-01-03T11:00:00+00:00"] * 5),
        ],
        ignore_index=True,
    )
    out = classify_all(customers, events)
    assert set(out["customer_id"]) == {"C1", "C2", "C3"}
    assert len(out) == 3
