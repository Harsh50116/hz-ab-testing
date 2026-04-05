"""Tests for A/B variant assignment."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from hz_ab_testing import config
from hz_ab_testing.assign import assign_variants


def _classified(n_per_pool: dict[str, int]) -> pd.DataFrame:
    rows = []
    cid = 1
    for pool, n in n_per_pool.items():
        for _ in range(n):
            rows.append({"customer_id": f"C{cid:04d}", "derived_pool": pool})
            cid += 1
    return pd.DataFrame(rows)


def test_all_customers_assigned():
    classified = _classified({"evening_workday": 40, "weekend_noon": 35, "unknown": 25})
    out = assign_variants(classified)
    assert len(out) == 100
    assert set(out["customer_id"]) == set(classified["customer_id"])
    assert set(out["assigned_variant"]).issubset(set(config.VARIANTS))


def test_split_within_pool_is_balanced():
    # With 33 customers per pool, we should get exactly 11/11/11.
    classified = _classified({"evening_workday": 33, "weekend_noon": 33, "unknown": 33})
    out = assign_variants(classified)
    for pool in ["evening_workday", "weekend_noon", "unknown"]:
        counts = Counter(
            out[out["derived_pool"] == pool]["assigned_variant"]
        )
        assert counts[config.VARIANTS[0]] == 11
        assert counts[config.VARIANTS[1]] == 11
        assert counts[config.VARIANTS[2]] == 11


def test_split_33_33_34():
    # 100 in one pool → 34/33/33 (first variant gets the extras by position%3 rule).
    classified = _classified({"evening_workday": 100})
    out = assign_variants(classified)
    counts = Counter(out["assigned_variant"])
    assert sum(counts.values()) == 100
    # one variant has 34, two have 33
    assert sorted(counts.values()) == [33, 33, 34]


def test_determinism():
    classified = _classified({"evening_workday": 40, "weekend_noon": 35, "unknown": 25})
    out1 = assign_variants(classified, seed=42)
    out2 = assign_variants(classified, seed=42)
    pd.testing.assert_frame_equal(out1, out2)


def test_different_seeds_differ():
    classified = _classified({"evening_workday": 40, "weekend_noon": 35, "unknown": 25})
    out1 = assign_variants(classified, seed=42)
    out2 = assign_variants(classified, seed=99)
    # Extremely unlikely the assignments match byte-for-byte across seeds.
    assert not out1.equals(out2)


def test_empty_pool_handled():
    classified = _classified({"evening_workday": 10, "weekend_noon": 0, "unknown": 5})
    # Note: pools with 0 customers won't appear in unique(); nothing should break.
    out = assign_variants(classified)
    assert len(out) == 15
