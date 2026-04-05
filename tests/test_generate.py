"""Sanity checks for Phase 1 data generation.

These tests use a synthetic in-memory catalog and name pool — no LLM calls.
"""
from __future__ import annotations

from collections import Counter

import pytest

from hz_ab_testing import config
from hz_ab_testing.generate import generate_all
from hz_ab_testing.models import Product


@pytest.fixture
def products() -> list[Product]:
    categories = [
        "jacket", "sweater", "coat", "boots", "scarf", "gloves",
        "hat", "thermal", "pants", "socks",
    ]
    return [
        Product(
            product_id=f"P{i:03d}",
            name=f"Test Item {i}",
            category=categories[i % len(categories)],
            price=50.0 + i * 10,
            rating=4.5,
            review_count=100 + i,
        )
        for i in range(1, config.N_PRODUCTS + 1)
    ]


@pytest.fixture
def name_pool() -> list[str]:
    return [f"First{i} Last{i}" for i in range(1, config.N_NAMES + 1)]


def test_customer_count(products, name_pool):
    result = generate_all(products, name_pool)
    assert len(result["customers"]) == config.N_CUSTOMERS
    assert len(result["carts"]) == config.N_CUSTOMERS


def test_segment_distribution(products, name_pool):
    result = generate_all(products, name_pool)
    counts = Counter(c.segment for c in result["customers"])
    # loose tolerance for 100 samples
    assert 0.45 <= counts["returning"] / 100 <= 0.75
    assert 0.15 <= counts["new"] / 100 <= 0.35
    assert 0.05 <= counts["vip"] / 100 <= 0.25


def test_new_customers_always_unknown_pool(products, name_pool):
    result = generate_all(products, name_pool)
    for c in result["customers"]:
        if c.segment == "new":
            assert c.preferred_send_time == "unknown"


def test_pool_distribution(products, name_pool):
    result = generate_all(products, name_pool)
    counts = Counter(c.preferred_send_time for c in result["customers"])
    assert 0.25 <= counts["evening_workday"] / 100 <= 0.55
    assert 0.20 <= counts["weekend_noon"] / 100 <= 0.50
    assert 0.10 <= counts["unknown"] / 100 <= 0.40


def test_cart_sizes(products, name_pool):
    result = generate_all(products, name_pool)
    sizes = Counter(len(cart.items) for cart in result["carts"])
    assert set(sizes).issubset({1, 2, 3})
    # size=1 should be most common
    assert sizes[1] >= sizes[2] >= sizes[3]


def test_engagement_counts_by_segment(products, name_pool):
    result = generate_all(products, name_pool)
    events_by_customer: dict[str, int] = Counter()
    for e in result["events"]:
        events_by_customer[e.customer_id] += 1
    for c in result["customers"]:
        n = events_by_customer[c.customer_id]
        lo, hi = config.ENGAGEMENT_RANGE_BY_SEGMENT[c.segment]
        assert lo <= n <= hi, f"{c.customer_id} ({c.segment}): {n} events"


def test_determinism(products, name_pool):
    r1 = generate_all(products, name_pool)
    r2 = generate_all(products, name_pool)
    ids1 = [c.customer_id + c.name for c in r1["customers"]]
    ids2 = [c.customer_id + c.name for c in r2["customers"]]
    assert ids1 == ids2
    assert len(r1["events"]) == len(r2["events"])


def test_engagement_time_bias(products, name_pool):
    """Engagement events for evening_workday customers should land on weekdays
    17:00-22:00 at least PREFERRED_WINDOW_PROB - slack of the time.
    """
    result = generate_all(products, name_pool)
    events_by_customer: dict[str, list] = {}
    for e in result["events"]:
        events_by_customer.setdefault(e.customer_id, []).append(e.occurred_at)

    # Aggregate across all evening_workday customers
    total = 0
    in_window = 0
    for c in result["customers"]:
        if c.preferred_send_time != "evening_workday":
            continue
        for ts in events_by_customer.get(c.customer_id, []):
            total += 1
            if ts.weekday() < 5 and 17 <= ts.hour <= 21:
                in_window += 1
    assert total > 0
    ratio = in_window / total
    # Expect ~ PREFERRED_WINDOW_PROB (0.75); allow slack.
    assert ratio >= 0.6, f"evening_workday in-window ratio = {ratio:.2f}"


def test_cart_total_matches_items(products, name_pool):
    result = generate_all(products, name_pool)
    for cart in result["carts"]:
        computed = round(sum(i.price * i.quantity for i in cart.items), 2)
        assert abs(computed - cart.cart_total) < 0.01
