"""Rule-based, seeded generation of customers, carts, and engagement events."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

from . import config
from .models import (
    Cart,
    CartItem,
    Customer,
    EngagementEvent,
    LifestyleInterest,
    Product,
    Segment,
    TimingPool,
)

# Single "now" anchor per run for reproducibility.
_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weighted_choice(rng: random.Random, weights: dict) -> str | int:
    keys = list(weights.keys())
    vals = list(weights.values())
    return rng.choices(keys, weights=vals, k=1)[0]


def _make_email(rng: random.Random, name: str, used: set[str]) -> str:
    parts = name.lower().replace("'", "").split()
    base = ".".join(parts) if len(parts) >= 2 else parts[0]
    for _ in range(20):
        suffix = rng.randint(1, 9999)
        domain = rng.choice(EMAIL_DOMAINS)
        candidate = f"{base}{suffix}@{domain}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    # very unlikely fallback
    candidate = f"{base}{rng.randint(10000, 99999)}@{rng.choice(EMAIL_DOMAINS)}"
    used.add(candidate)
    return candidate


def _assign_timing_pool(rng: random.Random, segment: Segment) -> TimingPool:
    """New customers lack engagement history → always 'unknown'.
    Returning/VIP split between the two known pools to hit the overall target.
    """
    if segment == "new":
        return "unknown"
    # 40/35 over the 75% of non-new customers → normalize.
    # ~53.3% evening_workday, ~46.7% weekend_noon.
    return rng.choices(
        ["evening_workday", "weekend_noon"],
        weights=[40 / 75, 35 / 75],
        k=1,
    )[0]


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------

def generate_customers(
    rng: random.Random,
    name_pool: Iterable[str],
    n: int = config.N_CUSTOMERS,
) -> list[Customer]:
    names = list(name_pool)
    rng.shuffle(names)
    if len(names) < n:
        raise ValueError(f"Name pool has {len(names)} names; need at least {n}.")

    used_emails: set[str] = set()
    customers: list[Customer] = []
    for i in range(n):
        segment = _weighted_choice(rng, config.SEGMENT_WEIGHTS)
        aov_lo, aov_hi = config.AOV_RANGE_BY_SEGMENT[segment]
        ord_lo, ord_hi = config.TOTAL_ORDERS_RANGE_BY_SEGMENT[segment]
        city, state = rng.choice(config.COLD_STATES)
        age_range = rng.choices(config.AGE_RANGES, weights=config.AGE_WEIGHTS, k=1)[0]
        income = rng.choices(
            config.INCOME_BRACKETS, weights=config.INCOME_WEIGHTS, k=1
        )[0]
        pool = _assign_timing_pool(rng, segment)

        aov = round(rng.uniform(aov_lo, aov_hi), 2)
        total_orders = rng.randint(ord_lo, ord_hi)
        clv = round(aov * total_orders, 2)

        lp_lo, lp_hi = config.LAST_PURCHASE_RANGE_BY_SEGMENT[segment]
        last_purchase_days_ago = rng.randint(lp_lo, lp_hi)

        rr_mean, rr_std = config.RETURN_RATE_BY_SEGMENT[segment]
        return_rate = round(max(0.0, min(100.0, rng.gauss(rr_mean, rr_std))), 1)

        acq_channel = rng.choices(
            config.ACQUISITION_CHANNELS,
            weights=config.ACQUISITION_WEIGHTS_BY_SEGMENT[segment],
            k=1,
        )[0]

        n_interests = rng.choices([1, 2, 3], weights=[0.30, 0.50, 0.20], k=1)[0]
        interests: list[LifestyleInterest] = []
        available = list(config.LIFESTYLE_INTERESTS)
        available_weights = list(config.LIFESTYLE_WEIGHTS_BY_SEGMENT[segment])
        for _ in range(n_interests):
            chosen = rng.choices(available, weights=available_weights, k=1)[0]
            interests.append(chosen)
            idx = available.index(chosen)
            available.pop(idx)
            available_weights.pop(idx)

        customers.append(
            Customer(
                customer_id=f"C{i+1:04d}",
                name=names[i],
                email=_make_email(rng, names[i], used_emails),
                segment=segment,
                avg_order_value=aov,
                total_orders=total_orders,
                clv=clv,
                last_purchase_days_ago=last_purchase_days_ago,
                return_rate=return_rate,
                acquisition_channel=acq_channel,
                lifestyle_interests=interests,
                preferred_send_time=pool,
                age_range=age_range,
                income_bracket=income,
                city=city,
                state=state,
            )
        )
    return customers


# ---------------------------------------------------------------------------
# Carts
# ---------------------------------------------------------------------------

def generate_carts(
    rng: random.Random,
    customers: list[Customer],
    products: list[Product],
) -> list[Cart]:
    carts: list[Cart] = []
    for i, c in enumerate(customers):
        size = _weighted_choice(rng, config.CART_SIZE_WEIGHTS)
        chosen_products = rng.sample(products, k=size)
        items = []
        total = 0.0
        for p in chosen_products:
            qty = 1 if rng.random() < 0.85 else 2
            items.append(
                CartItem(
                    product_id=p.product_id,
                    product_name=p.name,
                    category=p.category,
                    price=p.price,
                    quantity=qty,
                )
            )
            total += p.price * qty
        abandoned_at = _NOW - timedelta(
            days=rng.randint(0, config.ABANDONMENT_WINDOW_DAYS - 1),
            hours=rng.randint(0, 23),
            minutes=rng.randint(0, 59),
        )
        carts.append(
            Cart(
                cart_id=f"K{i+1:04d}",
                customer_id=c.customer_id,
                items=items,
                cart_total=round(total, 2),
                abandoned_at=abandoned_at,
            )
        )
    return carts


# ---------------------------------------------------------------------------
# Engagement events
# ---------------------------------------------------------------------------

def _sample_event_time(rng: random.Random, pool: TimingPool) -> datetime:
    """Sample a timestamp within the history window, biased toward `pool`'s window."""
    day_offset = rng.randint(1, config.HISTORY_WINDOW_DAYS)
    base = _NOW - timedelta(days=day_offset)

    prefer = rng.random() < config.PREFERRED_WINDOW_PROB

    if pool == "evening_workday" and prefer:
        # weekday 17:00-22:00
        while base.weekday() >= 5:  # 5=Sat, 6=Sun
            base -= timedelta(days=1)
        hour = rng.randint(17, 21)
    elif pool == "weekend_noon" and prefer:
        # Sat/Sun 10:00-14:00
        while base.weekday() < 5:
            base -= timedelta(days=1)
        hour = rng.randint(10, 13)
    else:
        hour = rng.randint(0, 23)

    return base.replace(
        hour=hour,
        minute=rng.randint(0, 59),
        second=rng.randint(0, 59),
    )


def generate_engagement_events(
    rng: random.Random,
    customers: list[Customer],
) -> list[EngagementEvent]:
    events: list[EngagementEvent] = []
    counter = 0
    for c in customers:
        lo, hi = config.ENGAGEMENT_RANGE_BY_SEGMENT[c.segment]
        n_events = rng.randint(lo, hi)
        for _ in range(n_events):
            counter += 1
            event_type = "open" if rng.random() < 0.7 else "click"
            # For `unknown` pool (new customers) — pure random times; few events anyway.
            occurred_at = _sample_event_time(rng, c.preferred_send_time)
            events.append(
                EngagementEvent(
                    event_id=f"E{counter:06d}",
                    customer_id=c.customer_id,
                    event_type=event_type,
                    occurred_at=occurred_at,
                )
            )
    return events


# ---------------------------------------------------------------------------
# Orchestrator + CSV serialization
# ---------------------------------------------------------------------------

def _customers_to_df(customers: list[Customer]) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in customers])


def _carts_to_df(carts: list[Cart]) -> pd.DataFrame:
    # Flatten: one row per cart-item, plus a cart_total column repeated per row.
    rows = []
    for cart in carts:
        for item in cart.items:
            rows.append(
                {
                    "cart_id": cart.cart_id,
                    "customer_id": cart.customer_id,
                    "product_id": item.product_id,
                    "product_name": item.product_name,
                    "category": item.category,
                    "price": item.price,
                    "quantity": item.quantity,
                    "cart_total": cart.cart_total,
                    "abandoned_at": cart.abandoned_at.isoformat(),
                }
            )
    return pd.DataFrame(rows)


def _events_to_df(events: list[EngagementEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": e.event_id,
                "customer_id": e.customer_id,
                "event_type": e.event_type,
                "occurred_at": e.occurred_at.isoformat(),
            }
            for e in events
        ]
    )


def generate_all(
    products: list[Product],
    name_pool: list[str],
    *,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Run the full Phase 1 generation pipeline, seeded.

    Returns a dict with lists of pydantic models (customers, carts, events).
    """
    rng = random.Random(seed)
    customers = generate_customers(rng, name_pool)
    carts = generate_carts(rng, customers, products)
    events = generate_engagement_events(rng, customers)
    return {"customers": customers, "carts": carts, "events": events}


def write_all(result: dict) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _customers_to_df(result["customers"]).to_csv(config.CUSTOMERS_CSV, index=False)
    _carts_to_df(result["carts"]).to_csv(config.CARTS_CSV, index=False)
    _events_to_df(result["events"]).to_csv(config.ENGAGEMENT_CSV, index=False)
