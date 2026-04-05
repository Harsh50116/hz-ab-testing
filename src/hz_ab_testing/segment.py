"""Phase 2: derive each customer's send-timing pool from engagement events.

This is a deliberately transparent rule-based classifier. The agent's job is:
given a stream of open/click events per customer, figure out *when* they tend
to engage with email, and bucket them into one of three timing pools.

Rule:
    1. If (weighted) event count < MIN_EVENTS_FOR_CLASSIFICATION → `unknown`.
    2. Compute share of events falling in (a) weekday evenings 17:00-22:00
       and (b) weekend middays 10:00-14:00.
    3. If the larger share exceeds POOL_DOMINANCE_THRESHOLD, assign that pool.
    4. Otherwise → `unknown` (signal is present but ambiguous).

Clicks are weighted higher than opens (see config.EVENT_WEIGHTS) because a
click is a more deliberate action than an open.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from . import config
from .models import TimingPool


@dataclass
class ClassificationResult:
    customer_id: str
    derived_pool: TimingPool
    n_events: int
    weighted_events: float
    evening_share: float
    weekend_share: float
    other_share: float


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _bucket(ts: datetime) -> str:
    """Return 'evening_workday', 'weekend_noon', or 'other' for a timestamp."""
    weekday = ts.weekday()  # Mon=0..Sun=6
    hour = ts.hour
    if weekday < 5 and hour in config.EVENING_HOURS:
        return "evening_workday"
    if weekday >= 5 and hour in config.WEEKEND_NOON_HOURS:
        return "weekend_noon"
    return "other"


def classify_customer(
    customer_id: str,
    events: pd.DataFrame,
) -> ClassificationResult:
    """Classify a single customer given their events (subset of the events df)."""
    if events.empty:
        return ClassificationResult(
            customer_id=customer_id,
            derived_pool="unknown",
            n_events=0,
            weighted_events=0.0,
            evening_share=0.0,
            weekend_share=0.0,
            other_share=0.0,
        )

    weights = events["event_type"].map(config.EVENT_WEIGHTS).astype(float)
    buckets = events["occurred_at"].map(_parse_ts).map(_bucket)
    total_w = float(weights.sum())

    if total_w < config.MIN_EVENTS_FOR_CLASSIFICATION:
        return ClassificationResult(
            customer_id=customer_id,
            derived_pool="unknown",
            n_events=int(len(events)),
            weighted_events=total_w,
            evening_share=0.0,
            weekend_share=0.0,
            other_share=0.0,
        )

    evening_w = float(weights[buckets == "evening_workday"].sum())
    weekend_w = float(weights[buckets == "weekend_noon"].sum())
    other_w = total_w - evening_w - weekend_w

    evening_share = evening_w / total_w
    weekend_share = weekend_w / total_w
    other_share = other_w / total_w

    if (
        evening_share >= config.POOL_DOMINANCE_THRESHOLD
        and evening_share > weekend_share
    ):
        pool: TimingPool = "evening_workday"
    elif (
        weekend_share >= config.POOL_DOMINANCE_THRESHOLD
        and weekend_share > evening_share
    ):
        pool = "weekend_noon"
    else:
        pool = "unknown"

    return ClassificationResult(
        customer_id=customer_id,
        derived_pool=pool,
        n_events=int(len(events)),
        weighted_events=total_w,
        evening_share=evening_share,
        weekend_share=weekend_share,
        other_share=other_share,
    )


def classify_all(
    customers: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Classify every customer. Returns a DataFrame keyed by customer_id."""
    events_by_customer = dict(tuple(events.groupby("customer_id")))
    rows = []
    for cid in customers["customer_id"]:
        cust_events = events_by_customer.get(cid, events.iloc[0:0])
        res = classify_customer(cid, cust_events)
        rows.append(res.__dict__)
    return pd.DataFrame(rows)


def evaluate(
    classified: pd.DataFrame,
    customers: pd.DataFrame,
) -> dict:
    """Compare derived pools against ground-truth `preferred_send_time`."""
    merged = classified.merge(
        customers[["customer_id", "preferred_send_time", "segment"]],
        on="customer_id",
    )
    merged["correct"] = merged["derived_pool"] == merged["preferred_send_time"]
    overall = float(merged["correct"].mean())

    # Confusion matrix: rows = true pool, cols = derived pool
    confusion = pd.crosstab(
        merged["preferred_send_time"],
        merged["derived_pool"],
        dropna=False,
    )

    by_segment = merged.groupby("segment")["correct"].mean().to_dict()

    return {
        "overall_accuracy": overall,
        "accuracy_by_segment": by_segment,
        "confusion_matrix": confusion,
        "merged": merged,
    }
