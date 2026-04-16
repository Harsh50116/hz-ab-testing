"""Context assembler: pulls the 10 customer parameters for the agentic prompt."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CustomerContext:
    """All 10 parameters the LLM agent receives to decide strategy + write the email."""

    # First-party business data
    first_name: str
    cart_items: list[dict]          # [{product_name, category, price, quantity}]
    cart_total: float
    clv: float
    segment: str                    # new / returning / vip
    total_orders: int
    last_purchase_days_ago: int
    return_rate: float              # percentage 0-100

    # Marketing / channel data
    preferred_send_time: str        # evening_workday / weekend_noon / unknown
    engagement_summary: str         # pre-summarized from events
    acquisition_channel: str

    # Third-party (Faraday-style)
    income_bracket: str
    location: str                   # "city, state"
    lifestyle_interests: list[str]


def _summarize_engagement(events_df: pd.DataFrame, customer_id: str) -> str:
    """Build a human-readable engagement summary from raw events."""
    cust_events = events_df[events_df["customer_id"] == customer_id]
    total = len(cust_events)

    if total == 0:
        return "No prior email engagement on record."

    opens = len(cust_events[cust_events["event_type"] == "open"])
    clicks = len(cust_events[cust_events["event_type"] == "click"])
    open_rate_desc = f"{opens} opens, {clicks} clicks out of {total} events"

    # Timing pattern
    hours = cust_events["occurred_at"].dt.hour
    evening_count = ((hours >= 17) & (hours <= 21)).sum()
    weekend_events = cust_events["occurred_at"].dt.weekday >= 5
    weekend_noon = (weekend_events & (hours >= 10) & (hours <= 13)).sum()

    patterns = []
    if evening_count / total > 0.4:
        patterns.append("mostly engages on weekday evenings")
    if weekend_noon / total > 0.3:
        patterns.append("often engages weekend midday")
    if not patterns:
        patterns.append("no strong timing pattern")

    return f"{open_rate_desc}; {', '.join(patterns)}"


def _parse_interests(raw: str) -> list[str]:
    """Parse the lifestyle_interests column (stored as string repr of list)."""
    cleaned = raw.strip("[]").replace("'", "").replace('"', "")
    return [s.strip() for s in cleaned.split(",") if s.strip()]


def assemble_context(
    customer_id: str,
    customers: pd.DataFrame,
    carts: pd.DataFrame,
    events: pd.DataFrame,
    products: pd.DataFrame,
) -> CustomerContext:
    """Pull all 10 parameters for a single customer."""
    cust = customers[customers["customer_id"] == customer_id].iloc[0]

    # Cart items enriched with ratings
    cart_rows = carts[carts["customer_id"] == customer_id]
    items = []
    for _, row in cart_rows.iterrows():
        prod = products[products["product_id"] == row["product_id"]]
        rating = prod["rating"].values[0] if len(prod) else 0.0
        review_count = int(prod["review_count"].values[0]) if len(prod) else 0
        items.append({
            "product_name": row["product_name"],
            "category": row["category"],
            "price": float(row["price"]),
            "quantity": int(row["quantity"]),
            "rating": float(rating),
            "review_count": review_count,
        })

    cart_total = float(cart_rows["cart_total"].iloc[0]) if len(cart_rows) else 0.0

    # Parse lifestyle interests
    interests_raw = cust.get("lifestyle_interests", "[]")
    interests = _parse_interests(str(interests_raw))

    return CustomerContext(
        first_name=str(cust["name"]).split()[0],
        cart_items=items,
        cart_total=cart_total,
        clv=float(cust["clv"]),
        segment=str(cust["segment"]),
        total_orders=int(cust["total_orders"]),
        last_purchase_days_ago=int(cust["last_purchase_days_ago"]),
        return_rate=float(cust["return_rate"]),
        preferred_send_time=str(cust["preferred_send_time"]),
        engagement_summary=_summarize_engagement(events, customer_id),
        acquisition_channel=str(cust["acquisition_channel"]),
        income_bracket=str(cust["income_bracket"]),
        location=f"{cust['city']}, {cust['state']}",
        lifestyle_interests=interests,
    )
