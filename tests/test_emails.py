"""Tests for email generation — uses a fake LLM client, no real API calls."""
from __future__ import annotations

import pandas as pd
import pytest

from hz_ab_testing.emails import (
    generate_email,
    generate_all_emails,
    _PROMPT_BUILDERS,
)


class FakeClient:
    """Records calls and returns deterministic fake emails."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, system: str, user: str, **kwargs) -> dict:
        self.calls.append((system, user))
        return {
            "subject": "Your cart is waiting",
            "body": "Hi Friend, come back. [Complete Your Purchase] — The Hazel Apparel Team",
        }


@pytest.fixture
def sample_items():
    return [
        {
            "product_id": "P001",
            "product_name": "Winter Parka",
            "category": "jacket",
            "price": 229.0,
            "quantity": 1,
            "rating": 4.8,
            "review_count": 512,
        }
    ]


def test_generate_email_seasonal_discount(sample_items):
    client = FakeClient()
    email = generate_email(client, "seasonal_discount", "Oliver", sample_items)
    assert email.subject == "Your cart is waiting"
    assert "Oliver" in client.calls[0][1]
    assert "Winter Parka" in client.calls[0][1]
    assert "20%" in client.calls[0][1]  # discount pct injected


def test_generate_email_urgency_has_no_discount(sample_items):
    client = FakeClient()
    generate_email(client, "urgency", "Oliver", sample_items)
    assert "Do NOT offer any discount" in client.calls[0][1]


def test_generate_email_personalized_rec_uses_rating(sample_items):
    client = FakeClient()
    generate_email(client, "personalized_rec", "Oliver", sample_items)
    user_prompt = client.calls[0][1]
    assert "4.8/5" in user_prompt
    assert "512 customers" in user_prompt


def test_unknown_variant_raises(sample_items):
    client = FakeClient()
    with pytest.raises(ValueError):
        generate_email(client, "bogus_variant", "Oliver", sample_items)


def test_all_variants_covered():
    assert set(_PROMPT_BUILDERS.keys()) == {
        "seasonal_discount",
        "urgency",
        "personalized_rec",
    }


def test_generate_all_emails_fills_both_columns():
    client = FakeClient()
    assignments = pd.DataFrame(
        {
            "customer_id": ["C0001", "C0002"],
            "derived_pool": ["evening_workday", "unknown"],
            "assigned_variant": ["seasonal_discount", "urgency"],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["C0001", "C0002"],
            "name": ["Oliver Brown", "Reese Garcia"],
        }
    )
    carts = pd.DataFrame(
        [
            {
                "cart_id": "K0001", "customer_id": "C0001",
                "product_id": "P001", "product_name": "Winter Parka",
                "category": "jacket", "price": 229.0, "quantity": 1,
            },
            {
                "cart_id": "K0002", "customer_id": "C0002",
                "product_id": "P002", "product_name": "Fleece Sweater",
                "category": "sweater", "price": 69.0, "quantity": 1,
            },
        ]
    )
    products = pd.DataFrame(
        [
            {"product_id": "P001", "rating": 4.8, "review_count": 512},
            {"product_id": "P002", "rating": 4.4, "review_count": 201},
        ]
    )

    out = generate_all_emails(
        client, assignments, customers, carts, products, progress=False
    )
    assert out["email_subject"].notna().all()
    assert out["email_body"].notna().all()
    assert len(client.calls) == 2


def test_cache_skips_existing_rows():
    client = FakeClient()
    assignments = pd.DataFrame(
        {
            "customer_id": ["C0001", "C0002"],
            "derived_pool": ["evening_workday", "unknown"],
            "assigned_variant": ["seasonal_discount", "urgency"],
            "email_subject": ["already done", None],
            "email_body": ["already done body", None],
        }
    )
    customers = pd.DataFrame(
        {"customer_id": ["C0001", "C0002"], "name": ["A B", "C D"]}
    )
    carts = pd.DataFrame(
        [
            {"cart_id": "K1", "customer_id": "C0001", "product_id": "P1",
             "product_name": "X", "category": "jacket", "price": 10.0, "quantity": 1},
            {"cart_id": "K2", "customer_id": "C0002", "product_id": "P2",
             "product_name": "Y", "category": "hat", "price": 5.0, "quantity": 1},
        ]
    )
    products = pd.DataFrame(
        [
            {"product_id": "P1", "rating": 4.0, "review_count": 10},
            {"product_id": "P2", "rating": 4.0, "review_count": 10},
        ]
    )

    out = generate_all_emails(
        client, assignments, customers, carts, products, progress=False
    )
    # Only one call should have been made (C0002).
    assert len(client.calls) == 1
    assert out.loc[0, "email_subject"] == "already done"
    assert out.loc[1, "email_subject"] == "Your cart is waiting"


def test_force_regenerates_all():
    client = FakeClient()
    assignments = pd.DataFrame(
        {
            "customer_id": ["C0001"],
            "derived_pool": ["evening_workday"],
            "assigned_variant": ["seasonal_discount"],
            "email_subject": ["already done"],
            "email_body": ["already done body"],
        }
    )
    customers = pd.DataFrame({"customer_id": ["C0001"], "name": ["A B"]})
    carts = pd.DataFrame(
        [{"cart_id": "K1", "customer_id": "C0001", "product_id": "P1",
          "product_name": "X", "category": "jacket", "price": 10.0, "quantity": 1}]
    )
    products = pd.DataFrame([{"product_id": "P1", "rating": 4.0, "review_count": 10}])

    out = generate_all_emails(
        client, assignments, customers, carts, products, force=True, progress=False
    )
    assert len(client.calls) == 1
    assert out.loc[0, "email_subject"] == "Your cart is waiting"
