"""Phase 3: generate personalized cart-abandonment emails per customer.

For each customer, we take their assigned variant + cart contents and ask the
LLM to write a subject line and body following one of three strategy templates.
Output is enriched back into ab_assignments.csv with two new columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from . import config
from .llm import HyperbolicClient

# ---------------------------------------------------------------------------
# System prompt — brand voice rules, shared across all variants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""You are a copywriter for {config.BRAND_NAME}, a DTC winter apparel brand.

Write cart-abandonment recovery emails. Follow these rules:
- Friendly, warm, conversational tone. No exclamation stacking, no ALL CAPS.
- Body length: {config.EMAIL_WORD_MIN}-{config.EMAIL_WORD_MAX} words.
- Use the customer's first name once, near the opening.
- Reference the specific item(s) in their cart by name.
- End with the CTA placeholder exactly: {config.EMAIL_CTA}
- Sign off: "— The {config.BRAND_NAME} Team"
- Return ONLY valid JSON: {{"subject": "...", "body": "..."}}
- No markdown, no code fences, no commentary outside the JSON.
"""


# ---------------------------------------------------------------------------
# Per-variant user-prompt builders
# ---------------------------------------------------------------------------

def _format_cart_items(items: list[dict]) -> str:
    lines = []
    for it in items:
        lines.append(
            f"- {it['product_name']} ({it['category']}) — ${it['price']:.2f}"
        )
    return "\n".join(lines)


def _format_social_proof(items: list[dict]) -> str:
    lines = []
    for it in items:
        lines.append(
            f"- {it['product_name']}: rated {it['rating']}/5 by "
            f"{it['review_count']} customers"
        )
    return "\n".join(lines)


def _seasonal_discount_prompt(first_name: str, items: list[dict]) -> str:
    return f"""Customer first name: {first_name}
Items left in cart:
{_format_cart_items(items)}

Strategy: SEASONAL DISCOUNT.
- Frame the email around the current winter season and the winter collection.
- Offer exactly {config.DISCOUNT_PCT}% off, expiring in {config.DISCOUNT_EXPIRY}.
- Create gentle urgency tied to the discount expiration, not stock.
- Do NOT mention a discount code; just state the offer.
"""


def _urgency_prompt(first_name: str, items: list[dict]) -> str:
    return f"""Customer first name: {first_name}
Items left in cart:
{_format_cart_items(items)}

Strategy: URGENCY / SCARCITY.
- Do NOT offer any discount.
- Frame around limited availability or cart expiration.
- Suggest their item may not be held much longer.
- Keep it honest — no fake countdowns or made-up stock numbers.
"""


def _personalized_rec_prompt(first_name: str, items: list[dict]) -> str:
    return f"""Customer first name: {first_name}
Items left in cart with social proof:
{_format_social_proof(items)}

Strategy: PERSONALIZED RECOMMENDATION / SOCIAL PROOF.
- Do NOT offer any discount.
- Highlight the product rating and number of reviewers.
- Emphasize why other customers love the specific item(s).
- Make it feel like a warm nudge based on what others chose.
"""


_PROMPT_BUILDERS = {
    "seasonal_discount": _seasonal_discount_prompt,
    "urgency": _urgency_prompt,
    "personalized_rec": _personalized_rec_prompt,
}


# ---------------------------------------------------------------------------
# LLM interface abstraction (makes testing easy)
# ---------------------------------------------------------------------------

class _ChatClient(Protocol):
    def chat_json(self, system: str, user: str, **kwargs) -> dict: ...


@dataclass
class GeneratedEmail:
    subject: str
    body: str


def generate_email(
    client: _ChatClient,
    variant: str,
    first_name: str,
    items: list[dict],
) -> GeneratedEmail:
    """Generate a single email for one customer."""
    if variant not in _PROMPT_BUILDERS:
        raise ValueError(f"Unknown variant: {variant!r}")
    user_prompt = _PROMPT_BUILDERS[variant](first_name, items)
    data = client.chat_json(_SYSTEM_PROMPT, user_prompt)
    return GeneratedEmail(subject=str(data["subject"]), body=str(data["body"]))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _first_name(full_name: str) -> str:
    return full_name.split()[0]


def _cart_items_for(carts: pd.DataFrame, customer_id: str) -> list[dict]:
    rows = carts[carts["customer_id"] == customer_id]
    return rows[
        ["product_id", "product_name", "category", "price", "quantity"]
    ].to_dict(orient="records")


def _enrich_with_ratings(items: list[dict], products: pd.DataFrame) -> list[dict]:
    prod_lookup = products.set_index("product_id")[
        ["rating", "review_count"]
    ].to_dict(orient="index")
    for it in items:
        p = prod_lookup.get(it["product_id"], {})
        it["rating"] = p.get("rating", 0.0)
        it["review_count"] = p.get("review_count", 0)
    return items


def generate_all_emails(
    client: _ChatClient,
    assignments: pd.DataFrame,
    customers: pd.DataFrame,
    carts: pd.DataFrame,
    products: pd.DataFrame,
    *,
    force: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Generate emails for every row in `assignments`.

    Returns a DataFrame with email_subject/email_body columns added.
    Rows already containing non-empty email_subject are skipped unless force=True.
    """
    # Ensure the two target columns exist.
    out = assignments.copy()
    if "email_subject" not in out.columns:
        out["email_subject"] = None
    if "email_body" not in out.columns:
        out["email_body"] = None

    name_lookup = dict(zip(customers["customer_id"], customers["name"]))

    total = len(out)
    for i, row in out.iterrows():
        already = (
            isinstance(row["email_subject"], str) and row["email_subject"].strip()
        )
        if already and not force:
            continue

        cid = row["customer_id"]
        first_name = _first_name(name_lookup[cid])
        items = _cart_items_for(carts, cid)
        items = _enrich_with_ratings(items, products)
        variant = row["assigned_variant"]

        if progress:
            print(
                f"  [{i + 1:>3}/{total}] {cid} ({variant}) ...",
                end="",
                flush=True,
            )
        for attempt in range(3):
            try:
                email = generate_email(client, variant, first_name, items)
                out.at[i, "email_subject"] = email.subject
                out.at[i, "email_body"] = email.body
                if progress:
                    print(" ok")
                break
            except Exception as exc:  # noqa: BLE001
                if attempt < 2:
                    if progress:
                        print(f" retry ({exc})", end="", flush=True)
                else:
                    if progress:
                        print(f" FAILED: {exc}")
                    raise

    return out


def write_enriched(df: pd.DataFrame) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.ASSIGNMENTS_CSV, index=False)
