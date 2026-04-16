"""Agentic email generation: LLM decides strategy + writes the email."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from . import config
from .context import CustomerContext


class _ChatClient(Protocol):
    def chat_json(self, system: str, user: str, **kwargs) -> dict: ...


@dataclass
class AgenticEmail:
    subject: str
    body: str
    strategy: str
    reasoning: str


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""You are an expert email strategist for {config.BRAND_NAME}, a DTC winter apparel brand.

You will receive the full profile of a customer who abandoned their cart. Your job:
1. Analyze the customer context to decide the BEST email strategy.
2. Write a personalized cart-abandonment email using that strategy.

Available strategies (pick exactly one):
- **discount**: Offer a time-limited percentage discount. Best for price-sensitive or lapsed customers.
- **urgency**: Emphasize limited stock or cart expiration. No discount. Best for engaged customers with high intent.
- **social_proof**: Highlight ratings, reviews, and popularity. No discount. Best for newer customers or high-rated items.
- **personalized_rec**: Warm, personal nudge based on their interests and browsing. No discount. Best for loyal or lifestyle-aligned customers.

Brand voice rules:
- Friendly, warm, conversational tone. No exclamation stacking, no ALL CAPS.
- Body length: 60-150 words.
- You MUST use the customer's EXACT first name as provided — do NOT invent or substitute a different name. Use it once, near the opening.
- Reference the specific item(s) in their cart by name.
- End the body with the CTA placeholder exactly: {config.EMAIL_CTA}
- Sign off: "— The {config.BRAND_NAME} Team"

Return ONLY valid JSON with these keys:
{{
  "strategy": "discount | urgency | social_proof | personalized_rec",
  "reasoning": "2-3 sentences explaining WHY you chose this strategy for this specific customer",
  "subject": "the email subject line",
  "body": "the email body"
}}

No markdown, no code fences, no commentary outside the JSON."""


def _build_user_prompt(ctx: CustomerContext) -> str:
    cart_lines = []
    for it in ctx.cart_items:
        line = f"- {it['product_name']} ({it['category']}) — ${it['price']:.2f} x{it['quantity']}"
        if it.get("rating"):
            line += f" | rated {it['rating']}/5 by {it['review_count']} customers"
        cart_lines.append(line)

    interests_str = ", ".join(ctx.lifestyle_interests) if ctx.lifestyle_interests else "none identified"

    return f"""Customer first name: {ctx.first_name}
(IMPORTANT: Use EXACTLY this name in the email — "{ctx.first_name}" — do not change it.)

FIRST-PARTY BUSINESS DATA:
- Cart contents:
{chr(10).join(cart_lines)}
- Cart total: ${ctx.cart_total:.2f}
- Customer lifetime value (CLV): ${ctx.clv:.2f} ({ctx.segment} tier)
- Total past orders: {ctx.total_orders}
- Last purchase: {ctx.last_purchase_days_ago} days ago
- Return rate: {ctx.return_rate}%

MARKETING / CHANNEL DATA:
- Preferred send time: {ctx.preferred_send_time}
- Past engagement: {ctx.engagement_summary}
- Acquisition channel: {ctx.acquisition_channel}

THIRD-PARTY DATA:
- Income bracket: {ctx.income_bracket}
- Location: {ctx.location}
- Lifestyle interests: {interests_str}

Analyze this customer and generate the email."""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_agentic_email(
    client: _ChatClient,
    ctx: CustomerContext,
) -> AgenticEmail:
    """Send full customer context to LLM; it picks the strategy and writes the email."""
    user_prompt = _build_user_prompt(ctx)
    data = client.chat_json(_SYSTEM_PROMPT, user_prompt)
    return AgenticEmail(
        subject=str(data["subject"]),
        body=str(data["body"]),
        strategy=str(data["strategy"]),
        reasoning=str(data["reasoning"]),
    )
