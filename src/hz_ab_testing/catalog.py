"""LLM-powered generation of the winter apparel product catalog and name pool.

Both outputs are cached to disk so repeated runs are deterministic and cheap.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config
from .llm import HyperbolicClient
from .models import Product

# ---------------------------------------------------------------------------
# Product catalog
# ---------------------------------------------------------------------------

_CATALOG_SYSTEM = (
    "You are a product catalog generator for a DTC winter apparel brand. "
    "Respond only with valid JSON matching the requested schema."
)

_CATALOG_USER_TEMPLATE = """Generate a catalog of {n} distinct winter apparel SKUs for a US DTC brand.

Requirements:
- Mix across these categories: jacket, sweater, coat, boots, scarf, gloves, hat, thermal, pants, socks.
- Each item: a realistic, brandable product name (2-5 words, no brand prefix).
- Prices in USD, reflecting the category (e.g., socks $15-35, jackets $120-320, coats $180-450).
- Rating between 3.8 and 4.9 (one decimal).
- review_count between 20 and 1200.

Return JSON of the exact shape:
{{
  "products": [
    {{"name": "...", "category": "jacket", "price": 189.00, "rating": 4.6, "review_count": 312}},
    ...
  ]
}}

Return exactly {n} products. No commentary.
"""


def _generate_catalog_via_llm(client: HyperbolicClient, n: int) -> list[Product]:
    user = _CATALOG_USER_TEMPLATE.format(n=n)
    data = client.chat_json(_CATALOG_SYSTEM, user)
    raw_items = data["products"] if isinstance(data, dict) else data
    products: list[Product] = []
    for i, item in enumerate(raw_items[:n], start=1):
        products.append(
            Product(
                product_id=f"P{i:03d}",
                name=item["name"],
                category=item["category"],
                price=round(float(item["price"]), 2),
                rating=round(float(item["rating"]), 1),
                review_count=int(item["review_count"]),
            )
        )
    if len(products) < n:
        raise RuntimeError(
            f"LLM returned only {len(products)} products; expected {n}."
        )
    return products


def load_or_generate_catalog(
    client: HyperbolicClient | None = None,
    *,
    n: int = config.N_PRODUCTS,
    force: bool = False,
) -> list[Product]:
    """Return the product catalog, generating via LLM if not yet cached."""
    path: Path = config.PRODUCTS_CSV
    if path.exists() and not force:
        df = pd.read_csv(path)
        return [Product(**row) for row in df.to_dict(orient="records")]

    path.parent.mkdir(parents=True, exist_ok=True)
    client = client or HyperbolicClient()
    products = _generate_catalog_via_llm(client, n)
    pd.DataFrame([p.model_dump() for p in products]).to_csv(path, index=False)
    return products


# ---------------------------------------------------------------------------
# Name pool
# ---------------------------------------------------------------------------

_NAMES_SYSTEM = (
    "You generate realistic, diverse US customer name lists. "
    "Respond only with valid JSON."
)

_NAMES_USER_TEMPLATE = """Generate {n} realistic US customer full names (first + last).

Requirements:
- Ethnic and gender diversity reflecting a US consumer base.
- No duplicates.
- No celebrities or obviously fake names.

Return JSON:
{{"names": ["Jane Doe", "Marcus Chen", ...]}}

Return exactly {n} names. No commentary.
"""


def _generate_names_via_llm(client: HyperbolicClient, n: int) -> list[str]:
    user = _NAMES_USER_TEMPLATE.format(n=n)
    data = client.chat_json(_NAMES_SYSTEM, user)
    names = data["names"] if isinstance(data, dict) else data
    # dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    if len(out) < n:
        raise RuntimeError(f"LLM returned only {len(out)} unique names; expected {n}.")
    return out[:n]


def load_or_generate_names(
    client: HyperbolicClient | None = None,
    *,
    n: int = config.N_NAMES,
    force: bool = False,
) -> list[str]:
    """Return the name pool, generating via LLM if not yet cached."""
    path: Path = config.NAMES_JSON
    if path.exists() and not force:
        with open(path) as f:
            return json.load(f)["names"]

    path.parent.mkdir(parents=True, exist_ok=True)
    client = client or HyperbolicClient()
    names = _generate_names_via_llm(client, n)
    with open(path, "w") as f:
        json.dump({"names": names}, f, indent=2)
    return names
