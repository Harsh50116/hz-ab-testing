"""CLI entrypoint for Phase 3: per-customer email generation via Llama 3.3."""
from __future__ import annotations

import argparse

import pandas as pd

from . import config
from .emails import generate_all_emails, write_enriched
from .llm import HyperbolicClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate personalized cart-abandonment emails per customer."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all emails, overwriting existing subject/body columns.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only generate emails for the first N customers (debug/cost).",
    )
    args = parser.parse_args()

    print("→ Loading Phase 1 + 2 data...")
    assignments = pd.read_csv(config.ASSIGNMENTS_CSV)
    customers = pd.read_csv(config.CUSTOMERS_CSV)
    carts = pd.read_csv(config.CARTS_CSV)
    products = pd.read_csv(config.PRODUCTS_CSV)
    print(
        f"  assignments: {len(assignments)}  customers: {len(customers)}  "
        f"carts: {len(carts)}  products: {len(products)}"
    )

    client = HyperbolicClient()
    print(f"\n→ Generating emails (model={client.model})...")

    if args.limit:
        print(f"  (limited to first {args.limit}; remaining rows preserved)")
        head = assignments.head(args.limit)
        tail = assignments.iloc[args.limit:]
        enriched_head = generate_all_emails(
            client, head, customers, carts, products, force=args.force,
        )
        # Preserve untouched tail, ensuring email columns exist.
        tail = tail.copy()
        for col in ("email_subject", "email_body"):
            if col not in tail.columns:
                tail[col] = None
        enriched = pd.concat([enriched_head, tail], ignore_index=True)
    else:
        enriched = generate_all_emails(
            client, assignments, customers, carts, products, force=args.force,
        )

    print("\n→ Writing enriched ab_assignments.csv...")
    write_enriched(enriched)

    # Summary
    per_variant = (
        enriched.groupby("assigned_variant")["email_subject"]
        .apply(lambda s: s.notna().sum())
        .to_dict()
    )
    print(f"\nEmails generated per variant: {per_variant}")
    print("\nSample — first 3 emails:")
    for _, row in enriched.head(3).iterrows():
        print(f"\n  [{row['customer_id']} / {row['assigned_variant']}]")
        print(f"  subject: {row['email_subject']}")
        body = str(row["email_body"])
        preview = body if len(body) <= 200 else body[:200] + "..."
        print(f"  body:    {preview}")
    print("\nDone.")


if __name__ == "__main__":
    main()
