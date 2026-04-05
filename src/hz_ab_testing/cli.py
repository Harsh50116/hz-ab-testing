"""CLI entrypoint for Phase 1 data generation."""
from __future__ import annotations

import argparse

from . import config
from .catalog import load_or_generate_catalog, load_or_generate_names
from .generate import generate_all, write_all
from .llm import HyperbolicClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic customer/cart/engagement data for Hazel A/B demo."
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Regenerate product catalog and name pool via LLM, overwriting caches.",
    )
    parser.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED, help="Random seed."
    )
    args = parser.parse_args()

    client = HyperbolicClient()
    print("→ Loading product catalog...")
    products = load_or_generate_catalog(client, force=args.force_llm)
    print(f"  catalog: {len(products)} SKUs")

    print("→ Loading name pool...")
    names = load_or_generate_names(client, force=args.force_llm)
    print(f"  names:   {len(names)} names")

    print(f"→ Generating customers/carts/engagement (seed={args.seed})...")
    result = generate_all(products, names, seed=args.seed)
    print(
        f"  customers: {len(result['customers'])}  "
        f"carts: {len(result['carts'])}  "
        f"events: {len(result['events'])}"
    )

    print("→ Writing CSVs to data/...")
    write_all(result)

    # Quick distribution summary
    from collections import Counter
    seg_counts = Counter(c.segment for c in result["customers"])
    pool_counts = Counter(c.preferred_send_time for c in result["customers"])
    print(f"\nSegments: {dict(seg_counts)}")
    print(f"Pools:    {dict(pool_counts)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
