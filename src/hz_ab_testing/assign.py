"""Phase 2: assign each customer to one of three email variants.

Within each derived pool, customers are split ~33/33/34 across the variants
in config.VARIANTS. Assignment is seeded and shuffled *within* each pool so
that the same customer always gets the same variant across runs, regardless
of pool size.
"""
from __future__ import annotations

import random

import pandas as pd

from . import config


def assign_variants(
    classified: pd.DataFrame,
    *,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Assign a variant to each customer based on their derived pool.

    Returns DataFrame with columns: customer_id, derived_pool, assigned_variant.
    """
    rng = random.Random(seed)
    variants = config.VARIANTS
    n_variants = len(variants)

    out_rows = []
    # Stable pool ordering so the RNG consumption order is deterministic.
    for pool in sorted(classified["derived_pool"].unique()):
        in_pool = classified[classified["derived_pool"] == pool].copy()
        # Sort by customer_id for determinism, then shuffle with seeded RNG.
        in_pool = in_pool.sort_values("customer_id").reset_index(drop=True)
        order = list(range(len(in_pool)))
        rng.shuffle(order)

        for position, idx in enumerate(order):
            variant = variants[position % n_variants]
            row = in_pool.iloc[idx]
            out_rows.append(
                {
                    "customer_id": row["customer_id"],
                    "derived_pool": pool,
                    "assigned_variant": variant,
                }
            )

    return (
        pd.DataFrame(out_rows)
        .sort_values("customer_id")
        .reset_index(drop=True)
    )


def write_assignments(assignments: pd.DataFrame) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(config.ASSIGNMENTS_CSV, index=False)
