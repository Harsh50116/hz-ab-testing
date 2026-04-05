"""CLI entrypoint for Phase 2: derive timing pools + assign email variants."""
from __future__ import annotations

import argparse

import pandas as pd

from . import config
from .assign import assign_variants, write_assignments
from .segment import classify_all, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment customers into timing pools and assign A/B variants."
    )
    parser.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED, help="Random seed."
    )
    args = parser.parse_args()

    print("→ Loading Phase 1 data...")
    customers = pd.read_csv(config.CUSTOMERS_CSV)
    events = pd.read_csv(config.ENGAGEMENT_CSV)
    print(f"  customers: {len(customers)}  events: {len(events)}")

    print("\n→ Classifying customers into timing pools...")
    classified = classify_all(customers, events)
    derived_counts = classified["derived_pool"].value_counts().to_dict()
    print(f"  derived pools: {derived_counts}")

    print("\n→ Evaluating against ground truth...")
    metrics = evaluate(classified, customers)
    print(f"  overall accuracy: {metrics['overall_accuracy']:.1%}")
    print(f"  by segment:")
    for seg, acc in metrics["accuracy_by_segment"].items():
        print(f"    {seg:10s} {acc:.1%}")
    print("\n  confusion matrix (rows=true, cols=derived):")
    print(metrics["confusion_matrix"].to_string())

    print(f"\n→ Assigning variants within pools (seed={args.seed})...")
    assignments = assign_variants(classified, seed=args.seed)
    dist = (
        assignments.groupby(["derived_pool", "assigned_variant"])
        .size()
        .unstack(fill_value=0)
    )
    print("  variant distribution per pool:")
    print(dist.to_string())

    print(f"\n→ Writing {config.ASSIGNMENTS_CSV.name}...")
    write_assignments(assignments)
    print("\nDone.")


if __name__ == "__main__":
    main()
