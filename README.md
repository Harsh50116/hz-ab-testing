# Hazel A/B Email Testing — Demo

A demo pipeline that picks the right cart-abandonment email for each customer, using the three data layers Hazel already sits on (Shopify + Klaviyo + Faraday).

Built independently to show how Hazel's agent could turn its read-only insights into a first **action** capability.

---

## The problem

DTC brands send cart-recovery emails through Klaviyo, but pick strategy by gut. They don't know which email works for which customer — because the data is spread across three systems that don't talk:

| Layer | Source | What it has |
|---|---|---|
| First-party | Shopify | cart, orders, catalog |
| Channel | Klaviyo | email engagement (opens, clicks, timing) |
| Consumer | Faraday | demographics, income, household |

Hazel has all three. This demo shows what you can do once they're joined.

---

## The pipeline

```
  Shopify + Klaviyo + Faraday (simulated)
                │
                ▼
  Phase 1 — Data generation
     100 customers • carts • 790 engagement events • catalog
                │
                ▼
  Phase 2 — Segment + assign
     derive timing pool from raw events (95% accuracy)
     assign one of 3 variants within each pool (33/33/34)
                │
                ▼
  Phase 3 — Personalized email generation
     per-customer LLM email using cart items + variant strategy
                │
                ▼
  data/ab_assignments.csv (100 ready-to-send emails)
```

---

## Key design choices

**Timing pool is derived, not labeled.** We store raw `open`/`click` events per customer. Phase 2 classifies each customer from the event stream alone (weekday evening vs. weekend midday dominance). This is what Hazel's agent would actually do — reason from the Klaviyo feed, not from a pre-computed column.

**Classifier is conservative.** When the signal is weak (few events, or mixed pattern), the customer goes to Pool C (`unknown`) rather than being misclassified. In the run on our synthetic data, 5 customers landed in Pool C instead of their true pool, but **zero** were sent to the wrong pool.

**New customers auto-route to Pool C.** They lack engagement history, so the agent says "I don't have enough signal" rather than guessing. That's the honest answer a marketer needs.

**Three variants, three psychological levers:**
- `seasonal_discount` — 20% off, 48h expiry, winter framing
- `urgency` — no discount, scarcity framing
- `personalized_rec` — no discount, product rating + review count as social proof

---

## Results on 100 synthetic customers

**Pool derivation accuracy vs. ground truth:** 95%

| Segment | Accuracy |
|---|---|
| new | 100% (all correctly → Pool C) |
| returning | 94.7% |
| vip | 90.5% |

**Confusion matrix** (rows = true pool, cols = derived):
```
                evening   unknown   weekend
evening           45         1         0
unknown            0        22         0
weekend            0         4        28
```

All 5 misses are conservative (true pool → unknown). No wrong-pool assignments.

**Variant distribution per pool:**
| Pool | seasonal_discount | urgency | personalized_rec |
|---|---|---|---|
| evening_workday (45) | 15 | 15 | 15 |
| weekend_noon (28) | 10 | 9 | 9 |
| unknown (27) | 9 | 9 | 9 |

---

## Example output

```
Customer: C0001 (Oliver Brown, Buffalo NY, vip)
Pool:     evening_workday (derived from 14 past events)
Variant:  personalized_rec

Subject: you left something behind, oliver

oliver, it looks like you left some great items in your cart.
our touchscreen gloves, rated 4.4/5 by 250 customers, are a
favorite for their warmth and functionality. you also had the
thermal insulated boots, loved by 421 customers with a 4.5/5
rating... [Complete Your Purchase] — The Hazel Apparel Team
```

---

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
echo "HYPERBOLIC_API_KEY=..." > .env

# Full pipeline (LLM calls cached — reruns are instant)
.venv/bin/python scripts/generate_data.py          # Phase 1
.venv/bin/python scripts/segment_and_assign.py     # Phase 2
.venv/bin/python scripts/generate_emails.py        # Phase 3

# Tests
.venv/bin/pytest tests/ -q
```

Seeded at 42 — all outputs are reproducible.

---

## Tech

- **Python** 3.10+, **pydantic** for schemas, **pandas** for tables
- **Hyperbolic** API with **Llama-3.3-70B-Instruct** for catalog + names + emails
- **CSV** storage (portable, inspectable)
- 30 tests covering generation, classification, assignment, email prompting

---

## What this demo deliberately doesn't do

- No analytics / dashboards
- No send tracking or conversion attribution
- No autonomous strategy discovery — strategies are hand-written

These are the natural next steps, not gaps.

---

## v2 — what the real Hazel agent would do

Today, the three variants and the timing-pool rule are **written by hand**. In production:

1. Agent analyzes historical recovery data across all three data layers
2. Discovers which `(pool, segment, variant)` combos actually drove conversion
3. Generates new strategies based on those patterns (not just three hardcoded ones)
4. Continuously re-assigns and re-learns

Same pipeline shape. Smarter brain. The CSV output row (`customer_id, pool, variant, subject, body`) becomes the **action the agent takes**, and the conversion outcome becomes the **learning signal** that closes the loop.

That's how Hazel goes from read-only insights to a first-class action surface.

---

## Files

```
data/
  products.csv              22 winter apparel SKUs (LLM-generated)
  names.json                120 US name pool
  customers.csv             100 customers × 11 attributes
  carts.csv                 178 cart-items across 100 carts
  engagement_events.csv     790 open/click events
  ab_assignments.csv        100 rows: pool + variant + email ← the demo output

src/hz_ab_testing/
  config.py                 seeds, distributions, paths
  models.py                 pydantic schemas
  llm.py                    Hyperbolic client
  catalog.py                Phase 1: LLM-generated catalog + names
  generate.py               Phase 1: seeded customer/cart/event sampling
  segment.py                Phase 2: rule-based timing pool classifier
  assign.py                 Phase 2: within-pool variant assignment
  emails.py                 Phase 3: per-customer email generation
```
