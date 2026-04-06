# A/B Email Testing Pipeline

An end-to-end pipeline for optimizing cart-abandonment email campaigns using customer segmentation and A/B variant assignment.

---

## Pipeline

```
Phase 1 — Synthetic data generation
   customers • carts • engagement events • product catalog
                │
                ▼
Phase 2 — Segment + assign
   derive timing pools from engagement data
   assign variants within each pool (33/33/34 split)
                │
                ▼
Phase 3 — Personalized email generation
   per-customer LLM email based on cart + variant strategy
                │
                ▼
data/ab_assignments.csv (ready-to-send emails)
```

---

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
echo "HYPERBOLIC_API_KEY=..." > .env

.venv/bin/python scripts/generate_data.py          # Phase 1
.venv/bin/python scripts/segment_and_assign.py     # Phase 2
.venv/bin/python scripts/generate_emails.py        # Phase 3

# Dashboard
.venv/bin/streamlit run app.py

# Tests
.venv/bin/pytest tests/ -q
```

---

## Tech

- Python 3.10+, pydantic, pandas
- Hyperbolic API (Llama-3.3-70B-Instruct) for email generation
- Streamlit + Plotly for the dashboard
- 30 tests across all three phases
