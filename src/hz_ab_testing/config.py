"""Central configuration: seeds, distributions, paths."""
from __future__ import annotations

from pathlib import Path

# ---- Reproducibility -------------------------------------------------------
RANDOM_SEED = 42

# ---- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PRODUCTS_CSV = DATA_DIR / "products.csv"
NAMES_JSON = DATA_DIR / "names.json"
CUSTOMERS_CSV = DATA_DIR / "customers.csv"
CARTS_CSV = DATA_DIR / "carts.csv"
ENGAGEMENT_CSV = DATA_DIR / "engagement_events.csv"

# ---- Population sizes ------------------------------------------------------
N_CUSTOMERS = 100
N_PRODUCTS = 22
N_NAMES = 120  # generate extra, sample from pool

# ---- Distributions ---------------------------------------------------------
# Customer segments
SEGMENT_WEIGHTS = {"returning": 0.60, "new": 0.25, "vip": 0.15}

# Target send-timing pool distribution (derived post-hoc from engagement)
# Pool C (unknown) is skewed toward `new` customers since they lack history.
POOL_WEIGHTS = {
    "evening_workday": 0.40,
    "weekend_noon": 0.35,
    "unknown": 0.25,
}

# Cart sizes
CART_SIZE_WEIGHTS = {1: 0.50, 2: 0.35, 3: 0.15}

# Engagement events per customer by segment
ENGAGEMENT_RANGE_BY_SEGMENT = {
    "returning": (5, 15),
    "vip": (8, 15),
    "new": (0, 2),
}

# Average order value ranges ($) by segment
AOV_RANGE_BY_SEGMENT = {
    "new": (40, 120),
    "returning": (80, 220),
    "vip": (180, 500),
}

# Total past orders by segment
TOTAL_ORDERS_RANGE_BY_SEGMENT = {
    "new": (0, 1),
    "returning": (2, 12),
    "vip": (8, 40),
}

# ---- Demographics (Faraday-style) ------------------------------------------
AGE_RANGES = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_WEIGHTS = [0.12, 0.28, 0.26, 0.18, 0.11, 0.05]

INCOME_BRACKETS = ["<50k", "50-75k", "75-100k", "100-150k", "150k+"]
INCOME_WEIGHTS = [0.18, 0.25, 0.23, 0.20, 0.14]

# US cold-weather states (NE + Midwest) — keeps winter theme believable.
COLD_STATES = [
    ("Boston", "MA"),
    ("New York", "NY"),
    ("Philadelphia", "PA"),
    ("Pittsburgh", "PA"),
    ("Buffalo", "NY"),
    ("Burlington", "VT"),
    ("Portland", "ME"),
    ("Hartford", "CT"),
    ("Providence", "RI"),
    ("Chicago", "IL"),
    ("Minneapolis", "MN"),
    ("Milwaukee", "WI"),
    ("Detroit", "MI"),
    ("Cleveland", "OH"),
    ("Columbus", "OH"),
    ("Indianapolis", "IN"),
    ("Des Moines", "IA"),
    ("Madison", "WI"),
    ("Ann Arbor", "MI"),
    ("Rochester", "NY"),
]

# ---- Timing behavior -------------------------------------------------------
# Window (days) over which engagement events + abandonment timestamps are spread.
HISTORY_WINDOW_DAYS = 90
ABANDONMENT_WINDOW_DAYS = 7

# Probability that an event lands in the customer's "preferred" window
# (vs. noise). Higher = clearer signal → easier to derive pool in Phase 2.
PREFERRED_WINDOW_PROB = 0.75

# ---- LLM -------------------------------------------------------------------
HYPERBOLIC_BASE_URL = "https://api.hyperbolic.xyz/v1"
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 4096
