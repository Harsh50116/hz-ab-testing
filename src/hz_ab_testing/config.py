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

# ---- New v2 fields ---------------------------------------------------------
# Last purchase recency (days ago) by segment
LAST_PURCHASE_RANGE_BY_SEGMENT = {
    "new": (30, 90),
    "returning": (5, 30),
    "vip": (1, 15),
}

# Return rate (%) by segment — (mean, std_dev) for normal distribution, clamped 0-100
RETURN_RATE_BY_SEGMENT = {
    "new": (5.0, 5.0),       # low / unknown history
    "returning": (12.0, 8.0),  # moderate
    "vip": (8.0, 6.0),        # generally lower, but varies
}

# Acquisition channel weights by segment
ACQUISITION_CHANNELS = ["organic_search", "paid_social", "referral", "email", "direct"]
ACQUISITION_WEIGHTS_BY_SEGMENT = {
    "new": [0.20, 0.40, 0.10, 0.15, 0.15],      # new skews paid social
    "returning": [0.30, 0.15, 0.25, 0.20, 0.10],  # returning skews organic + referral
    "vip": [0.20, 0.05, 0.30, 0.30, 0.15],        # vip skews referral + email
}

# Lifestyle interests — pick 1-3 per customer, segment-biased weights
LIFESTYLE_INTERESTS = [
    "outdoor_enthusiast", "fashion_forward", "budget_conscious",
    "athleisure", "sustainable_fashion", "minimalist", "streetwear",
]
LIFESTYLE_WEIGHTS_BY_SEGMENT = {
    "new": [0.15, 0.15, 0.25, 0.15, 0.10, 0.10, 0.10],       # new skews budget_conscious
    "returning": [0.18, 0.20, 0.10, 0.18, 0.15, 0.10, 0.09],  # returning skews fashion
    "vip": [0.12, 0.25, 0.03, 0.15, 0.20, 0.15, 0.10],        # vip skews fashion + sustainable
}

# ---- Timing behavior -------------------------------------------------------
# Window (days) over which engagement events + abandonment timestamps are spread.
HISTORY_WINDOW_DAYS = 90
ABANDONMENT_WINDOW_DAYS = 7

# Probability that an event lands in the customer's "preferred" window
# (vs. noise). Higher = clearer signal → easier to derive pool in Phase 2.
PREFERRED_WINDOW_PROB = 0.75

# ---- Phase 2: segmentation classifier --------------------------------------
# Customers with fewer than this many events lack enough signal → Pool C.
MIN_EVENTS_FOR_CLASSIFICATION = 3
# Share of events that must fall into a window for that window to "win".
POOL_DOMINANCE_THRESHOLD = 0.55
# Clicks count more than opens when inferring timing preference.
EVENT_WEIGHTS = {"open": 1.0, "click": 2.0}
# Weekday evening window (local time, but we treat UTC timestamps as-is for demo).
EVENING_HOURS = range(17, 22)    # 17:00–21:59
WEEKEND_NOON_HOURS = range(10, 14)  # 10:00–13:59

# ---- Phase 2: variant assignment -------------------------------------------
VARIANTS = ["seasonal_discount", "urgency", "personalized_rec"]
# 33/33/34 split — the "34" goes to the last variant.
ASSIGNMENTS_CSV = DATA_DIR / "ab_assignments.csv"

# ---- Phase 3: email generation ---------------------------------------------
BRAND_NAME = "Hazel Apparel"
DISCOUNT_PCT = 20
DISCOUNT_EXPIRY = "48 hours"
EMAIL_WORD_MIN = 60
EMAIL_WORD_MAX = 100
EMAIL_CTA = "[Complete Your Purchase]"

# ---- LLM -------------------------------------------------------------------
HYPERBOLIC_BASE_URL = "https://api.hyperbolic.xyz/v1"
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 4096
