"""Central knobs for the whole engine.

Everything an owner might tune lives here or, per-SKU, in the products CSV.
The values here are sensible defaults that work out of the box; any product
can override lead time, costs, MOQ, etc. through its own row in products.
"""
from __future__ import annotations

import os

# ── Forecast horizon ───────────────────────────────────────────────
# How far ahead we forecast. By default we look one lead time ahead,
# but we never forecast on a horizon shorter than this floor.
MIN_FORECAST_HORIZON_DAYS = 14
BACKTEST_FOLDS = 4          # walk-forward folds in the accuracy report
BACKTEST_MIN_TRAIN_DAYS = 56  # need at least this much history to backtest a SKU

# ── Service levels by ABC class ────────────────────────────────────
# The few SKUs that drive most revenue get a high service level (and a
# bigger safety buffer); the long tail gets a leaner one.
SERVICE_LEVEL = {"A": 0.98, "B": 0.95, "C": 0.90}

# Z-scores matching the service levels above (one-sided normal).
Z_SCORE = {"A": 2.05, "B": 1.65, "C": 1.28}

# Review cadence in days by class — how often we re-evaluate a SKU.
REVIEW_PERIOD_DAYS = {"A": 7, "B": 14, "C": 28}

# ── ABC revenue cut points (cumulative share of revenue) ───────────
ABC_A_CUTOFF = 0.80   # top 80% of revenue
ABC_B_CUTOFF = 0.95   # next 15% (cumulative 95%)

# ── XYZ variability cut points (coefficient of variation) ──────────
XYZ_X_CUTOFF = 0.5    # CoV <= 0.5  -> stable
XYZ_Y_CUTOFF = 1.0    # 0.5 < CoV <= 1.0 -> medium; above -> erratic

# ── Cost / inventory defaults (overridable per SKU in products.csv) ─
DEFAULT_LEAD_TIME_DAYS = 7
DEFAULT_LEAD_TIME_STD = 2.0
DEFAULT_HOLDING_RATE = 0.25     # 25%/yr carrying cost of inventory value
DEFAULT_ORDER_COST = 50.0       # fixed cost to place one purchase order
DEFAULT_MOQ = 1                 # minimum order quantity

# ── Risk thresholds ────────────────────────────────────────────────
ANOMALY_SIGMA = 3.0             # flag demand residuals beyond this many sigma
COLD_START_MIN_DAYS = 28        # below this history, treat SKU as cold-start

# ── NIM / LLM ──────────────────────────────────────────────────────
NIM_URL = os.getenv("NIM_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
NIM_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct")
NIM_API_KEY = os.getenv("NIM_API_KEY", "")

FALLBACK_URL = os.getenv("FALLBACK_URL", "")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o-mini")
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", "")

# ── Paths ──────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
CHART_DIR = os.path.join(REPO_ROOT, "charts_out")
SAMPLE_ORDERS = os.path.join(DATA_DIR, "sample_orders.csv")
SAMPLE_PRODUCTS = os.path.join(DATA_DIR, "sample_products.csv")
