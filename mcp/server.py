"""MCP server: expose the engine as tools so you (or any MCP client, e.g. Claude)
can ask about a store in plain language and it calls the math.

  get_reorder_plan()          this week's plan with reasons
  forecast_sku(sku)           the demand path and band for one product
  explain_sku(sku)            why this product is or is not on the list
  simulate(sku, scenario)     what changes if lead time or demand shifts

Run it:  python mcp/server.py        (stdio transport)
Point your MCP client at this command. Uses the sample CSVs by default; set
STOCKPILOT_ORDERS / STOCKPILOT_PRODUCTS env vars to use other data.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import config, data as data_mod
from engine.pipeline import run_pipeline, sku_detail

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    raise SystemExit("Install the MCP SDK first:  pip install mcp")

mcp = FastMCP("stockpilot")

ORDERS = os.getenv("STOCKPILOT_ORDERS", config.SAMPLE_ORDERS)
PRODUCTS = os.getenv("STOCKPILOT_PRODUCTS", config.SAMPLE_PRODUCTS)

_cache: dict = {}


def _result():
    if "result" not in _cache:
        _cache["result"] = run_pipeline(ORDERS, PRODUCTS, use_llm=False, render_charts=False)
    return _cache["result"]


@mcp.tool()
def get_reorder_plan() -> dict:
    """Return this week's reorder plan: the SKUs that need an order, with
    quantity, days of cover, stockout risk, supplier, and reason codes."""
    r = _result()
    action = [row for row in r["plan"] if row["order_qty"] > 0]
    return {"reorder_count": r["reorder_count"],
            "total_order_value": r["total_order_value"],
            "summary": r["reasoning"]["summary"],
            "orders": action,
            "purchase_orders": r["reasoning"]["purchase_orders"]}


@mcp.tool()
def forecast_sku(sku: str) -> dict:
    """The demand path, mean, variability, and 90% band for one product."""
    r = _result()
    products = data_mod.load_products(PRODUCTS)
    detail = sku_detail(r["_series"], products, sku)
    return {"sku": sku, "forecast": detail["forecast"]}


@mcp.tool()
def explain_sku(sku: str) -> dict:
    """Why this product is, or is not, on the reorder list."""
    r = _result()
    row = next((p for p in r["plan"] if p["sku"] == sku), None)
    if row is None:
        return {"sku": sku, "found": False, "reason": "unknown SKU"}
    on_list = row["order_qty"] > 0
    why = (f"On hand {row['on_hand']} gives {row['days_of_cover']} days of cover "
           f"against a {row['lead_time_days']:.0f}-day lead time; stockout "
           f"probability {row['stockout_probability']:.0%}. ")
    why += ("Ordering %d units. " % row["order_qty"]) if on_list else "No order needed. "
    why += "Drivers: " + ", ".join(row["reason_codes"]) + "."
    return {"sku": sku, "found": True, "on_reorder_list": on_list,
            "order_qty": row["order_qty"], "explanation": why, "detail": row}


@mcp.tool()
def simulate(sku: str, lead_time_days: float | None = None,
            demand_multiplier: float = 1.0) -> dict:
    """What-if: recompute one SKU's policy and risk under a different lead time
    or a demand shift (e.g. demand_multiplier=1.5 for a 50% spike)."""
    from engine import policy as policy_mod, risk as risk_mod
    r = _result()
    products = data_mod.load_products(PRODUCTS)
    detail = sku_detail(r["_series"], products, sku)
    prod = detail["product"]
    fc = detail["forecast"]

    lead = float(lead_time_days) if lead_time_days else float(prod["lead_time_days"])
    dmean = fc["demand_mean"] * demand_multiplier
    dstd = fc["demand_std"] * demand_multiplier
    from engine.segment import segment as seg_fn
    seg = seg_fn(products, r["_series"]).set_index("sku")
    z = float(seg.loc[sku, "z"]) if sku in seg.index else config.Z_SCORE["C"]

    prod2 = dict(prod)
    prod2["lead_time_days"] = lead
    pol = policy_mod.policy_for_sku(dmean, dstd, prod2, z)
    rk = risk_mod.risk_for_sku(prod2, dmean, dstd, lead)
    return {"sku": sku, "scenario": {"lead_time_days": lead, "demand_multiplier": demand_multiplier},
            "policy": pol, "risk": rk}


if __name__ == "__main__":
    mcp.run()
