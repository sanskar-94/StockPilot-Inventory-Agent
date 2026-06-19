"""One entry point that runs the whole engine end to end.

The API, the MCP server, and the demo script all call run_pipeline so there is
exactly one path through the math. Give it order + product sources (CSV paths,
DataFrames, or records posted from n8n) and it returns the finished payload:
the ranked plan, the grouped purchase order, the summary, and chart paths.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from . import charts as charts_mod
from . import config, data
from .plan import build_plan, plan_records
from .reason import reason_over_plan


def run_pipeline(orders_source, products_source, *, use_llm: bool = True,
                 render_charts: bool = True, horizon: Optional[int] = None,
                 chart_dir: str = "") -> dict:
    """Forecast -> segment -> policy -> risk -> plan -> reason, then charts.

    Returns a JSON-friendly dict:
      plan            list of per-SKU plan rows (ranked)
      reorder_count   how many SKUs need an order
      total_order_value
      reasoning       {ranked_orders, purchase_orders, summary, _source}
      charts          {name: path}
      segments        list of ABC/XYZ rows
      generated_for   date span of the input data
    """
    orders = data.load_orders(orders_source)
    products = data.load_products(products_source)
    series = data.build_demand_series(orders)

    result = build_plan(series, products, horizon=horizon)
    plan = result["plan"]
    records = plan_records(plan, action_only=True)
    reasoning = reason_over_plan(records, use_llm=use_llm)

    charts = {}
    if render_charts:
        charts = charts_mod.render_all(series, result, out_dir=chart_dir or config.CHART_DIR)

    action = plan[plan["order_qty"] > 0] if not plan.empty else plan
    return {
        "plan": plan.to_dict(orient="records"),
        "reorder_count": int(len(action)),
        "total_order_value": round(float(action["order_value"].sum()) if not action.empty else 0.0, 2),
        "reasoning": reasoning,
        "charts": charts,
        "segments": result["segments"].to_dict(orient="records"),
        "generated_for": {
            "from": str(series["date"].min().date()) if not series.empty else None,
            "to": str(series["date"].max().date()) if not series.empty else None,
            "skus": int(series["sku"].nunique()),
        },
        # kept in-process for /sku drill-downs; not serialized to the client
        "_forecasts": result["forecasts"],
        "_series": series,
    }


def sku_detail(series: pd.DataFrame, products: pd.DataFrame, sku: str,
               horizon: Optional[int] = None) -> dict:
    """Forecast, policy, and risk for a single product — the /sku/{sku} view."""
    from .forecast import forecast_sku
    from . import policy as policy_mod, risk as risk_mod, segment as segment_mod
    import numpy as np

    g = series[series["sku"] == sku]
    if g.empty:
        raise KeyError(f"unknown sku {sku!r}")
    prod = products.set_index("sku").loc[sku].to_dict()
    prod["sku"] = sku

    seg = segment_mod.segment(products, series).set_index("sku")
    z = float(seg.loc[sku, "z"]) if sku in seg.index else config.Z_SCORE["C"]
    lead = float(prod["lead_time_days"])
    h = horizon or max(int(round(lead)), config.MIN_FORECAST_HORIZON_DAYS)

    fc = forecast_sku(g, h)
    pol = policy_mod.policy_for_sku(fc["demand_mean"], fc["demand_std"], prod, z)
    recent_actual = g["units"].tail(14).values
    recent_fc = np.full(len(recent_actual), float(g["units"].tail(28).mean()))
    rk = risk_mod.risk_for_sku(prod, fc["demand_mean"], fc["demand_std"], lead,
                               recent_actual, recent_fc, fc["resid_std"])
    return {"sku": sku, "segment": seg.loc[sku, "segment"] if sku in seg.index else "CZ",
            "forecast": fc, "policy": pol, "risk": rk, "product": prod}
