"""Pull everything together into one reorder plan.

For each SKU that needs action we carry the recommended quantity, the reorder
point, the days of cover, the stockout risk, the supplier, and a short reason
code (low cover, anomaly, seasonal ramp). That frame is the input to the
reasoning layer, and on its own it is already a complete plan a person could
act on.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import config, policy, risk, segment as segment_mod
from .forecast import forecast_sku


def _reason_codes(row: dict, trailing_mean: float, forecast_mean: float) -> list:
    codes = []
    if row["below_lead_time"]:
        codes.append("low_cover")
    if row["stockout_probability"] >= 0.20:
        codes.append("stockout_risk")
    if row["anomaly"]:
        codes.append("anomaly")
    if trailing_mean > 0 and forecast_mean > 1.20 * trailing_mean:
        codes.append("seasonal_ramp")
    if not codes and row["order_qty"] > 0:
        codes.append("scheduled_topup")
    if not codes:
        codes.append("healthy")
    return codes


def build_plan(series: pd.DataFrame, products: pd.DataFrame,
               horizon: Optional[int] = None) -> dict:
    """Run forecast -> segment -> policy -> risk for every SKU and assemble the
    plan. Returns {"plan": DataFrame, "forecasts": {sku: fc}, "segments": df}.
    """
    seg = segment_mod.segment(products, series)
    seg_by_sku = seg.set_index("sku")
    prod_by_sku = products.set_index("sku")

    plan_rows = []
    forecasts = {}

    for sku, g in series.groupby("sku"):
        if sku not in prod_by_sku.index:
            continue
        product = prod_by_sku.loc[sku].to_dict()
        product["sku"] = sku
        s = seg_by_sku.loc[sku] if sku in seg_by_sku.index else None
        z = float(s["z"]) if s is not None else config.Z_SCORE["C"]

        lead = float(product["lead_time_days"])
        h = horizon or max(int(round(lead)), config.MIN_FORECAST_HORIZON_DAYS)

        fc = forecast_sku(g, h)
        forecasts[sku] = fc
        dmean, dstd = fc["demand_mean"], fc["demand_std"]

        pol = policy.policy_for_sku(dmean, dstd, product, z)

        # recent actuals vs the model's recent fit, for the anomaly flag
        recent_actual = g["units"].tail(14).values
        recent_fc = np.full(len(recent_actual), float(g["units"].tail(28).mean()))
        rk = risk.risk_for_sku(product, dmean, dstd, lead,
                               recent_actual, recent_fc, fc["resid_std"])

        trailing_mean = float(g["units"].tail(28).mean())
        row = {
            "sku": sku,
            "supplier": product["supplier"],
            "abc": s["abc"] if s is not None else "C",
            "xyz": s["xyz"] if s is not None else "Z",
            "segment": s["segment"] if s is not None else "CZ",
            "service_level": float(s["service_level"]) if s is not None else 0.90,
            "on_hand": int(product["on_hand"]),
            "on_order": int(product["on_order"]),
            "demand_mean": round(dmean, 2),
            "demand_std": round(dstd, 2),
            "model_used": fc["model_used"],
            "lead_time_days": lead,
            "safety_stock": pol["safety_stock"],
            "reorder_point": pol["reorder_point"],
            "eoq": pol["eoq"],
            "order_up_to": pol["order_up_to"],
            "order_qty": pol["order_qty"],
            "days_of_cover": rk["days_of_cover"],
            "stockout_probability": rk["stockout_probability"],
            "anomaly": rk["anomaly"],
            "below_lead_time": rk["below_lead_time"],
            "unit_cost": round(float(product["cost"]), 2),
            "order_value": round(pol["order_qty"] * float(product["cost"]), 2),
        }
        # attach the winning model's accuracy if present
        acc = fc.get("accuracy") or {}
        winner = acc.get("winner", "baseline")
        win_score = acc.get(winner if winner in acc else "baseline") or {}
        row["mae"] = win_score.get("mae")
        row["mape"] = win_score.get("mape")

        row["reason_codes"] = _reason_codes(row, trailing_mean, dmean)
        plan_rows.append(row)

    plan = pd.DataFrame(plan_rows)
    if not plan.empty:
        # rank: anything needing action first, by stockout risk then value
        plan["needs_action"] = plan["order_qty"] > 0
        plan = plan.sort_values(
            ["needs_action", "stockout_probability", "order_value"],
            ascending=[False, False, False]).reset_index(drop=True)

    return {"plan": plan, "forecasts": forecasts, "segments": seg}


def plan_records(plan: pd.DataFrame, action_only: bool = True) -> list:
    """The list-of-dicts handed to the reasoning layer (JSON-friendly)."""
    df = plan.copy()
    if action_only and "order_qty" in df.columns:
        df = df[df["order_qty"] > 0]
    keep = ["sku", "supplier", "segment", "order_qty", "reorder_point",
            "days_of_cover", "stockout_probability", "reason_codes",
            "unit_cost", "order_value", "demand_mean", "anomaly"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].to_dict(orient="records")
