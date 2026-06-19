"""Dashboard-shaped read endpoints.

Mounted under /dashboard so they never collide with the engine's own /run and
/plan (which n8n depends on, in a different shape). These map the engine's
pipeline output into the contract the React dashboard expects. Point the
dashboard at `<engine-url>/dashboard` via VITE_API_BASE to go live.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from engine import config, data as data_mod
from engine.pipeline import run_pipeline, sku_detail

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# in-memory state for the dashboard view
_STATE: dict = {"result": None, "series": None, "products": None}
_APPROVALS: dict[str, str] = {}          # sku -> "approved" | "rejected"
_RUNS: list[dict] = []
OVERSTOCK_COVER_DAYS = 60


def _ensure() -> dict:
    """Run the pipeline on sample data once if nothing has been computed yet."""
    if _STATE["result"] is None:
        refresh(config.SAMPLE_ORDERS, config.SAMPLE_PRODUCTS)
    return _STATE["result"]


def refresh(orders_src, products_src, *, use_llm: bool = False) -> dict:
    result = run_pipeline(orders_src, products_src, use_llm=use_llm, render_charts=False)
    _STATE["result"] = result
    _STATE["series"] = result["_series"]
    _STATE["products"] = data_mod.load_products(products_src)
    # log a run snapshot
    plan = result["plan"]
    proposed = sum(1 for r in plan if r["order_qty"] > 0)
    approved = sum(1 for s in _APPROVALS.values() if s == "approved")
    rejected = sum(1 for s in _APPROVALS.values() if s == "rejected")
    _RUNS.insert(0, {"timestamp": dt.datetime.utcnow().isoformat() + "Z",
                     "proposed": proposed, "approved": approved,
                     "rejected": rejected, "sent": approved})
    return result


def _status(row: dict) -> str:
    if row["stockout_probability"] >= 0.5 or row["below_lead_time"]:
        return "stockout_risk"
    if row["order_qty"] > 0:
        return "reorder_soon"
    if row["days_of_cover"] > OVERSTOCK_COVER_DAYS:
        return "overstock"
    return "healthy"


def _trend(sku: str) -> list:
    series = _STATE["series"]
    g = series[series["sku"] == sku].tail(7)
    return [int(v) for v in g["units"].tolist()]


@router.get("/summary")
def summary() -> dict:
    r = _ensure()
    plan = r["plan"]
    statuses = [_status(row) for row in plan]
    capital = sum(row["on_hand"] * row["unit_cost"] for row in plan)
    excess = sum(row["on_hand"] * row["unit_cost"]
                 for row, s in zip(plan, statuses) if s == "overstock")
    mapes = [row["mape"] for row in plan if row.get("mape") is not None]
    return {
        "skus_tracked": len(plan),
        "reorder_now": sum(1 for row in plan if row["order_qty"] > 0),
        "stockout_risk": sum(1 for s in statuses if s == "stockout_risk"),
        "capital_in_stock": round(capital),
        "excess_inventory": round(excess),
        "forecast_mape": round(float(np.mean(mapes)), 1) if mapes else 0.0,
        "pending_approvals": sum(1 for row in plan
                                 if row["order_qty"] > 0 and row["sku"] not in _APPROVALS),
    }


@router.get("/plan")
def plan() -> list:
    r = _ensure()
    out = []
    for row in r["plan"]:
        if row["order_qty"] <= 0:
            continue
        out.append({
            "sku": row["sku"], "name": row["sku"],
            "recommended_qty": row["order_qty"], "on_hand": row["on_hand"],
            "reorder_point": row["reorder_point"], "days_of_cover": row["days_of_cover"],
            "stockout_risk": row["stockout_probability"], "status": _status(row),
            "supplier": row["supplier"], "unit_cost": row["unit_cost"],
            "reason": ", ".join(row["reason_codes"]),
            "approval": _APPROVALS.get(row["sku"], "pending"),
        })
    return out


@router.get("/inventory")
def inventory() -> list:
    r = _ensure()
    return [{
        "sku": row["sku"], "name": row["sku"], "on_hand": row["on_hand"],
        "on_order": row["on_order"], "days_of_cover": row["days_of_cover"],
        "reorder_point": row["reorder_point"], "status": _status(row),
        "abc_class": row["abc"], "trend": _trend(row["sku"]),
    } for row in r["plan"]]


@router.get("/accuracy")
def accuracy() -> list:
    r = _ensure()
    return [{
        "sku": row["sku"], "name": row["sku"],
        "model_used": "gradient boosting" if row["model_used"] == "gbm" else "exponential smoothing",
        "mae": row.get("mae") or 0.0, "mape": row.get("mape") or 0.0,
        "abc_class": row["abc"],
    } for row in r["plan"]]


@router.get("/runs")
def runs() -> list:
    _ensure()
    return _RUNS[:20]


@router.get("/sku/{sku}")
def sku(sku: str, horizon: Optional[int] = None) -> dict:
    _ensure()
    try:
        d = sku_detail(_STATE["series"], _STATE["products"], sku, horizon=horizon)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown sku {sku!r}")

    fc = d["forecast"]
    g = _STATE["series"]
    hist = g[g["sku"] == sku].tail(120)
    history = [{"date": str(dd.date()), "units": int(u)}
               for dd, u in zip(hist["date"], hist["units"])]
    last = pd.to_datetime(hist["date"].max())
    h = len(fc["demand_path"])
    dates = [str((last + pd.Timedelta(days=i + 1)).date()) for i in range(h)]

    # sawtooth from policy
    pol, prod = d["policy"], d["product"]
    dmean = max(fc["demand_mean"], 1e-6)
    lead = float(prod["lead_time_days"])
    path, level, pending, pend_in = [], float(prod["on_hand"]), 0.0, -1
    for i in range(30):
        path.append({"date": str((last + pd.Timedelta(days=i + 1)).date()), "on_hand": round(level)})
        level -= dmean
        if pend_in == 0:
            level += pending; pending = 0; pend_in = -1
        elif pend_in > 0:
            pend_in -= 1
        if level <= pol["reorder_point"] and pending == 0 and pend_in < 0:
            pending = pol["eoq"] or pol["order_qty"]; pend_in = int(round(lead))

    row = next((p for p in _STATE["result"]["plan"] if p["sku"] == sku), {})
    status = _status(row) if row else "healthy"
    return {
        "sku": sku, "name": sku, "status": status, "supplier": str(prod.get("supplier", "")),
        "history": history,
        "forecast": {"dates": dates, "demand_path": fc["demand_path"],
                     "lower": fc["lower"], "upper": fc["upper"]},
        "inventory_path": path,
        "policy": {"reorder_point": pol["reorder_point"], "safety_stock": pol["safety_stock"],
                   "eoq": pol["eoq"], "order_qty": pol["order_qty"]},
        "risk": {"days_of_cover": d["risk"]["days_of_cover"],
                 "stockout_probability": d["risk"]["stockout_probability"]},
        "model_used": "gradient boosting" if fc["model_used"] == "gbm" else "exponential smoothing",
        "mae": row.get("mae") or 0.0, "mape": row.get("mape") or 0.0,
        "reason": ", ".join(row.get("reason_codes", [])) or "above reorder point",
        "on_plan": bool(row.get("order_qty", 0) > 0),
    }


@router.post("/plan/{sku}/approve")
def approve(sku: str) -> dict:
    _APPROVALS[sku] = "approved"
    return {"sku": sku, "approval": "approved"}


@router.post("/plan/{sku}/reject")
def reject(sku: str) -> dict:
    _APPROVALS[sku] = "rejected"
    return {"sku": sku, "approval": "rejected"}
