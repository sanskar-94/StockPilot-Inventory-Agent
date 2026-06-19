"""The reasoning layer.

One rule sits above everything here: the model never invents a number. The
quantities, reorder points, and risks all come from Python. NIM only ranks the
orders, groups them by supplier into a clean purchase order, and writes the
plain-language summary.

After the model replies we check that every quantity it returned equals the
quantity we sent. If anything drifted, we repair it from the source numbers.
And if no LLM is configured at all (pure demo mode), we build the same shape
deterministically so the pipeline still produces a full, trustworthy plan.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from llm import nim_client

SYSTEM = (
    "You are an inventory operations assistant. You are given a JSON list of "
    "reorder recommendations whose numbers have already been computed. You MUST "
    "NOT change any number. Your only job is to (1) rank the orders by urgency, "
    "(2) group them by supplier into clean purchase orders, and (3) write a short "
    "plain-language summary for a store owner. Reply with ONLY a JSON object with "
    "keys: ranked_orders (list of {sku, order_qty, why}), purchase_orders "
    "(list of {supplier, lines:[{sku, order_qty}], total_units, total_value}), "
    "and summary (string). Every order_qty must exactly match the input."
)


def build_messages(plan_records: list) -> list:
    """Pass the computed plan as JSON. Ask only for ordering, grouping, and prose."""
    user = (
        "Here is today's computed reorder plan as JSON. Rank, group by supplier, "
        "and summarize. Do not alter any order_qty.\n\n"
        + json.dumps(plan_records, indent=2, default=str)
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_json(raw: str) -> dict:
    """Pull the first JSON object out of the model's reply, tolerating code
    fences and stray prose around it."""
    if not raw:
        raise ValueError("empty model reply")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = raw.find("{")
        end = raw.rfind("}")
        candidate = raw[start:end + 1] if start != -1 and end != -1 else raw
    return json.loads(candidate)


def validate_quantities_unchanged(sent: list, result: dict) -> dict:
    """Repair any quantity the model changed back to the source-of-truth value.
    Returns the (repaired) result and never lets a model-altered number through.
    """
    truth = {r["sku"]: int(r["order_qty"]) for r in sent}

    repaired = 0
    for order in result.get("ranked_orders", []):
        sku = order.get("sku")
        if sku in truth and int(order.get("order_qty", -1)) != truth[sku]:
            order["order_qty"] = truth[sku]
            repaired += 1

    for po in result.get("purchase_orders", []):
        for line in po.get("lines", []):
            sku = line.get("sku")
            if sku in truth and int(line.get("order_qty", -1)) != truth[sku]:
                line["order_qty"] = truth[sku]
                repaired += 1

    result["_quantities_repaired"] = repaired
    return result


def deterministic_reason(plan_records: list) -> dict:
    """Local fallback: rank, group, and summarize with no LLM. Same output shape
    as the model path, so demo mode needs no account or API key."""
    ranked = sorted(plan_records,
                    key=lambda r: (r.get("stockout_probability", 0),
                                   r.get("order_value", 0)), reverse=True)
    ranked_orders = [{
        "sku": r["sku"], "order_qty": int(r["order_qty"]),
        "why": ", ".join(r.get("reason_codes", [])) or "scheduled top-up",
    } for r in ranked]

    by_supplier: dict = {}
    for r in plan_records:
        by_supplier.setdefault(r.get("supplier", "default"), []).append(r)
    purchase_orders = []
    for supplier, rows in by_supplier.items():
        purchase_orders.append({
            "supplier": supplier,
            "lines": [{"sku": r["sku"], "order_qty": int(r["order_qty"])} for r in rows],
            "total_units": int(sum(r["order_qty"] for r in rows)),
            "total_value": round(sum(r.get("order_value", 0) for r in rows), 2),
        })

    n = len(plan_records)
    total_value = round(sum(r.get("order_value", 0) for r in plan_records), 2)
    urgent = [r["sku"] for r in plan_records if r.get("stockout_probability", 0) >= 0.2]
    summary = (
        f"{n} SKU(s) need reordering this run for an estimated ${total_value:,.0f} "
        f"across {len(purchase_orders)} supplier(s)."
    )
    if urgent:
        summary += f" Highest urgency: {', '.join(urgent[:5])} (elevated stockout risk)."
    else:
        summary += " No SKU is at immediate stockout risk; these are scheduled top-ups."

    return {"ranked_orders": ranked_orders, "purchase_orders": purchase_orders,
            "summary": summary, "_source": "deterministic"}


def reason_over_plan(plan_records: list, use_llm: bool = True) -> dict:
    """Rank, group, and summarize the plan. Tries NIM (with fallback provider),
    validates the numbers, and degrades gracefully to the local builder so the
    pipeline always returns a complete result."""
    if not plan_records:
        return {"ranked_orders": [], "purchase_orders": [], "summary":
                "Nothing to order: every SKU is above its reorder point.",
                "_source": "empty"}

    if not use_llm:
        return deterministic_reason(plan_records)

    try:
        raw = nim_client.call_nim(build_messages(plan_records))
        result = parse_json(raw)
        result = validate_quantities_unchanged(plan_records, result)
        result.setdefault("_source", "nim")
        # make sure nothing was dropped
        if len(result.get("ranked_orders", [])) != len(plan_records):
            det = deterministic_reason(plan_records)
            result["ranked_orders"] = det["ranked_orders"]
            result["purchase_orders"] = det["purchase_orders"]
        return result
    except Exception as e:
        result = deterministic_reason(plan_records)
        result["_source"] = f"deterministic_fallback ({type(e).__name__})"
        return result
