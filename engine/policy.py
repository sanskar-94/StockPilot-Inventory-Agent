"""The three formulas that decide when and how much.

Reorder point = average daily demand x lead time + safety stock.
Safety stock  = service-level Z x sqrt(lead time x demand variance
                + demand^2 x lead-time variance), which absorbs both a bad
                demand week and a late supplier.
EOQ           = sqrt(2 x annual demand x order cost / holding cost per unit).
Order qty fires only when inventory position has fallen to the reorder point,
and tops up to a clean order-up-to level rather than guessing a round number.
"""
from __future__ import annotations

import numpy as np


def safety_stock(z: float, demand_mean: float, demand_std: float,
                 lead_time: float, lead_time_std: float) -> float:
    """Buffer sized from the service level, demand variability, and lead-time
    variability."""
    variance = lead_time * demand_std ** 2 + demand_mean ** 2 * lead_time_std ** 2
    return float(z * np.sqrt(max(variance, 0.0)))


def reorder_point(demand_mean: float, lead_time: float, ss: float) -> float:
    """The stock level that triggers a new order, set so the order arrives
    before the buffer is touched."""
    return float(demand_mean * lead_time + ss)


def eoq(annual_demand: float, order_cost: float, holding_cost_per_unit: float) -> float:
    """The order size that balances the cost of ordering against the cost of
    holding."""
    if holding_cost_per_unit <= 0 or annual_demand <= 0:
        return 0.0
    return float(np.sqrt(2 * annual_demand * order_cost / holding_cost_per_unit))


def order_quantity(on_hand: float, on_order: float, rop: float,
                   eoq_value: float, moq: float) -> int:
    """Order nothing while inventory position is above the reorder point;
    otherwise top up to (reorder point + EOQ), respecting the MOQ."""
    inventory_position = on_hand + on_order
    if inventory_position > rop:
        return 0
    order_up_to = rop + eoq_value
    return int(max(moq, round(order_up_to - inventory_position)))


def policy_for_sku(demand_mean: float, demand_std: float, product: dict,
                   z: float) -> dict:
    """Run all three formulas for one SKU and return the levels plus the
    recommended order quantity. `product` is one row of the product master."""
    lead = float(product["lead_time_days"])
    lead_std = float(product["lead_time_std"])
    ss = safety_stock(z, demand_mean, demand_std, lead, lead_std)
    rop = reorder_point(demand_mean, lead, ss)

    annual_demand = demand_mean * 365.0
    holding_cost_per_unit = float(product["cost"]) * float(product["holding_rate"])
    eoq_value = eoq(annual_demand, float(product["order_cost"]), holding_cost_per_unit)

    qty = order_quantity(float(product["on_hand"]), float(product["on_order"]),
                         rop, eoq_value, float(product["moq"]))

    return {
        "safety_stock": round(ss, 1),
        "reorder_point": round(rop, 1),
        "eoq": round(eoq_value, 1),
        "order_qty": qty,
        "inventory_position": float(product["on_hand"]) + float(product["on_order"]),
        "order_up_to": round(rop + eoq_value, 1),
    }
