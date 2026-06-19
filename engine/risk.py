"""Early warning, before the plan even runs.

Days of cover is the fast read: if it is below the lead time, that product
runs out before any reorder can land. Stockout probability turns that into a
number you can rank on. The anomaly flag is the human safety valve, so a sudden
spike gets a set of eyes instead of a silent reorder.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def days_of_cover(on_hand: float, demand_mean: float) -> float:
    return float(on_hand / max(demand_mean, 1e-9))


def stockout_probability(on_hand: float, demand_mean: float,
                         demand_std: float, lead_time: float) -> float:
    """Probability that demand over the lead time exceeds stock on hand."""
    mu = demand_mean * lead_time
    sigma = np.sqrt(lead_time) * demand_std
    if sigma <= 0:
        return 0.0 if on_hand >= mu else 1.0
    return float(1 - norm.cdf(on_hand, loc=mu, scale=sigma))


def anomaly_flag(recent_actual, recent_forecast, resid_std: float, k: float = 3.0) -> bool:
    """Flag a demand spike or drop when recent residuals run past k sigma, so a
    person looks before the model bakes it in."""
    recent_actual = np.asarray(recent_actual, dtype=float)
    recent_forecast = np.asarray(recent_forecast, dtype=float)
    if recent_actual.size == 0:
        return False
    z = (recent_actual - recent_forecast) / max(resid_std, 1e-9)
    return bool(np.abs(z).max() > k)


def risk_for_sku(product: dict, demand_mean: float, demand_std: float,
                 lead_time: float, recent_actual=None, recent_forecast=None,
                 resid_std: float = 1.0) -> dict:
    """Bundle the three risk reads for one SKU."""
    on_hand = float(product["on_hand"])
    doc = days_of_cover(on_hand, demand_mean)
    sp = stockout_probability(on_hand, demand_mean, demand_std, lead_time)
    flag = False
    if recent_actual is not None and recent_forecast is not None:
        flag = anomaly_flag(recent_actual, recent_forecast, resid_std)
    return {
        "days_of_cover": round(doc, 1),
        "stockout_probability": round(sp, 4),
        "anomaly": flag,
        "below_lead_time": bool(doc < lead_time),
    }
