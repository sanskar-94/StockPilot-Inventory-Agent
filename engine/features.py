"""Build the model matrix from a demand series.

These are the features that let a tree model see weekly rhythm, a promotion
bump, and a recent shift — things a flat average never sees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 28]

FEATURE_COLUMNS = (
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
    + ["dow", "weekofyear", "month", "day_of_month", "is_payday",
       "is_weekend", "on_promo", "days_since_stockout"]
)


def _days_since_stockout(units: pd.Series) -> np.ndarray:
    """Count days since the last zero-sales (proxy for stockout) day."""
    out = np.empty(len(units), dtype=float)
    counter = 0.0
    for i, u in enumerate(units.values):
        out[i] = counter
        counter = 0.0 if u <= 0 else counter + 1.0
    return out


def make_features_for_sku(g: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU feature frame, aligned to the demand series rows.

    Lags and rollings are shifted by one day so a row only ever sees the
    past — no leakage of today's demand into today's features.
    """
    g = g.sort_values("date").reset_index(drop=True).copy()
    u = g["units"].astype(float)

    for l in LAGS:
        g[f"lag_{l}"] = u.shift(l)
    for w in ROLL_WINDOWS:
        past = u.shift(1)
        g[f"roll_mean_{w}"] = past.rolling(w, min_periods=1).mean()
        g[f"roll_std_{w}"] = past.rolling(w, min_periods=1).std()

    dt = pd.to_datetime(g["date"])
    g["dow"] = dt.dt.dayofweek
    g["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    g["month"] = dt.dt.month
    g["day_of_month"] = dt.dt.day
    # payday window: monthly spend spikes around the 1st and 15th — a real
    # retail pattern the weekly-seasonal baseline cannot see, but a tree can.
    g["is_payday"] = dt.dt.day.isin([1, 2, 15, 16]).astype(int)
    g["is_weekend"] = (g["dow"] >= 5).astype(int)
    if "on_promo" not in g.columns:
        g["on_promo"] = 0
    g["days_since_stockout"] = _days_since_stockout(g["units"])

    g[FEATURE_COLUMNS] = g[FEATURE_COLUMNS].fillna(0.0)
    return g


def make_features(series: pd.DataFrame) -> pd.DataFrame:
    """Build features for every SKU in the series.

    Returns the series frame with FEATURE_COLUMNS added, aligned row-for-row.
    """
    parts = [make_features_for_sku(g) for _, g in series.groupby("sku")]
    return pd.concat(parts, ignore_index=True)
