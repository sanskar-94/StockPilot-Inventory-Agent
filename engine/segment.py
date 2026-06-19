"""Effort goes where the money is.

ABC ranks SKUs by revenue: a few products drive most of it. XYZ ranks them by
demand variability. The pair of letters sets the target service level and the
review cadence, so we hold availability where it pays and free up cash where
it does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def abc_classes(products: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    """Rank SKUs by trailing revenue, take the cumulative share.
    A = top 80% of revenue, B = next 15%, C = last 5%.
    Returns: sku, revenue, abc.
    """
    price = products.set_index("sku")["price"]
    units = demand.groupby("sku")["units"].sum()
    skus = sorted(set(units.index) | set(price.index))

    rev = pd.DataFrame({"sku": skus})
    rev["units"] = rev["sku"].map(units).fillna(0.0)
    rev["price"] = rev["sku"].map(price).fillna(0.0)
    rev["revenue"] = rev["units"] * rev["price"]
    rev = rev.sort_values("revenue", ascending=False).reset_index(drop=True)

    total = max(rev["revenue"].sum(), 1e-9)
    rev["cum_share"] = rev["revenue"].cumsum() / total

    def _cls(c):
        if c <= config.ABC_A_CUTOFF:
            return "A"
        if c <= config.ABC_B_CUTOFF:
            return "B"
        return "C"

    rev["abc"] = rev["cum_share"].apply(_cls)
    return rev[["sku", "revenue", "cum_share", "abc"]]


def xyz_classes(series: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of variation of daily demand per SKU.
    X = stable, Y = medium, Z = erratic.
    Returns: sku, cov, xyz.
    """
    rows = []
    for sku, g in series.groupby("sku"):
        u = g["units"].astype(float)
        mean = u.mean()
        cov = float(u.std() / mean) if mean > 0 else float("inf")
        if cov <= config.XYZ_X_CUTOFF:
            xyz = "X"
        elif cov <= config.XYZ_Y_CUTOFF:
            xyz = "Y"
        else:
            xyz = "Z"
        rows.append({"sku": sku, "cov": round(cov, 3) if np.isfinite(cov) else None, "xyz": xyz})
    # always return the typed columns so an empty series still merges cleanly
    return pd.DataFrame(rows, columns=["sku", "cov", "xyz"])


def segment(products: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Combine ABC and XYZ and attach the service level, Z-score, and review
    period each SKU should be managed at.
    Returns: sku, revenue, abc, cov, xyz, segment, service_level, z, review_period_days.
    """
    abc = abc_classes(products, series)
    xyz = xyz_classes(series)
    # coerce the join key so an all-empty input (empty float64 vs object) merges
    abc["sku"] = abc["sku"].astype(str)
    xyz["sku"] = xyz["sku"].astype(str)
    out = abc.merge(xyz, on="sku", how="outer")
    out["abc"] = out["abc"].fillna("C")
    out["xyz"] = out["xyz"].fillna("Z")
    out["segment"] = out["abc"] + out["xyz"]
    out["service_level"] = out["abc"].map(config.SERVICE_LEVEL)
    out["z"] = out["abc"].map(config.Z_SCORE)
    out["review_period_days"] = out["abc"].map(config.REVIEW_PERIOD_DAYS)
    return out
