"""Load orders and products, and build the continuous daily demand series.

The one thing people get wrong here is dropping the zero-sales days. If a
product sold nothing on Tuesday, the series still needs a Tuesday row with
zero units, or the model reads a false steady trend instead of a real gap.
"""
from __future__ import annotations

import io
from typing import Union

import pandas as pd

from . import config

Source = Union[str, pd.DataFrame, list, dict, bytes]

ORDER_COLUMNS = ["date", "sku", "units", "price", "on_promo"]
PRODUCT_COLUMNS = [
    "sku", "cost", "price", "lead_time_days", "lead_time_std",
    "supplier", "on_hand", "on_order", "moq", "order_cost", "holding_rate",
]


def _to_frame(source: Source) -> pd.DataFrame:
    """Accept a CSV path, a raw CSV string/bytes, a DataFrame, or records.

    This is what lets the same loader serve both demo mode (a CSV on disk)
    and connected mode (a list of order dicts posted from n8n/Shopify).
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (list, dict)):
        return pd.DataFrame(source)
    if isinstance(source, bytes):
        return pd.read_csv(io.BytesIO(source))
    if isinstance(source, str):
        looks_like_csv = "\n" in source or "," in source.splitlines()[0] if source else False
        if looks_like_csv and not source.strip().lower().endswith(".csv"):
            return pd.read_csv(io.StringIO(source))
        return pd.read_csv(source)
    raise TypeError(f"Unsupported order source: {type(source)!r}")


def load_orders(source: Source) -> pd.DataFrame:
    """Read order lines from a CSV export or a Shopify pull.

    Returns columns: date (datetime), sku, units, price, on_promo.
    Tolerates missing price/on_promo columns by defaulting them.
    """
    df = _to_frame(source)
    df.columns = [c.strip().lower() for c in df.columns]

    # A fresh store sends products but no orders — return an empty typed frame
    # rather than erroring, so the pipeline yields an empty plan, not a 400.
    if df.empty:
        return pd.DataFrame(columns=ORDER_COLUMNS)

    if "sku" not in df.columns:
        raise ValueError("orders need a 'sku' column")
    if "date" not in df.columns:
        raise ValueError("orders need a 'date' column")
    if "units" not in df.columns:
        # Shopify line items sometimes call it 'quantity'
        if "quantity" in df.columns:
            df = df.rename(columns={"quantity": "units"})
        else:
            raise ValueError("orders need a 'units' (or 'quantity') column")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["sku"] = df["sku"].astype(str)
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0).clip(lower=0)
    df["price"] = pd.to_numeric(df.get("price", 0.0), errors="coerce").fillna(0.0)
    if "on_promo" not in df.columns:
        df["on_promo"] = 0
    df["on_promo"] = pd.to_numeric(df["on_promo"], errors="coerce").fillna(0).astype(int)
    return df[ORDER_COLUMNS]


def build_demand_series(orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order lines into one row per sku per day.

    Fills the missing days with zero sales so the series is continuous and
    the forecast can see real gaps in demand. Carries a per-day on_promo flag
    (1 if any line that day was on promo) and a per-day mean price.
    Returns columns: date, sku, units, on_promo, price.
    """
    if orders.empty:
        return pd.DataFrame(columns=["date", "sku", "units", "on_promo", "price"])

    daily = (
        orders.groupby(["sku", "date"])
        .agg(units=("units", "sum"),
             on_promo=("on_promo", "max"),
             price=("price", "mean"))
        .reset_index()
    )

    full = []
    for sku, g in daily.groupby("sku"):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(idx)
        g["sku"] = sku
        g["units"] = g["units"].fillna(0.0)
        g["on_promo"] = g["on_promo"].fillna(0).astype(int)
        g["price"] = g["price"].ffill().bfill()
        g.index.name = "date"
        full.append(g.reset_index())

    series = pd.concat(full, ignore_index=True)
    return series[["date", "sku", "units", "on_promo", "price"]].sort_values(
        ["sku", "date"]).reset_index(drop=True)


def load_products(source: Source) -> pd.DataFrame:
    """Read the product master.

    Returns: sku, cost, price, lead_time_days, lead_time_std, supplier,
    on_hand, on_order, moq, order_cost, holding_rate.
    Missing optional columns fall back to the defaults in config.
    """
    df = _to_frame(source)
    df.columns = [c.strip().lower() for c in df.columns]
    if df.empty:
        return pd.DataFrame(columns=PRODUCT_COLUMNS)
    if "sku" not in df.columns:
        raise ValueError("products need a 'sku' column")
    df["sku"] = df["sku"].astype(str)

    defaults = {
        "cost": 0.0,
        "price": 0.0,
        "lead_time_days": config.DEFAULT_LEAD_TIME_DAYS,
        "lead_time_std": config.DEFAULT_LEAD_TIME_STD,
        "supplier": "default",
        "on_hand": 0,
        "on_order": 0,
        "moq": config.DEFAULT_MOQ,
        "order_cost": config.DEFAULT_ORDER_COST,
        "holding_rate": config.DEFAULT_HOLDING_RATE,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        if col != "supplier":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    df["supplier"] = df["supplier"].fillna("default").astype(str)
    return df[PRODUCT_COLUMNS]
