"""Generate realistic sample data so the demo has something honest to chew on.

Produces:
  data/sample_orders.csv     date, sku, units, price, on_promo   (order lines)
  data/sample_products.csv   sku, cost, price, lead_time_days, ...

The demand has weekly rhythm, trends, promo lifts, and different volume and
variability per SKU on purpose, so ABC/XYZ segmentation and the forecast each
have something real to find. Deterministic via a fixed seed.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

SEED = 42
DAYS = 420                      # ~14 months of daily history
END = pd.Timestamp("2026-06-15")
START = END - pd.Timedelta(days=DAYS - 1)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

# sku, base/day, weekend_mult, trend/yr, promo_freq, promo_lift, noise_cov, price, cost, supplier, lead, lead_std, moq
SKUS = [
    ("WIDGET-RED",   22, 1.15,  0.10, 0.04, 1.6, 0.18, 19.99,  7.50, "Acme Supply",    7, 1.5,  50),
    ("WIDGET-BLUE",  18, 1.15,  0.05, 0.04, 1.6, 0.20, 19.99,  7.50, "Acme Supply",    7, 1.5,  50),
    ("GADGET-PRO",   14, 1.35,  0.25, 0.05, 2.0, 0.30, 49.99, 22.00, "Globex Trading",10, 3.0,  25),
    ("GADGET-MINI",   9, 1.25,  0.00, 0.07, 2.4, 0.35, 24.99, 11.00, "Globex Trading",10, 3.0,  25),
    ("CABLE-USBC",   40, 1.05,  0.15, 0.03, 1.4, 0.15,  9.99,  2.20, "Acme Supply",    5, 1.0, 100),
    ("CASE-CLEAR",    7, 1.20,  0.00, 0.06, 1.8, 0.55, 14.99,  4.00, "Initech Parts", 14, 4.0,  50),
    ("CHARGER-65W",  11, 1.30,  0.20, 0.06, 2.2, 0.32, 39.99, 16.50, "Globex Trading",12, 3.5,  25),
    ("STICKER-PACK",  4, 1.10, -0.10, 0.05, 1.5, 0.70,  4.99,  0.80, "Initech Parts",  6, 2.0, 200),
    ("STAND-ALU",     3, 1.15,  0.05, 0.04, 1.7, 0.80, 29.99, 12.00, "Initech Parts", 21, 6.0,  10),
    ("SCREEN-GUARD",  8, 1.10, -0.25, 0.05, 1.6, 0.40, 12.99,  3.20, "Acme Supply",    6, 2.0, 100),
]


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, END, freq="D")
    n = len(dates)
    dow = dates.dayofweek.values            # 0=Mon ... 6=Sun
    dom = dates.day.values                   # day of month, for the payday bump
    t = np.arange(n) / 365.0                 # years elapsed, for trend
    # payday spend spike around the 1st and 15th — a monthly pattern the weekly
    # baseline can't capture but the tree model can learn from day_of_month.
    payday = np.where(np.isin(dom, [1, 2, 15, 16]), 1.35, 1.0)

    order_rows = []
    product_rows = []

    for (sku, base, wke, trend, pfreq, plift, cov,
         price, cost, supplier, lead, lead_std, moq) in SKUS:
        # weekly shape: weekend lift, slight mid-week dip
        weekly = np.where(dow >= 5, wke, 1.0)
        weekly = weekly * (1.0 + 0.05 * np.sin(2 * np.pi * dow / 7))
        # gentle yearly seasonality + linear-ish trend
        seasonal = 1.0 + 0.12 * np.sin(2 * np.pi * (np.arange(n) / 365.0) + 1.0)
        trend_mult = 1.0 + trend * t
        # promotions: random days get a demand lift, flagged in the data
        on_promo = (rng.random(n) < pfreq).astype(int)
        promo_mult = np.where(on_promo == 1, plift, 1.0)

        mean = base * weekly * seasonal * trend_mult * promo_mult * payday
        mean = np.clip(mean, 0.05, None)
        # negative-binomial-ish noise via gamma-Poisson for realistic dispersion
        shape = max(1.0 / (cov ** 2), 0.4)
        gamma = rng.gamma(shape, mean / shape)
        units = rng.poisson(gamma)

        for d, u, pr in zip(dates, units, on_promo):
            if u > 0:                        # zero-days are reconstructed downstream
                order_rows.append((d.date().isoformat(), sku, int(u), price, int(pr)))

        # set opening on_hand near a couple weeks of cover so the demo has
        # both "order now" and "fine for now" SKUs
        recent_mean = float(np.mean(mean[-28:]))
        on_hand = int(round(recent_mean * rng.uniform(3, 16)))
        on_order = int(round(recent_mean * rng.choice([0, 0, 0, 5])))
        product_rows.append((
            sku, round(cost, 2), round(price, 2), lead, lead_std,
            supplier, on_hand, on_order, moq, 50.0, 0.25,
        ))

    orders = pd.DataFrame(order_rows, columns=["date", "sku", "units", "price", "on_promo"])
    orders = orders.sort_values(["date", "sku"]).reset_index(drop=True)
    products = pd.DataFrame(product_rows, columns=[
        "sku", "cost", "price", "lead_time_days", "lead_time_std",
        "supplier", "on_hand", "on_order", "moq", "order_cost", "holding_rate",
    ])

    os.makedirs(DATA_DIR, exist_ok=True)
    orders.to_csv(os.path.join(DATA_DIR, "sample_orders.csv"), index=False)
    products.to_csv(os.path.join(DATA_DIR, "sample_products.csv"), index=False)
    print(f"Wrote {len(orders):,} order lines across {orders.sku.nunique()} SKUs "
          f"({orders.date.min()} .. {orders.date.max()})")
    print(f"Wrote {len(products)} products")


if __name__ == "__main__":
    main()
