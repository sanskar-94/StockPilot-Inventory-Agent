"""The accuracy report — the ML showcase.

Walk-forward backtest per SKU comparing the exponential-smoothing baseline
against the gradient-boosting model, printing MAE and MAPE for each and which
one wins. Also writes a forecast plot for the highest-revenue SKU.

    python scripts/backtest.py
    python scripts/backtest.py --orders data/sample_orders.csv --horizon 14
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine import charts, config, data
from engine.forecast import backtest_sku, forecast_sku
from engine.segment import segment as segment_fn


def main() -> None:
    ap = argparse.ArgumentParser(description="StockPilot accuracy report")
    ap.add_argument("--orders", default=config.SAMPLE_ORDERS)
    ap.add_argument("--products", default=config.SAMPLE_PRODUCTS)
    ap.add_argument("--horizon", type=int, default=14)
    args = ap.parse_args()

    orders = data.load_orders(args.orders)
    products = data.load_products(args.products)
    series = data.build_demand_series(orders)

    print(f"Walk-forward backtest  (horizon = {args.horizon} days, "
          f"{config.BACKTEST_FOLDS} folds)\n")
    header = (f"{'SKU':<14}{'BASE MAE':>10}{'BASE MAPE':>11}"
              f"{'GBM MAE':>10}{'GBM MAPE':>11}{'WINNER':>9}{'LIFT':>8}")
    print(header)
    print("-" * len(header))

    base_maes, gbm_maes, winners = [], [], []
    for sku, g in series.groupby("sku"):
        bt = backtest_sku(g, args.horizon)
        base = bt.get("baseline") or {}
        gbm = bt.get("gbm") or {}
        bm = base.get("mae", float("nan"))
        gm = gbm.get("mae", float("nan"))
        winner = bt.get("winner", "baseline")
        lift = (bm - gm) / bm * 100 if (bm and not np.isnan(bm) and not np.isnan(gm)) else float("nan")
        print(f"{sku:<14}{bm:>10.2f}{base.get('mape', float('nan')):>11.2f}"
              f"{gm:>10.2f}{gbm.get('mape', float('nan')):>11.2f}{winner:>9}{lift:>7.1f}%")
        if not np.isnan(bm):
            base_maes.append(bm)
        if not np.isnan(gm):
            gbm_maes.append(gm)
        winners.append(winner)

    print("-" * len(header))
    print(f"Mean MAE   baseline {np.mean(base_maes):.2f}   gbm {np.mean(gbm_maes):.2f}")
    won = sum(1 for w in winners if w == "gbm")
    print(f"GBM wins on {won}/{len(winners)} SKUs; baseline keeps the rest "
          f"(quiet SKUs stay simple, on purpose).")

    # forecast plot for the top-revenue SKU
    seg = segment_fn(products, series).sort_values("revenue", ascending=False)
    top_sku = seg.iloc[0]["sku"]
    g = series[series["sku"] == top_sku]
    fc = forecast_sku(g, args.horizon)
    path = charts.forecast_chart(g, fc, out_dir=config.CHART_DIR)
    print(f"\nForecast plot for top SKU {top_sku}: {path}")


if __name__ == "__main__":
    main()
