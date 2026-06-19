"""One command CSV run: forecast, plan, draft the PO, write the charts.

    python scripts/run_demo.py
    python scripts/run_demo.py --orders data/sample_orders.csv \
                               --products data/sample_products.csv --llm

No Shopify, no Slack, no account needed. Add --llm to route the summary through
NIM (needs NIM_API_KEY in .env); without it the summary is built locally.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from engine import config
from engine.pipeline import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="StockPilot demo run")
    ap.add_argument("--orders", default=config.SAMPLE_ORDERS)
    ap.add_argument("--products", default=config.SAMPLE_PRODUCTS)
    ap.add_argument("--llm", action="store_true", help="route summary through NIM")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    print("StockPilot — demo run")
    print(f"  orders   : {args.orders}")
    print(f"  products : {args.products}")
    print("  forecasting, segmenting, sizing orders, scoring risk ...\n")

    r = run_pipeline(args.orders, args.products, use_llm=args.llm,
                     render_charts=not args.no_charts)

    span = r["generated_for"]
    print(f"Data: {span['skus']} SKUs, {span['from']} -> {span['to']}")
    print("=" * 92)
    hdr = f"{'SKU':<14}{'SEG':<5}{'ON HAND':>8}{'ORDER':>7}{'COVER':>7}{'P(out)':>8}{'MODEL':>16}  REASONS"
    print(hdr)
    print("-" * 92)
    for row in r["plan"]:
        print(f"{row['sku']:<14}{row['segment']:<5}{row['on_hand']:>8}{row['order_qty']:>7}"
              f"{row['days_of_cover']:>7.1f}{row['stockout_probability']:>8.2f}"
              f"{row['model_used']:>16}  {', '.join(row['reason_codes'])}")
    print("=" * 92)

    print(f"\nReorder now: {r['reorder_count']} SKU(s)   "
          f"Estimated spend: ${r['total_order_value']:,.0f}")
    print(f"\nSummary ({r['reasoning']['_source']}):\n  {r['reasoning']['summary']}")

    print("\nDraft purchase orders:")
    for po in r["reasoning"]["purchase_orders"]:
        print(f"  {po['supplier']}: {po['total_units']} units, ${po['total_value']:,.0f}")
        for line in po["lines"]:
            print(f"      {line['sku']:<14} x {line['order_qty']}")

    if r["charts"]:
        print("\nCharts written:")
        for name, path in r["charts"].items():
            print(f"  {name:<22} {path}")


if __name__ == "__main__":
    main()
