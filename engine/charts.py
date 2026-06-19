"""The three charts.

  forecast_chart  per-SKU history + forecast with the shaded 90% band
  sawtooth_chart  inventory drawdown with a reorder landing before the buffer
  abc_chart       the ABC Pareto of revenue

These are what make a Slack post land. In the live system you render the current
ones from real data and attach the relevant image to the message.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def _ensure_dir(out_dir: str) -> str:
    out_dir = out_dir or config.CHART_DIR
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def forecast_chart(series_g: pd.DataFrame, fc: dict, out_dir: str = "",
                   history_days: int = 90) -> str:
    """History plus the forecast path and its shaded band."""
    out_dir = _ensure_dir(out_dir)
    sku = fc["sku"]
    g = series_g.sort_values("date").tail(history_days)
    hist_dates = pd.to_datetime(g["date"])
    last = hist_dates.max()
    horizon = fc["horizon"]
    future_dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hist_dates, g["units"], color="#3b4252", lw=1.3, label="actual demand")
    ax.plot(future_dates, fc["demand_path"], color="#bf616a", lw=2.0,
            marker="o", ms=3, label=f"forecast ({fc['model_used']})")
    ax.fill_between(future_dates, fc["lower"], fc["upper"], color="#bf616a",
                    alpha=0.18, label="90% band")
    ax.axvline(last, color="#888", ls="--", lw=0.8)
    ax.set_title(f"{sku} — demand forecast over the {horizon}-day lead time")
    ax.set_ylabel("units / day")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    path = os.path.join(out_dir, f"forecast_{sku}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def sawtooth_chart(plan_row: dict, fc: dict, out_dir: str = "") -> str:
    """The inventory sawtooth: stock drawing down, a reorder landing just before
    the safety buffer, then climbing back to the order-up-to level."""
    out_dir = _ensure_dir(out_dir)
    sku = plan_row["sku"]
    dmean = max(plan_row["demand_mean"], 1e-6)
    lead = plan_row["lead_time_days"]
    rop = plan_row["reorder_point"]
    ss = plan_row["safety_stock"]
    order_up_to = plan_row["order_up_to"]
    on_hand = plan_row["on_hand"]

    days = np.arange(0, 60)
    # deterministic sawtooth: deplete at mean demand, place an order when we
    # cross the reorder point, and it lands one lead time later.
    levels = []
    level = max(on_hand, order_up_to)
    pending = 0
    pending_in = -1
    for d in days:
        levels.append(level)
        level -= dmean
        if pending_in == 0:
            level += pending
            pending = 0
            pending_in = -1
        elif pending_in > 0:
            pending_in -= 1
        if level <= rop and pending == 0 and pending_in < 0:
            pending = order_up_to - level
            pending_in = int(round(lead))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(days, levels, color="#5e81ac", lw=1.8, label="on hand")
    ax.axhline(rop, color="#d08770", ls="--", lw=1.2, label="reorder point")
    ax.axhline(ss, color="#bf616a", ls=":", lw=1.2, label="safety stock")
    ax.axhline(order_up_to, color="#a3be8c", ls="--", lw=0.9, label="order-up-to")
    ax.set_title(f"{sku} — inventory sawtooth (reorder lands before the buffer)")
    ax.set_xlabel("days")
    ax.set_ylabel("units on hand")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    path = os.path.join(out_dir, f"sawtooth_{sku}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def abc_chart(segments: pd.DataFrame, out_dir: str = "") -> str:
    """ABC Pareto: revenue bars sorted high-to-low with the cumulative share."""
    out_dir = _ensure_dir(out_dir)
    df = segments.sort_values("revenue", ascending=False).reset_index(drop=True)
    colors = {"A": "#a3be8c", "B": "#ebcb8b", "C": "#bf616a"}
    bar_colors = [colors.get(c, "#888") for c in df["abc"]]

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar(df["sku"], df["revenue"], color=bar_colors)
    ax1.set_ylabel("trailing revenue")
    ax1.set_xticklabels(df["sku"], rotation=45, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(df["sku"], df["cum_share"] * 100, color="#3b4252", marker="o", ms=4)
    ax2.axhline(80, color="#888", ls="--", lw=0.8)
    ax2.set_ylabel("cumulative % of revenue")
    ax2.set_ylim(0, 105)
    ax1.set_title("ABC Pareto — where the revenue concentrates")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in ["A", "B", "C"]]
    ax1.legend(handles, ["A", "B", "C"], loc="center right", fontsize=8)
    path = os.path.join(out_dir, "abc_pareto.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def render_all(series: pd.DataFrame, result: dict, out_dir: str = "",
               top_n: int = 3) -> dict:
    """Render the ABC chart plus forecast+sawtooth for the top-N action SKUs.
    Returns {chart_name: path}."""
    out_dir = _ensure_dir(out_dir)
    plan = result["plan"]
    forecasts = result["forecasts"]
    charts = {"abc_pareto": abc_chart(result["segments"], out_dir)}

    action = plan[plan["order_qty"] > 0] if not plan.empty else plan
    top = action.head(top_n) if not action.empty else plan.head(top_n)
    for _, row in top.iterrows():
        sku = row["sku"]
        g = series[series["sku"] == sku]
        if sku in forecasts:
            charts[f"forecast_{sku}"] = forecast_chart(g, forecasts[sku], out_dir)
        charts[f"sawtooth_{sku}"] = sawtooth_chart(row.to_dict(), forecasts.get(sku, {}), out_dir)
    return charts
