# StockPilot architecture and build guide

This is the how. The brief covered what StockPilot does and how it sells. This document is the build: the architecture, every component, the math the engine runs, the workflow that ties it together, and a three day plan to ship it. It is written to live in the repo as `ARCHITECTURE.md`.

## How it runs

One engine, two modes.

1. **Demo mode (CSV).** You point the engine at a CSV of order history. It forecasts demand, computes the reorder plan, drafts the purchase order, and writes charts to disk. No Shopify, no Slack, no account needed. This is what runs in the public repo so anyone can try it in one command.
2. **Connected mode (live store).** n8n runs on a schedule, pulls the store's data from Shopify, sends it to the engine, posts the plan to Slack for approval, and on a yes writes back to Shopify and sends the order. Same engine, real data, a person in the loop.

The split matters. The brain is plain Python that runs anywhere. The plumbing (schedule, store, chat) is the only part that needs accounts, and it stays thin.

## Architecture at a glance

```
   n8n schedule fires
          │
          ▼
   pull from Shopify  (API 1)        products, live stock, order history
          │
          ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Engine   (local Python, no cost)                         │
   │                                                           │
   │     data  ▶  features  ▶  forecast  ▶  segment            │
   │                                            │              │
   │                                            ▼              │
   │                              policy  ▶  risk  ▶  plan      │
   │                                            │              │
   │                                            ▼              │
   │                                reason  ◀──▶  NVIDIA NIM    │
   │                                              (API 2)      │
   └─────────────────────────────────────────────────────────┘
          │
          ▼   plan, purchase order, weekly summary, chart
   post to Slack  (API 3)            approve or reject buttons
          │
          ▼   click returns to n8n through a webhook
   on approve:  write order back to Shopify, adjust expected stock, log
```

## The three external APIs

Restating the budget rule and naming exactly where each one is used.

1. **Shopify Admin API.** Reads products, variants, live inventory levels, and order history. The order history is what becomes the demand series. Writes back the approved order as a draft order or a metafield and adjusts expected stock. Used in the n8n pull step and the write back step.
2. **NVIDIA NIM.** The reasoning and writing layer. It takes the finished numbers and returns a ranked plan, a grouped purchase order, and a short summary. Called once per run from inside the engine. Wrapped with a fallback provider so a rate limit does not stop the run.
3. **Slack.** The approval gate. The plan arrives as a Block Kit message with approve and reject buttons, and the click comes back to n8n through a webhook. Used in the Slack post step and the webhook step.

Everything else runs as local Python at no cost: forecasting, the inventory math, segmentation, anomaly scoring, the charts, the API service, and the MCP server.

## Repo structure

```
stockpilot/
  engine/
    data.py          load orders and products, build the daily demand series
    features.py      lag, calendar, rolling, promo features
    forecast.py      baseline plus gradient boosting, backtest, prediction band
    segment.py       ABC by revenue, XYZ by demand variability
    policy.py        reorder point, safety stock, order quantity
    risk.py          days of cover, stockout probability, anomaly flag
    plan.py          assemble the reorder plan from everything above
    reason.py        build the prompt, call NIM, parse and validate
    charts.py        forecast, sawtooth, and ABC charts
    config.py        service levels, costs, lead times, thresholds
  llm/
    nim_client.py    NIM wrapper with retry and a fallback provider
  api/
    main.py          FastAPI: run the pipeline, fetch the plan, drill into a SKU
  mcp/
    server.py        MCP tools so the engine is queryable from Claude
  data/
    sample_orders.csv
    sample_products.csv
  n8n/
    stockpilot_workflow.json
  scripts/
    run_demo.py      one command CSV run, prints the plan, writes charts
    backtest.py      the accuracy report, the showcase artifact
  README.md
  ARCHITECTURE.md
  requirements.txt
  .env.example
```

Names use underscores rather than hyphens, on purpose, so the whole tree stays clean.

## The engine

This is where the depth is, and the part worth showing on a profile. Each module is small and does one job, so the pipeline reads top to bottom. Skeletons below show the shape and the real math, not the full bodies. The full bodies are the build itself.

### data.py

```python
import pandas as pd

def load_orders(source):
    """Read order lines from a CSV export or a Shopify pull.
    Returns columns: date, sku, units, price, on_promo."""
    ...

def build_demand_series(orders):
    """Aggregate order lines into one row per sku per day.
    Fill missing days with zero sales so the series is continuous,
    which the forecast needs to see real gaps in demand."""
    ...

def load_products(source):
    """Returns: sku, cost, price, lead_time_days, lead_time_std,
    supplier, on_hand, on_order, moq, order_cost, holding_rate."""
    ...
```

The one thing people get wrong here is leaving out the zero days. If a product sold nothing on Tuesday, the series needs a Tuesday with zero, or the model reads a false steady trend.

### features.py

```python
def make_features(series):
    """Per sku, build the model matrix:
    lags at 1, 7, 14, 28 days
    rolling mean and rolling std over 7 and 28 days
    day of week, week of year, month
    on_promo flag
    days since the last stockout
    Returns a frame aligned to the demand series."""
    ...
```

These are the features that let a tree model see weekly rhythm, a promotion bump, and a recent shift, which a flat average never sees.

### forecast.py

Two models, one honest test.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def backtest_sku(series, features, horizon):
    """Walk forward validation with an expanding window.
    Train the baseline (exponential smoothing) and the gradient
    boosting model, score both on held out windows, return
    MAE and MAPE for each. This is what proves the forecast."""
    ...

def forecast_sku(series, features, horizon):
    """Over the lead time horizon, return:
    demand_path   predicted units per day
    demand_mean   average daily demand, for the policy layer
    demand_std    demand variability, for safety stock
    lower, upper  the 90 percent band
    model_used    whichever model won the backtest for this sku."""
    ...
```

The baseline is exponential smoothing, which captures level, trend, and a weekly season and sets a floor. The main model is a gradient boosting regressor on the feature matrix, which picks up the promotions and nonlinear patterns the baseline misses. You pick the winner per SKU from the backtest, so a quiet product keeps the simple model and a busy one gets the stronger one.

The 90 percent band comes from two quantile models, one at the fifth percentile and one at the ninety fifth:

```python
low  = HistGradientBoostingRegressor(loss="quantile", quantile=0.05)
high = HistGradientBoostingRegressor(loss="quantile", quantile=0.95)
```

That band is not decoration. It feeds the safety stock and it is the shaded area on the chart that makes the post land.

### segment.py

Effort goes where the money is.

```python
def abc_classes(products, demand):
    """Rank skus by trailing revenue, take the cumulative share.
    A = top 80 percent of revenue, B = next 15, C = last 5."""
    ...

def xyz_classes(series):
    """Coefficient of variation of daily demand per sku.
    X = stable, Y = medium, Z = erratic demand."""
    ...

SERVICE_LEVEL = {"A": 0.98, "B": 0.95, "C": 0.90}
Z_SCORE       = {"A": 2.05, "B": 1.65, "C": 1.28}
```

The pair of letters sets the target service level and the review cadence. The few products that drive most of the revenue get a high service level and a tighter buffer. The long tail gets a leaner one. That is how you hold availability where it pays and free up cash where it does not.

### policy.py

The three formulas that decide when and how much.

```python
import numpy as np

def safety_stock(z, demand_mean, demand_std, lead_time, lead_time_std):
    """Buffer sized from the service level, demand variability,
    and lead time variability."""
    return z * np.sqrt(lead_time * demand_std**2
                       + demand_mean**2 * lead_time_std**2)

def reorder_point(demand_mean, lead_time, ss):
    """The stock level that triggers a new order, set so the order
    arrives before the buffer is touched."""
    return demand_mean * lead_time + ss

def eoq(annual_demand, order_cost, holding_cost_per_unit):
    """The order size that balances the cost of ordering against
    the cost of holding."""
    return np.sqrt(2 * annual_demand * order_cost / holding_cost_per_unit)

def order_quantity(on_hand, on_order, rop, eoq_value, moq):
    inventory_position = on_hand + on_order
    if inventory_position > rop:
        return 0
    order_up_to = rop + eoq_value
    return max(moq, round(order_up_to - inventory_position))
```

In words. Reorder point is average daily demand times lead time plus safety stock. Safety stock is the service level Z times the square root of (lead time times demand variance plus demand squared times lead time variance), which absorbs both a bad demand week and a late supplier. Economic order quantity is the square root of two times annual demand times order cost divided by holding cost per unit. The order quantity only fires when the position has fallen to the reorder point, and it tops up to a clean level rather than guessing a round number.

### risk.py

Early warning, before the plan even runs.

```python
import numpy as np
from scipy.stats import norm

def days_of_cover(on_hand, demand_mean):
    return on_hand / max(demand_mean, 1e-9)

def stockout_probability(on_hand, demand_mean, demand_std, lead_time):
    """Probability that demand over the lead time exceeds stock on hand."""
    mu = demand_mean * lead_time
    sigma = np.sqrt(lead_time) * demand_std
    return 1 - norm.cdf(on_hand, loc=mu, scale=sigma)

def anomaly_flag(recent_actual, recent_forecast, resid_std, k=3):
    """Flag a demand spike or drop when recent residuals run past k sigma,
    so a person looks before the model bakes it in."""
    z = (recent_actual - recent_forecast) / max(resid_std, 1e-9)
    return bool(np.abs(z).max() > k)
```

Days of cover is the fast read: if it is below the lead time, that product will run out before any reorder can land. The stockout probability turns that into a number you can rank on. The anomaly flag is the human safety valve, so a sudden spike gets a set of eyes instead of a silent reorder.

### plan.py

`plan.py` pulls it together into one frame: the SKUs that need action, with the recommended quantity, the reorder point, the days of cover, the stockout risk, the supplier, and a short reason code for each (low cover, anomaly, seasonal ramp). That frame is the input to the reasoning layer, and on its own it is already a complete plan a person could act on.

### charts.py

The three charts are already built for the brief and carry straight over: the per SKU forecast with its band, the inventory sawtooth showing a reorder landing before the buffer, and the ABC Pareto. In the live system you render the current ones from real data and attach the relevant image to the Slack message.

## The reasoning layer

`reason.py` and `llm/nim_client.py`. One rule sits above this whole layer: the model never invents a number. The quantities, reorder points, and risks all come from Python. NIM only ranks the orders, groups them by supplier into a clean purchase order, and writes the plain language summary.

```python
def build_messages(plan_records):
    """Pass the computed plan as JSON. Ask only for ordering,
    grouping, and prose. Require a JSON reply with keys
    ranked_orders, purchase_orders, summary."""
    ...

def reason_over_plan(plan_records):
    raw = call_nim(build_messages(plan_records))
    result = parse_json(raw)
    validate_quantities_unchanged(plan_records, result)  # repair or reject
    return result
```

The validation step is small but it is what makes the output trustworthy and it is a good line for the repo. After the model replies, you check that every quantity it returned equals the quantity you sent. If anything drifted, you repair it from the source numbers or reject and retry.

The wrapper handles the free tier reality:

```python
import os, time, requests

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

def call_nim(messages, model="meta/llama-3.1-70b-instruct", retries=2):
    for attempt in range(retries + 1):
        try:
            r = requests.post(NIM_URL, headers=_headers(),
                              json={"model": model, "messages": messages},
                              timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                return call_fallback(messages)   # second OpenAI style provider
```

Two things keep you inside the rate limit. Send the whole plan in one call rather than one call per SKU. And fall back to a second OpenAI style endpoint on a 429 or a timeout. This wrapper is one of the reusable pieces for the rest of your stack.

## The API service and the MCP server

`api/main.py` is a thin FastAPI app so n8n and the MCP server both talk to the same engine.

```python
# POST /run     run the full pipeline on a source, return plan plus chart paths
# GET  /plan    return the latest plan
# GET  /sku/{sku}   forecast, policy, and risk for one product
```

`mcp/server.py` exposes the same engine as tools, so you or a client can ask Claude about a store in plain language and it calls the math.

```python
# tool get_reorder_plan()          this week's plan with reasons
# tool forecast_sku(sku)           the demand path and band for one product
# tool explain_sku(sku)            why this product is or is not on the list
# tool simulate(sku, scenario)     what changes if lead time or demand shifts
```

The MCP layer is the reason this build feeds the shared stack rather than sitting alone. The forecasting module, the policy engine, and these tools are all reusable in the next project, and the NIM wrapper goes with them.

## The n8n workflow

The orchestrator stays thin. The engine does the thinking and returns one finished payload. The nodes in order:

1. **Schedule trigger.** Fires daily, for example at six in the morning store time.
2. **Shopify pull.** Get products and live inventory, and get orders since the last run. On the first run, pull the full history once. Use GraphQL bulk operations for a large catalog.
3. **Call the engine.** POST the pulled data to `/run`. The engine forecasts, computes the plan, calls NIM, and returns the ranked orders, the purchase order, the summary, and a chart path.
4. **Slack post.** Send a Block Kit message with the summary, the top orders, the chart, and approve and reject buttons.
5. **Wait for the webhook.** Slack interactivity points at an n8n webhook node, so the button click comes back into the flow.
6. **Switch on the action.** On approve, write the order back to Shopify as a draft order or metafield, adjust expected stock, send the purchase order to the supplier, and log the run. On reject, log and stop.
7. **Confirm.** Post a short confirmation back to the Slack thread.

Read the API placement against these steps and it stays honest: Shopify in steps two and six, NIM inside step three, Slack in steps four through seven.

## What the owner configures

Sensible defaults that work, all overridable per SKU through the products CSV: lead time in days and its variability per supplier, unit cost, the fixed order cost per purchase order, the holding cost rate (around 25 percent a year is a fair default), the review period, the minimum order quantity, and the service levels per tier. A store with no lead time history just sets one number per supplier and the math still runs.

## The three day build

**Day 1, the engine core and the showcase.** Set up the repo, the environment, and the dependencies. Build `data.py` to load the sample CSV and the demand series, then `features.py` and `forecast.py` with the baseline, the gradient boosting model, the walk forward backtest, the accuracy numbers, and the band. Reuse `charts.py`. Finish the day with `scripts/backtest.py` printing a real MAE and MAPE on sample data with the forecast plot. That script is the ML showcase and the first LinkedIn screenshot.

**Day 2, the decision layer and the service.** Build `segment.py`, `policy.py`, and `risk.py`, then `plan.py` to assemble the frame. Add `llm/nim_client.py` and `reason.py`, and confirm the quantity validation holds so the model cannot change numbers. Stand up `api/main.py` and `mcp/server.py`. Finish the day with `POST /run` returning the full plan, the drafted purchase order, and the summary on sample data. Demo mode is now complete end to end with no store and no Slack.

**Day 3, orchestration and the live path.** Build the n8n workflow: schedule, Shopify pull, call the engine, Slack approval, the webhook, and the write back. Wire the Block Kit message and the interactivity webhook. Spin up a free Shopify development store with sample products and test the connected path. Write the README, this architecture file, the `.env.example`, the sample data, and the one command demo. Record the sixty second screen capture of a forecast turning into a Slack approval and a write back. The repo is ready to post.

## Things that will bite you

1. **Cold start SKUs.** A product with almost no history will make the tree model guess wildly. Fall back to a category average or a simple moving average until it has enough data.
2. **Shopify rate limits.** Use GraphQL bulk operations and cursor pagination for order history, cache the demand series, and pull only incremental orders after the first run.
3. **NIM rate limits.** One call per run, not per SKU, plus the fallback provider. This is the single biggest cost saver.
4. **The model touching numbers.** Always validate that returned quantities match the input. Treat the model as a writer and a ranker, never as the calculator.
5. **Double ordering on a rerun.** Tag or track what has already been ordered so a rerun of the workflow does not create the same purchase order twice.
6. **Day boundaries.** Aggregate orders to the store's local day, or the demand series drifts by a day and the weekly pattern smears.

## What to show

The repo and the post lead with three things, in this order. The inventory sawtooth, because it shows the decision at a glance. The accuracy report from `backtest.py`, because it proves the forecast is tested rather than assumed. The sixty second capture of forecast to Slack approval to write back, because it shows the whole loop working with a person in control. The free CSV stockout read is the hook that turns a viewer into a conversation.