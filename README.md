# StockPilot

Demand forecasting and automatic reordering for a store. One engine, two modes.

- **Demo mode (CSV).** Point the engine at a CSV of order history. It forecasts
  demand, computes the reorder plan, drafts the purchase order, and writes
  charts to disk. No Shopify, no Slack, no account needed.
- **Connected mode (live store).** n8n runs on a schedule, pulls the store's
  data from Shopify, sends it to the engine, posts the plan to Slack for
  approval, and on a *yes* writes the order back to Shopify. Same engine, real
  data, a person in the loop.

The brain is plain Python that runs anywhere. The plumbing (schedule, store,
chat) is the only part that needs accounts, and it stays thin.

---

## Quickstart (demo mode, 2 commands)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (optional) regenerate the sample data
python scripts/make_sample_data.py

# the one-command demo: forecast -> plan -> draft PO -> charts
python scripts/run_demo.py
```

You'll get a ranked reorder table, a drafted purchase order grouped by
supplier, a plain-language summary, and PNG charts in `charts_out/`.

The accuracy report (the ML showcase):

```bash
python scripts/backtest.py
```

It prints MAE / MAPE per SKU for the baseline vs the gradient-boosting model,
says which one wins, and writes a forecast plot for the top SKU.

---

## The engine (`engine/`)

Each module is small and does one job, so the pipeline reads top to bottom.

| module | job |
|---|---|
| `data.py` | load orders + products, build the continuous daily demand series (zero-days filled) |
| `features.py` | lag, rolling, calendar, promo, days-since-stockout features |
| `forecast.py` | exponential-smoothing baseline + gradient boosting, walk-forward backtest, 90% quantile band |
| `segment.py` | ABC by revenue, XYZ by demand variability → service level + Z per SKU |
| `policy.py` | safety stock, reorder point, EOQ, order quantity |
| `risk.py` | days of cover, stockout probability, anomaly flag |
| `plan.py` | assemble the reorder plan with reason codes |
| `reason.py` | rank + group + summarize via NIM, with a quantity-validation guard |
| `charts.py` | forecast band, inventory sawtooth, ABC Pareto |
| `pipeline.py` | one `run_pipeline()` that ties it all together |
| `config.py` | service levels, costs, lead times, thresholds |

**The one rule of the reasoning layer:** the model never invents a number. The
quantities, reorder points, and risks all come from Python. NIM only ranks the
orders, groups them by supplier, and writes the prose — and every quantity it
returns is checked against the source and repaired if it drifted. If no LLM key
is set, the same output is built deterministically, so demo mode needs no
account.

---

## The math (what actually decides the order)

```
safety_stock  = Z * sqrt(lead_time * demand_std^2 + demand_mean^2 * lead_time_std^2)
reorder_point = demand_mean * lead_time + safety_stock
EOQ           = sqrt(2 * annual_demand * order_cost / holding_cost_per_unit)
order_qty     = 0                       if on_hand + on_order > reorder_point
              = order_up_to - position  otherwise (clamped to MOQ)
```

`Z` and the target service level come from the SKU's ABC class (A=0.98,
B=0.95, C=0.90), so the few products that drive most revenue get a tighter
buffer and the long tail gets a leaner one.

---

## API service (`api/main.py`)

```bash
uvicorn api.main:app --reload --port 8000
```

- `POST /run` — run the pipeline on a source (CSV paths *or* posted Shopify
  records), return plan + reasoning + chart paths.
- `GET /plan` — the latest plan.
- `GET /sku/{sku}` — forecast, policy, and risk for one product.
- `GET /chart/{name}` — a rendered chart PNG.
- `GET /health` — liveness + whether NIM is configured.

Demo it with the sample CSVs:

```bash
curl -X POST localhost:8000/run \
  -H 'content-type: application/json' \
  -d '{"orders_csv":"data/sample_orders.csv","products_csv":"data/sample_products.csv","use_llm":false}'
```

## MCP server (`mcp/server.py`)

Exposes the engine as tools so you can ask Claude (or any MCP client) about a
store in plain language: `get_reorder_plan`, `forecast_sku`, `explain_sku`,
`simulate`. Run with `python mcp/server.py` (stdio).

---

## Connected mode (n8n + Shopify + Slack)

The importable workflow is `n8n/stockpilot_workflow.json`. See
[SETUP_CONNECTED.md](SETUP_CONNECTED.md) for the full step-by-step (the three
API keys, importing the workflow, wiring the Slack buttons).

In short, the workflow does: **schedule → pull Shopify → call engine → post to
Slack with Approve/Reject → on approve, write a draft order back to Shopify.**

---

## Things that will bite you

1. **Cold-start SKUs** with little history → the engine falls back to a moving average.
2. **Shopify rate limits** → pull incrementally (the workflow asks for the last 60 days).
3. **NIM rate limits** → one call per run (not per SKU) + a fallback provider in `llm/nim_client.py`.
4. **The model touching numbers** → always validated against the source in `reason.py`.
5. **Double ordering on a rerun** → draft orders are tagged `stockpilot`; de-dupe before creating.
6. **Day boundaries** → orders are aggregated to whole days so the weekly pattern stays sharp.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.
