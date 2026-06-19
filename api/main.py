"""Thin FastAPI app so n8n and the MCP server both talk to the same engine.

  POST /run        run the full pipeline on a source, return plan + chart paths
  GET  /plan       return the latest plan
  GET  /sku/{sku}  forecast, policy, and risk for one product
  GET  /chart/{name}  serve a rendered chart image (for Slack/n8n)
  GET  /health     liveness

Run it:  uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

from engine import config, data as data_mod
from engine.pipeline import run_pipeline, sku_detail
from api.dashboard import router as dashboard_router

app = FastAPI(title="StockPilot", version="1.0.0",
              description="Demand forecasting and reorder engine")

# Allow the dashboard (a separate origin in dev) to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dashboard-shaped read endpoints under /dashboard.
app.include_router(dashboard_router)

# the latest run, held in memory so /plan and /sku can serve without recomputing
_STATE: dict[str, Any] = {"result": None, "series": None, "products": None}


class RunRequest(BaseModel):
    # connected mode: post records pulled from Shopify ...
    orders: Optional[list[dict]] = None
    products: Optional[list[dict]] = None
    # ... or demo mode: point at CSV paths on the server
    orders_csv: Optional[str] = None
    products_csv: Optional[str] = None
    use_llm: bool = True
    render_charts: bool = True
    horizon: Optional[int] = None


def _client_view(result: dict) -> dict:
    """Drop the heavy in-process objects before returning to the client."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


@app.get("/")
def root() -> dict:
    """Friendly landing — this is an API, not a website. The pages are below."""
    return {
        "service": "StockPilot engine",
        "status": "running",
        "try": {
            "health": "/health",
            "interactive_docs": "/docs",
            "dashboard_summary": "/dashboard/summary",
            "latest_plan": "/plan",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "nim_configured": bool(os.getenv("NIM_API_KEY")),
            "has_run": _STATE["result"] is not None}


@app.post("/run")
def run(req: RunRequest) -> dict:
    """Run forecast -> plan -> reason. Accepts posted records (Shopify pull) or
    CSV paths (demo). Caches the result for /plan and /sku."""
    orders_src = req.orders if req.orders is not None else (req.orders_csv or config.SAMPLE_ORDERS)
    products_src = req.products if req.products is not None else (req.products_csv or config.SAMPLE_PRODUCTS)
    try:
        result = run_pipeline(orders_src, products_src, use_llm=req.use_llm,
                              render_charts=req.render_charts, horizon=req.horizon)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"pipeline failed: {e}")

    _STATE["result"] = result
    _STATE["series"] = result["_series"]
    _STATE["products"] = data_mod.load_products(products_src)
    return _client_view(result)


@app.get("/plan")
def plan() -> dict:
    """Return the latest plan (run /run first)."""
    if _STATE["result"] is None:
        raise HTTPException(status_code=404, detail="no run yet; POST /run first")
    return _client_view(_STATE["result"])


@app.get("/sku/{sku}")
def sku(sku: str, horizon: Optional[int] = None) -> dict:
    """Forecast, policy, and risk for one product."""
    if _STATE["series"] is None:
        raise HTTPException(status_code=404, detail="no run yet; POST /run first")
    try:
        return sku_detail(_STATE["series"], _STATE["products"], sku, horizon=horizon)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown sku {sku!r}")


@app.get("/chart/{name}")
def chart(name: str) -> FileResponse:
    """Serve a rendered chart PNG by file name (no path traversal)."""
    safe = os.path.basename(name)
    if not safe.endswith(".png"):
        safe += ".png"
    path = os.path.join(config.CHART_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"no chart {safe!r}")
    return FileResponse(path, media_type="image/png")
