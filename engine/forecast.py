"""Two models, one honest test.

The baseline is exponential smoothing (level + trend + weekly season) and it
sets a floor. The main model is a gradient-boosting regressor on the feature
matrix, which picks up promotions and nonlinear patterns the baseline misses.
We pick the winner per SKU from a walk-forward backtest, so a quiet product
keeps the simple model and a busy one gets the stronger one.

The 90% band comes from two quantile models, at the 5th and 95th percentiles.
That band is not decoration: it feeds safety stock and it is the shaded area
on the chart.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config
from .features import FEATURE_COLUMNS, make_features_for_sku

warnings.filterwarnings("ignore")  # statsmodels is chatty on short/edge series


# ── model builders ─────────────────────────────────────────────────
def _new_gbm(quantile: Optional[float] = None) -> HistGradientBoostingRegressor:
    if quantile is None:
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=None,
            min_samples_leaf=20, l2_regularization=1.0, random_state=0)
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, max_iter=200, learning_rate=0.05,
        min_samples_leaf=20, l2_regularization=1.0, random_state=0)


def _fit_gbm_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit the median model plus the 5th/95th quantile models for the band."""
    mean = _new_gbm()
    mean.fit(X, y)
    low = _new_gbm(quantile=0.05)
    low.fit(X, y)
    high = _new_gbm(quantile=0.95)
    high.fit(X, y)
    return {"mean": mean, "low": low, "high": high}


def _exp_smoothing_forecast(units: pd.Series, horizon: int) -> np.ndarray:
    """Holt-Winters baseline. Falls back to a moving average when the series
    is too short for a weekly season."""
    y = units.astype(float).values
    if len(y) >= 21 and np.count_nonzero(y) >= 10:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(
                np.clip(y, 0, None) + 1e-6,
                trend="add", seasonal="add", seasonal_periods=7,
                initialization_method="estimated")
            fit = model.fit(optimized=True)
            fc = np.asarray(fit.forecast(horizon))
            return np.clip(fc, 0, None)
        except Exception:
            pass
    # cold-start / fallback: trailing 28-day mean
    base = float(np.mean(y[-28:])) if len(y) else 0.0
    return np.full(horizon, max(base, 0.0))


# ── recursive multi-step GBM forecast ──────────────────────────────
def _recursive_gbm(series_g: pd.DataFrame, models: dict, horizon: int,
                   future_promo: Optional[list] = None) -> dict:
    """Step day-by-day into the future, feeding each prediction back in as the
    next lag so the model forecasts a coherent path rather than one flat number.
    Evaluates the quantile models on the same future rows to get the band."""
    sku = series_g["sku"].iloc[0]
    last_price = float(series_g["price"].iloc[-1]) if "price" in series_g else 0.0
    # only the last ~60 rows matter for the longest lag (28) and rolling (28);
    # trimming here keeps the per-step feature rebuild cheap.
    work = series_g[["date", "sku", "units", "on_promo", "price"]].tail(64).copy()
    last_date = pd.to_datetime(work["date"].max())

    path, lows, highs = [], [], []
    for h in range(1, horizon + 1):
        promo = 0
        if future_promo is not None and h - 1 < len(future_promo):
            promo = int(future_promo[h - 1])
        nxt = {"date": last_date + pd.Timedelta(days=h), "sku": sku,
               "units": 0.0, "on_promo": promo, "price": last_price}
        ext = pd.concat([work, pd.DataFrame([nxt])], ignore_index=True)
        feats = make_features_for_sku(ext)
        x = feats.iloc[[-1]][FEATURE_COLUMNS]

        yhat = max(0.0, float(models["mean"].predict(x)[0]))
        lo = max(0.0, float(models["low"].predict(x)[0]))
        hi = max(yhat, float(models["high"].predict(x)[0]))
        path.append(yhat)
        lows.append(lo)
        highs.append(hi)

        nxt["units"] = yhat            # feed prediction back as next lag
        work = pd.concat([work, pd.DataFrame([nxt])], ignore_index=True).tail(64)

    return {"path": np.array(path), "lower": np.array(lows), "upper": np.array(highs)}


# ── accuracy ───────────────────────────────────────────────────────
def _score(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = np.abs(actual - pred)
    mae = float(np.mean(err))
    # aggregate (WAPE-style) percentage error — well-defined even with zero-days
    denom = max(float(np.sum(actual)), 1.0)
    mape = float(100.0 * np.sum(err) / denom)
    return {"mae": round(mae, 3), "mape": round(mape, 2)}


def backtest_sku(series_g: pd.DataFrame, horizon: int) -> dict:
    """Walk-forward validation with an expanding window. Train baseline and GBM
    on each fold, forecast the next `horizon` days, score both on the held-out
    window. Returns MAE and MAPE for each model and which one won."""
    g = make_features_for_sku(series_g)
    n = len(g)
    folds = config.BACKTEST_FOLDS
    min_train = config.BACKTEST_MIN_TRAIN_DAYS

    if n < min_train + horizon:
        return {"baseline": None, "gbm": None, "winner": "baseline",
                "folds": 0, "note": "insufficient history for backtest"}

    # evenly spaced fold split points across the tail of the series
    first_split = max(min_train, n - folds * horizon)
    splits = list(range(first_split, n - horizon + 1,
                        max(1, (n - horizon - first_split) // max(folds - 1, 1))))[:folds]

    base_err, gbm_err = [], []
    for split in splits:
        train = g.iloc[:split]
        actual = g.iloc[split:split + horizon]["units"].values.astype(float)
        if len(actual) < horizon:
            continue
        # baseline
        base_fc = _exp_smoothing_forecast(train["units"], horizon)
        base_err.append((actual, base_fc[:len(actual)]))
        # gbm (use the mean model only for scoring; quantiles are for the band)
        X = train[FEATURE_COLUMNS]
        y = train["units"].astype(float)
        try:
            mean = _new_gbm()
            mean.fit(X, y)
            gbm_fc = _recursive_gbm(train[["date", "sku", "units", "on_promo", "price"]],
                                    {"mean": mean, "low": mean, "high": mean}, horizon)["path"]
            gbm_err.append((actual, gbm_fc[:len(actual)]))
        except Exception:
            gbm_err.append((actual, base_fc[:len(actual)]))

    def _agg(pairs):
        a = np.concatenate([p[0] for p in pairs])
        p = np.concatenate([p[1] for p in pairs])
        return _score(a, p)

    base_score = _agg(base_err) if base_err else None
    gbm_score = _agg(gbm_err) if gbm_err else None

    winner = "baseline"
    if base_score and gbm_score:
        winner = "gbm" if gbm_score["mae"] <= base_score["mae"] else "baseline"
    elif gbm_score:
        winner = "gbm"

    return {"baseline": base_score, "gbm": gbm_score,
            "winner": winner, "folds": len(base_err)}


# ── the forecast the rest of the engine consumes ───────────────────
def forecast_sku(series_g: pd.DataFrame, horizon: int,
                 future_promo: Optional[list] = None,
                 backtest: Optional[dict] = None) -> dict:
    """Over the lead-time horizon, return:
       demand_path   predicted units per day
       demand_mean   average daily demand, for the policy layer
       demand_std    demand variability, for safety stock
       lower, upper  the 90% band
       model_used    whichever model won the backtest for this SKU
       resid_std     residual std on recent history, for the anomaly flag
    """
    series_g = series_g.sort_values("date").reset_index(drop=True)
    if backtest is None:
        backtest = backtest_sku(series_g, horizon)
    winner = backtest.get("winner", "baseline")

    feats = make_features_for_sku(series_g)
    X = feats[FEATURE_COLUMNS]
    y = feats["units"].astype(float)

    # always fit the GBM (we need its quantile band even if baseline wins point)
    enough = len(series_g) >= config.COLD_START_MIN_DAYS and y.sum() > 0
    if enough:
        try:
            models = _fit_gbm_models(X, y)
            gbm = _recursive_gbm(series_g, models, horizon, future_promo)
            resid = y.values - models["mean"].predict(X)
            resid_std = float(np.std(resid[-56:])) if len(resid) else 1.0
        except Exception:
            enough = False

    base_path = _exp_smoothing_forecast(series_g["units"], horizon)

    if winner == "gbm" and enough:
        path, lower, upper, model_used = gbm["path"], gbm["lower"], gbm["upper"], "gbm"
    else:
        path = base_path
        # spread a band around the baseline using recent demand variability
        recent_std = float(series_g["units"].tail(56).std()) or 1.0
        lower = np.clip(path - 1.64 * recent_std, 0, None)
        upper = path + 1.64 * recent_std
        model_used = "baseline"
        if not enough:
            model_used = "baseline_coldstart"
        resid_std = recent_std

    demand_mean = float(np.mean(path))
    # demand variability for safety stock: blend the band width with history
    demand_std = float(max(np.std(path), (np.mean(upper - lower) / 3.29), 1e-6))

    return {
        "sku": series_g["sku"].iloc[0],
        "horizon": horizon,
        "demand_path": [round(float(v), 3) for v in path],
        "demand_mean": round(demand_mean, 3),
        "demand_std": round(demand_std, 3),
        "lower": [round(float(v), 3) for v in lower],
        "upper": [round(float(v), 3) for v in upper],
        "model_used": model_used,
        "resid_std": round(float(resid_std), 3),
        "accuracy": backtest,
    }
