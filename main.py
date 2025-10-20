from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import time

import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI(title="Trading Backend")

# CORS – dein Frontend erlauben
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trading-frontend-coje.onrender.com",
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Mini-Cache (5 Minuten) --------
CACHE_TTL = 300  # Sekunden
CACHE: Dict[str, Tuple[float, dict]] = {}

def _ck(ticker: str, period: str, interval: str) -> str:
    return f"{ticker}:{period}:{interval}"

def _get_cache(k: str) -> Optional[dict]:
    it = CACHE.get(k)
    if not it:
        return None
    ts, payload = it
    if time.time() - ts < CACHE_TTL:
        return payload
    CACHE.pop(k, None)
    return None

def _set_cache(k: str, payload: dict) -> None:
    CACHE[k] = (time.time(), payload)

# -------- Indikatoren --------
def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)

    # RSI(14)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    r_up = up.rolling(14, min_periods=14).mean()
    r_dn = down.rolling(14, min_periods=14).mean().replace(0, np.nan)
    rs = r_up / r_dn
    df["RSI"] = 100 - 100 / (1 + rs)

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_signal"] = signal

    # Bollinger(20, 2σ)
    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20

    # Stochastic(14, 3)
    low14 = df["Low"].rolling(14, min_periods=14).min()
    high14 = df["High"].rolling(14, min_periods=14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14)
    df["Stoch_K"] = stoch_k
    df["Stoch_D"] = stoch_k.rolling(3, min_periods=3).mean()

    return df

def _download_once(ticker: str, period: str, interval: str) -> Optional[dict]:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if df is None or df.empty:
            return None

        df = df.reset_index()
        # Datumsspalte vereinheitlichen
        if "Date" in df.columns:
            df.rename(columns={"Date": "datetime"}, inplace=True)
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "datetime"}, inplace=True)
        else:
            df.rename(columns={df.columns[0]: "datetime"}, inplace=True)

        # Indikatoren anhängen
        df = _add_indicators(df)

        payload = {
            "index": df["datetime"].astype(str).tolist(),
            "Open": df["Open"].round(2).tolist(),
            "High": df["High"].round(2).tolist(),
            "Low": df["Low"].round(2).tolist(),
            "Close": df["Close"].round(2).tolist(),
            "Volume": df.get("Volume", pd.Series([0]*len(df))).fillna(0).astype(int).tolist(),
            "RSI": np.nan_to_num(df["RSI"]).round(2).tolist(),
            "MACD": np.nan_to_num(df["MACD"]).round(4).tolist(),
            "MACD_signal": np.nan_to_num(df["MACD_signal"]).round(4).tolist(),
            "BB_upper": np.nan_to_num(df["BB_upper"]).round(2).tolist(),
            "BB_lower": np.nan_to_num(df["BB_lower"]).round(2).tolist(),
            "Stoch_K": np.nan_to_num(df["Stoch_K"]).round(2).tolist(),
            "Stoch_D": np.nan_to_num(df["Stoch_D"]).round(2).tolist(),
        }
        return payload
    except Exception:
        return None

def _fallback_plan(req_period: Optional[str], req_interval: Optional[str]):
    order = []
    if req_period and req_interval:
        order.append((req_period, req_interval))
    if req_period:
        order.append((req_period, "1d"))
    # robuste Defaults – erfahrungsgemäß zuverlässig
    order += [
        ("3mo", "1d"),
        ("6mo", "1d"),
        ("1y", "1d"),
        ("2y", "1wk"),
        ("5y", "1wk"),
        ("10y", "1mo"),
    ]
    # deduplizieren
    seen = set()
    plan = []
    for p, i in order:
        if (p, i) not in seen:
            plan.append((p, i))
            seen.add((p, i))
    return plan

@app.get("/")
def root():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}

@app.get("/api/stock")
def api_stock(
    ticker: str,
    period: Optional[str] = Query(default=None),
    interval: Optional[str] = Query(default=None),
):
    t = ticker.upper().strip()
    for p, i in _fallback_plan(period, interval):
        ck = _ck(t, p, i)
        cached = _get_cache(ck)
        if cached:
            out = dict(cached)
            out.update({"used_period": p, "used_interval": i, "source": "cache"})
            return out

        data = _download_once(t, p, i)
        if data:
            data.update({"used_period": p, "used_interval": i, "source": "live"})
            _set_cache(ck, dict(data))
            return data

    return JSONResponse({"error": "Keine Daten gefunden", "ticker": t}, status_code=404)
