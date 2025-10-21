from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import time, io
import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:
    yf = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tool.market-vision-pro.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = {}
_TTL = 180

def _now():
    return int(time.time())

def _get_cache(key):
    v = _cache.get(key)
    if not v:
        return None
    ts, payload = v
    if _now() - ts > _TTL:
        _cache.pop(key, None)
        return None
    return payload

def _set_cache(key, payload):
    _cache[key] = (_now(), payload)

def _to_points(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    tcol = None
    for c in ("Datetime","Date","date","Time","time"):
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        return []
    rec = []
    for _, r in df.iterrows():
        t = r.get(tcol)
        try:
            ts = t.isoformat() if hasattr(t, "isoformat") else pd.to_datetime(t).isoformat()
        except Exception:
            continue
        rec.append({
            "time": ts,
            "open": float(r.get("Open", r.get("open", 0)) or 0),
            "high": float(r.get("High", r.get("high", 0)) or 0),
            "low":  float(r.get("Low",  r.get("low",  0)) or 0),
            "close":float(r.get("Close",r.get("close",0)) or 0),
            "volume":float(r.get("Volume",r.get("volume",0)) or 0),
        })
    return rec

def _yf_df(ticker: str, period: str, interval: str):
    if yf is None:
        return None
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False, prepost=False)
        if df is None or df.empty:
            return None
        return df.reset_index()
    except Exception:
        return None

def _stooq_df(ticker: str):
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
    try:
        r = requests.get(url, headers={
            "User-Agent":"Mozilla/5.0",
            "Accept":"text/csv",
            "Cache-Control":"no-cache",
        }, timeout=12)
        if r.status_code != 200 or "Date,Open,High,Low,Close,Volume" not in r.text:
            return None
        df = pd.read_csv(io.StringIO(r.text))
        if df is None or df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    except Exception:
        return None

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/api/health", "/api/stock?ticker=AAPL&period=1y&interval=1d"]}

@app.head("/")
def root_head():
    return Response(status_code=200)

@app.get("/api/health")
def health():
    return {"ok": True}

@app.head("/api/health")
def health_head():
    return Response(status_code=200)

@app.get("/api/stock")
def stock(ticker: str, period: str = "1y", interval: str = "1d", force: int = 0):
    key = f"{ticker}:{period}:{interval}"
    if not force:
        cached = _get_cache(key)
        if cached:
            return cached

    df = _stooq_df(ticker)
    if df is None or df.empty:
        df = _yf_df(ticker, period, interval)

    points = _to_points(df)
    payload = {"ticker": ticker.upper(), "points": points}

    if points:
        _set_cache(key, payload)

    return payload
