from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import requests, io, time
import pandas as pd
from datetime import datetime as dt

try:
    import yfinance as yf
except Exception:
    yf = None

app = FastAPI()

ORIGIN = "https://tool.market-vision-pro.com"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = {}
_TTL = 180

def _now(): return int(time.time())

def _get_cache(k):
    v = _cache.get(k)
    if not v: return None
    ts, data = v
    if _now() - ts <= _TTL: return data
    _cache.pop(k, None)
    return None

def _set_cache(k, data):
    _cache[k] = (_now(), data)

def _to_points(df: pd.DataFrame):
    if df is None or df.empty: return []
    tcol = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
    if not tcol: return []
    rec = []
    for _, r in df.iterrows():
        t = r.get(tcol)
        try:
            ts = t.isoformat() if hasattr(t, "isoformat") else pd.to_datetime(t).isoformat()
        except Exception:
            continue
        rec.append({
            "time": ts,
            "open": float(r.get("Open", 0) or 0),
            "high": float(r.get("High", 0) or 0),
            "low":  float(r.get("Low",  0) or 0),
            "close":float(r.get("Close",0) or 0),
            "volume": float(r.get("Volume",0) or 0),
        })
    return rec

def _stooq_df(ticker: str):
    sym = ticker.lower()
    if "." not in sym:
        sym = f"{sym}.us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv",
            "Cache-Control": "no-cache",
        },
        timeout=12,
    )
    if r.status_code != 200: return None
    txt = r.text.strip()
    if not txt or txt.startswith("<"): return None
    df = pd.read_csv(io.StringIO(txt))
    if "Date" not in df.columns: return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    cols = {c.lower(): c for c in df.columns}
    df = df.rename(columns={
        cols.get("open","Open"): "Open",
        cols.get("high","High"): "High",
        cols.get("low","Low"): "Low",
        cols.get("close","Close"): "Close",
        cols.get("volume","Volume"): "Volume",
    })
    return df[["Date","Open","High","Low","Close","Volume"]]

def _yf_df(ticker: str, period: str, interval: str):
    if yf is None: return None
    try:
        h = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False, actions=False)
        if h is None or h.empty: return None
        h = h.reset_index()
        if "Date" not in h.columns and "Datetime" in h.columns:
            h = h.rename(columns={"Datetime":"Date"})
        h = h.rename(columns={
            "Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"
        })
        return h[["Date","Open","High","Low","Close","Volume"]]
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
    return {"ok": True, "ts": dt.utcnow().isoformat()}

@app.head("/api/health")
def health_head():
    return Response(status_code=200)

@app.get("/api/stock")
def stock(ticker: str, period: str = "1y", interval: str = "1d"):
    key = f"{ticker}|{period}|{interval}"
    cached = _get_cache(key)
    if cached: return cached
    df = _stooq_df(ticker)
    if df is None or df.empty:
        df = _yf_df(ticker, period, interval)
    points = _to_points(df)
    payload = {"ticker": ticker.upper(), "points": points}
    _set_cache(key, payload)
    return payload
