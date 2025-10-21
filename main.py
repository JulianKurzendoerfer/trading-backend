from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests, io
from datetime import datetime as dt, timedelta

try:
    import yfinance as yf  # fallback
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
    return dt.utcnow()

def _get_cache(key: str):
    v = _cache.get(key)
    if not v:
        return None
    data, ts = v
    if _now() - ts > timedelta(seconds=_TTL):
        _cache.pop(key, None)
        return None
    return data

def _set_cache(key: str, data):
    _cache[key] = (data, _now())

def _to_points(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    cols = {c.lower(): c for c in df.columns}
    tcol = None
    for k in ["date", "datetime", "time"]:
        if k in cols:
            tcol = cols[k]
            break
    if tcol is None:
        return []
    def _col(*names):
        for n in names:
            c = cols.get(n.lower())
            if c:
                return c
        return None
    c_open  = _col("Open")
    c_high  = _col("High")
    c_low   = _col("Low")
    c_close = _col("Close")
    c_vol   = _col("Volume","Vol")
    rec = []
    for _, r in df.iterrows():
        t = r.get(tcol)
        if hasattr(t, "isoformat"):
            ts = t.isoformat()
        else:
            ts = str(t)
        def fv(c): 
            try:
                return float(r.get(c, 0)) if c else 0.0
            except Exception:
                return 0.0
        v = r.get(c_vol, 0)
        try:
            v = float(v) if v == v else 0.0
        except Exception:
            v = 0.0
        rec.append({
            "time": ts,
            "open":  fv(c_open),
            "high":  fv(c_high),
            "low":   fv(c_low),
            "close": fv(c_close),
            "volume": v,
        })
    return rec

def _stooq_df(ticker: str):
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/csv",
                "Cache-Control": "no-cache",
            },
            timeout=12,
        )
        if r.status_code != 200 or "Date,Open,High,Low,Close,Volume" not in r.text.splitlines()[0]:
            return None
        df = pd.read_csv(io.StringIO(r.text))
        if df is None or df.empty or "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
        return df
    except Exception:
        return None

def _yf_df(ticker: str, period: str, interval: str):
    if yf is None:
        return None
    try:
        hist = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if hist is None or hist.empty:
            return None
        df = hist.reset_index()
        tcol = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
        if tcol:
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        return df[ [c for c in [tcol,"Open","High","Low","Close","Volume"] if c in df.columns] ]
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
    if cached:
        return cached
    df = _stooq_df(ticker)
    if df is None or df.empty:
        df = _yf_df(ticker, period, interval)
    if df is None or df.empty:
        raise HTTPException(status_code=502, detail="no data")
    points = _to_points(df)
    payload = {"ticker": ticker.upper(), "points": points}
    _set_cache(key, payload)
    return payload
