from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import time, io, datetime as dt
import requests
import pandas as pd

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
_TTL = 120

def _now():
    return int(time.time())

def _period_days(p: str) -> int:
    p = (p or "").lower()
    if p.endswith("y"): return int(p[:-1] or 1) * 365
    if p.endswith("mo"): return int(p[:-2] or 1) * 30
    if p.endswith("d"): return int(p[:-1] or 1)
    return 365

def _slice_period(df: pd.DataFrame, period: str):
    try:
        days = _period_days(period)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
        if "Date" in df.columns:
            df = df[df["Date"] >= cutoff]
        elif "Datetime" in df.columns:
            df = df[df["Datetime"] >= cutoff]
    except Exception:
        pass
    return df

def _stooq(ticker: str):
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        if r.status_code != 200 or not r.text or not r.text.strip():
            return None
        df = pd.read_csv(io.StringIO(r.text))
        if df is None or df.empty or "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
        return df
    except Exception:
        return None

def _yfin(ticker: str, period: str, interval: str):
    if yf is None:
        return None
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
            if "Datetime" in df.columns:
                df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
        # yfinance column names: ['Open','High','Low','Close','Adj Close','Volume']
        want = ["Open","High","Low","Close","Volume"]
        for w in want:
            if w not in df.columns:
                return None
        return df
    except Exception:
        return None

def _records_from_df(df: pd.DataFrame):
    ts_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
    if ts_col is None:
        return []
    out = []
    for _, r in df.iterrows():
        t = r.get(ts_col)
        t_iso = t.isoformat() if hasattr(t, "isoformat") else str(t)
        out.append({
            "time": t_iso,
            "open": float(r.get("Open", 0) or 0),
            "high": float(r.get("High", 0) or 0),
            "low": float(r.get("Low", 0) or 0),
            "close": float(r.get("Close", 0) or 0),
            "volume": float(r.get("Volume", 0) or 0),
        })
    return out

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/api/health", "/api/stock?ticker=AAPL&period=1y&interval=1d"]}

@app.get("/api/health")
def health():
    return {"ok": True}

@app.head("/api/health")
def health_head():
    return Response(status_code=200)

@app.get("/api/stock")
def stock(ticker: str, period: str = "1y", interval: str = "1d"):
    key = (ticker.upper(), period, interval)
    ts = _cache.get(key, (0, None))[0]
    if _now() - ts <= _TTL:
        return _cache[key][1]

    df = _stooq(ticker)
    if df is None or df.empty:
        df = _yfin(ticker, period, interval)
    if df is None or df.empty:
        resp = {"ticker": ticker.upper(), "points": []}
        _cache[key] = (_now(), resp)
        return resp

    df = _slice_period(df, period)
    rec = _records_from_df(df)
    resp = {"ticker": ticker.upper(), "points": rec}
    _cache[key] = (_now(), resp)
    return resp
