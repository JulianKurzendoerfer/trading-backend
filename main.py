from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import time, io, requests, pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tool.market-vision-pro.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = {}
_TTL = 300

def _now(): return int(time.time())

def _get_cache(k):
    v = _cache.get(k)
    if not v: return None
    exp, payload = v
    if _now() > exp: 
        _cache.pop(k, None)
        return None
    return payload

def _set_cache(k, payload):
    _cache[k] = (_now() + _TTL, payload)

def _stooq_symbol(t: str) -> str:
    t = t.lower().strip()
    if "." in t: 
        return t
    if not t.endswith(".us"):
        t = f"{t}.us"
    return t

def _stooq_df(ticker: str):
    sym = _stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, headers={
            "User-Agent":"Mozilla/5.0",
            "Accept":"text/csv",
            "Cache-Control":"no-cache",
        }, timeout=12)
        if r.status_code != 200 or "Date" not in r.text:
            return None
        df = pd.read_csv(io.StringIO(r.text))
        if df is None or df.empty or "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        df = df.dropna(subset=["Date"])
        df = df[["Date","open","high","low","close","volume"]]
        df = df.sort_values("Date")
        df = df.reset_index(drop=True)
        return df
    except Exception:
        return None

try:
    import yfinance as yf
except Exception:
    yf = None

def _yf_df(ticker: str, period: str, interval: str):
    if yf is None:
        return None
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        if "Datetime" in df.columns:
            df = df.reset_index().rename(columns={"Datetime":"Date"})
        elif "Date" not in df.columns:
            df = df.reset_index().rename(columns={"index":"Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df[["Date","open","high","low","close","volume"]]
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except Exception:
        return None

def _points(df: pd.DataFrame):
    out = []
    for _, r in df.iterrows():
        t = r.get("Date")
        try:
            ts = t.isoformat() if hasattr(t, "isoformat") else str(t)
        except Exception:
            ts = str(t)
        out.append({
            "time": ts,
            "open": float(r.get("open", 0) or 0),
            "high": float(r.get("high", 0) or 0),
            "low": float(r.get("low", 0) or 0),
            "close": float(r.get("close", 0) or 0),
            "volume": float(r.get("volume", 0) or 0),
        })
    return out

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/api/health", "/api/stock?ticker=AAPL&period=1y&interval=1d"]}

@app.get("/api/health")
def health():
    return {"ok": True, "ts": _now()}

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
        return {"ticker": ticker.upper(), "points": []}

    pts = _points(df)
    payload = {"ticker": ticker.upper(), "points": pts}
    _set_cache(key, payload)
    return payload
