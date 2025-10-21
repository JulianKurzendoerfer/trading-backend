from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import requests, io, time, datetime as dt
import pandas as pd

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

_CACHE = {}
_TTL = 180

def _now(): return int(time.time())

def _build_points(df: pd.DataFrame):
    if df is None or df.empty: 
        return []
    if "Date" in df.columns:
        tcol = "Date"
    elif "Datetime" in df.columns:
        tcol = "Datetime"
    elif "time" in df.columns:
        tcol = "time"
    else:
        return []
    rec = []
    for _, r in df.iterrows():
        t = r.get(tcol)
        if hasattr(t, "isoformat"):
            ts = t.isoformat()
        else:
            try:
                ts = pd.to_datetime(t).isoformat()
            except Exception:
                continue
        rec.append({
            "time": ts,
            "open": float(r.get("Open", r.get("open", 0)) or 0),
            "high": float(r.get("High", r.get("high", 0)) or 0),
            "low":  float(r.get("Low",  r.get("low", 0))  or 0),
            "close":float(r.get("Close",r.get("close",0)) or 0),
            "volume":float(r.get("Volume",r.get("volume",0)) or 0),
        })
    return rec

def _stooq_df(ticker: str):
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent":"Mozilla/5.0",
                "Accept":"text/csv",
                "Cache-Control":"no-cache",
            },
            timeout=12,
        )
        if r.status_code != 200 or not r.text or "Date,Open,High,Low,Close,Volume" not in r.text.splitlines()[0]:
            return None
        df = pd.read_csv(io.StringIO(r.text))
        if df is None or df.empty or "Date" not in df.columns: 
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).rename(
            columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}
        )
        return df
    except Exception:
        return None

def _yf_df(ticker: str, period: str, interval: str):
    if yf is None:
        return None
    try:
        h = yf.Ticker(ticker).history(period=period or "1y", interval=interval or "1d", auto_adjust=False)
        if h is None or h.empty:
            return None
        h = h.rename(columns={"Adj Close":"Close"}).reset_index()
        return h
    except Exception:
        return None

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/api/health", "/api/stock?ticker=AAPL&period=1y&interval=1d"]}

@app.head("/api/health")
@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/stock")
def stock(ticker: str, period: str = "1y", interval: str = "1d", request: Request = None):
    key = (ticker.upper(), period, interval)
    hit = _CACHE.get(key)
    if hit and _now() - hit["ts"] < _TTL:
        return hit["data"]

    df = _stooq_df(ticker)
    if df is None or df.empty:
        df = _yf_df(ticker, period, interval)

    pts = _build_points(df)
    if not pts:
        raise HTTPException(status_code=502, detail="no data")

    data = {"ticker": ticker.upper(), "points": pts}
    _CACHE[key] = {"ts": _now(), "data": data}
    return data
