from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd, time
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

@app.get("/api/health")
@app.head("/api/health")
def health():
    return {"ok": True}

def _records(df):
    if df is None or len(df) == 0:
        return []
    df = df.copy()
    ts_col = None
    for c in ("Date","Datetime","date","time","Time"):
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        df = df.reset_index()
        for c in ("Date","Datetime","date","time","index"):
            if c in df.columns:
                ts_col = c
                break
    out = []
    for _, r in df.iterrows():
        t = r[ts_col]
        t_iso = t.isoformat() if hasattr(t,"isoformat") else str(t)
        out.append({
            "time": t_iso,
            "open": float(r.get("Open", r.get("open", 0)) or 0),
            "high": float(r.get("High", r.get("high", 0)) or 0),
            "low": float(r.get("Low", r.get("low", 0)) or 0),
            "close": float(r.get("Close", r.get("close", 0)) or 0),
            "volume": float(r.get("Volume", r.get("volume", 0)) or 0),
        })
    return out

def _yf(ticker, period, interval):
    if yf is None:
        return None
    last = None
    for _ in range(3):
        try:
            df = yf.download(
                ticker, period=period, interval=interval,
                progress=False, auto_adjust=False, prepost=False, threads=False
            )
            if df is not None and not df.empty:
                if "Adj Close" in df.columns and "Close" not in df.columns:
                    df = df.rename(columns={"Adj Close":"Close"})
                return df.reset_index()
        except Exception as e:
            last = e
        time.sleep(1)
    return None

def _stooq(ticker):
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
        df = pd.read_csv(url)
        if df is None or df.empty:
            return None
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    except Exception:
        return None

@app.get("/api/stock")
def stock(ticker: str, period: str="1y", interval: str="1d"):
    df = _yf(ticker, period, interval)
    if df is None or df.empty:
        df = _stooq(ticker)
    if df is None or df.empty:
        return {"ticker": ticker.upper(), "points": []}
    return {"ticker": ticker.upper(), "points": _records(df)}
