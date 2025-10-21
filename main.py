from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import yfinance as yf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tool.market-vision-pro.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.api_route("/api/health", methods=["GET","HEAD"])
@app.api_route("/health", methods=["GET","HEAD"])
def health():
    return {"ok": True}

@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "endpoints": ["/api/health", "/api/stock?ticker=AAPL&period=1y&interval=1d"]}

def _records(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    df = df.reset_index()
    ts_col = next((c for c in ["Date","Datetime","date","Time","time"] if c in df.columns), None)
    if ts_col is None:
        return []
    out = []
    for _, r in df.iterrows():
        t = r[ts_col]
        t = t.isoformat() if hasattr(t, "isoformat") else str(t)
        out.append({
            "time": t,
            "open": float(r.get("Open", 0) or 0),
            "high": float(r.get("High", 0) or 0),
            "low": float(r.get("Low", 0) or 0),
            "close": float(r.get("Close", 0) or 0),
            "volume": float(r.get("Volume", 0) or 0),
        })
    return out

def _yf(ticker: str, period: str, interval: str):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    return None

def _stooq(ticker: str):
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
        df = pd.read_csv(url)
        if df is None or df.empty or "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}, inplace=True)
        return df
    except Exception:
        return None

@app.get("/api/stock")
def stock(ticker: str, period: str="1y", interval: str="1d"):
    df = _yf(ticker, period, interval)
    if df is None or df.empty:
        df = _stooq(ticker)
    return {"ticker": ticker.upper(), "points": _records(df)}
