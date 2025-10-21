from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import yfinance as yf
import requests, io, time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tool.market-vision-pro.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.api_route("/api/health", methods=["GET","HEAD"])
def health():
    return {"ok": True}

def fetch_yf(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    for _ in range(3):
        try:
            df = yf.download(
                ticker, period=period, interval=interval,
                auto_adjust=False, progress=False, threads=False
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                if getattr(df.index, "tz", None) is not None:
                    df.index = df.index.tz_localize(None)
                return df
        except Exception:
            time.sleep(1.2)
        time.sleep(0.4)
    return None

def fetch_stooq_daily(ticker: str) -> pd.DataFrame | None:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        if r.ok and "Date,Open,High,Low,Close,Volume" in r.text:
            df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"])
            df.rename(columns=str.lower, inplace=True)
            df.set_index("date", inplace=True)
            df.rename(columns={"close":"Close", "open":"Open","high":"High","low":"Low","volume":"Volume"}, inplace=True)
            return df
    except Exception:
        pass
    return None

@app.get("/api/stock")
def stock(ticker: str = "AAPL", period: str = "1y", interval: str = "1d"):
    try:
        df = fetch_yf(ticker, period, interval)
        if df is None or df.empty:
            df = fetch_stooq_daily(ticker)
        if df is None or df.empty:
            return {"ticker": ticker.upper(), "points": []}

        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index
        else:
            ts = pd.to_datetime(df.index, errors="coerce")

        records = []
        for i in range(len(df)):
            t = ts[i]
            t_iso = t.isoformat() if hasattr(t, "isoformat") else str(t)
            row = df.iloc[i]
            records.append({
                "time": t_iso,
                "open": float(row.get("Open", 0) or 0),
                "high": float(row.get("High", 0) or 0),
                "low":  float(row.get("Low", 0) or 0),
                "close":float(row.get("Close", 0) or 0),
                "volume": float(row.get("Volume", 0) or 0),
            })
        return {"ticker": ticker.upper(), "points": records}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
