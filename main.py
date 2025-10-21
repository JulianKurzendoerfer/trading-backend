from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import yfinance as yf
import requests
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tool.market-vision-pro.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/stock")
def stock(ticker: str, period: str = "1y", interval: str = "1d"):
    T = (ticker or "").upper().strip()
    if not T:
        return {"ticker": T, "points": []}

    df = None
    try:
        df = yf.download(T, period=period, interval=interval, progress=False, auto_adjust=False, threads=True)
        if df is not None and not df.empty:
            df = df.dropna()
    except Exception:
        df = None

    if df is None or df.empty:
        try:
            url = f"https://stooq.com/q/d/l/?s={T.lower()}.us&i=d"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if df is None or df.empty:
        return {"ticker": T, "points": []}

    df = df.reset_index()

    ts_col = next((c for c in ["Datetime","Date","datetime","date","Time","time"] if c in df.columns), None)
    if ts_col is None:
        return {"ticker": T, "points": []}

    rec = []
    for _, row in df.iterrows():
        t = row.get(ts_col)
        t_iso = t.isoformat() if hasattr(t, "isoformat") else str(t)
        rec.append({
            "time": t_iso,
            "open": float((row.get("Open", 0) or 0)),
            "high": float((row.get("High", 0) or 0)),
            "low": float((row.get("Low", 0) or 0)),
            "close": float((row.get("Close", 0) or 0)),
            "volume": float((row.get("Volume", 0) or 0)),
        })

    return {"ticker": T, "points": rec}
