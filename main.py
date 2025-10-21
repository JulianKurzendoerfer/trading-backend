from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tool.market-vision-pro.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/stock")
def stock(ticker: str, period: str = "1y", interval: str = "1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return {"ticker": ticker.upper(), "points": []}
        df = df.reset_index()
        ts_col = next((c for c in ["Date","Datetime","date","Time","time"] if c in df.columns), None)
        if ts_col is None:
            return {"ticker": ticker.upper(), "points": []}
        rec = []
        for _, r in df.iterrows():
            t = r[ts_col]
            t = t.isoformat() if hasattr(t, "isoformat") else str(t)
            rec.append({
                "time": t,
                "open": float(r.get("Open", 0) or 0),
                "high": float(r.get("High", 0) or 0),
                "low": float(r.get("Low", 0) or 0),
                "close": float(r.get("Close", 0) or 0),
                "volume": float(r.get("Volume", 0) or 0),
            })
        return {"ticker": ticker.upper(), "points": rec}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
