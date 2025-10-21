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
        def fetch_df():
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                df2 = yf.Ticker(ticker).history(period=period, interval=interval)
                return df2 if df2 is not None else df
            return df

        df = fetch_df()
        if df is None or df.empty:
            return {"ticker": ticker.upper(), "points": []}

        df = df.reset_index()
        ts_col = next((c for c in ["Date","Datetime","date","Time","time"] if c in df.columns), None)
        if ts_col is None:
            return {"ticker": ticker.upper(), "points": []}

        records = []
        for r in df.to_dict(orient="records"):
            t = r.get(ts_col)
            ts = t.isoformat() if hasattr(t, "isoformat") else str(t)
            records.append({
                "time": ts,
                "open": float(r.get("Open", 0) or 0),
                "high": float(r.get("High", 0) or 0),
                "low":  float(r.get("Low", 0) or 0),
                "close":float(r.get("Close", 0) or 0),
                "volume": float(r.get("Volume", 0) or 0),
            })
        return {"ticker": ticker.upper(), "points": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
