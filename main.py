from fastapi import FastAPI, HTTPException, Query
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
def stock(
    ticker: str = Query(..., min_length=1),
    period: str = "1d",
    interval: str = "1h",
    limit: int = 5,
):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="no data")
        out = (
            df.reset_index()
              .rename(columns={"Datetime": "time", "Date": "time"})
              .head(max(1, min(limit, len(df))))
        )
        records = []
        for r in out.to_dict(orient="records"):
            t = r.get("time")
            if hasattr(t, "isoformat"):
                r["time"] = t.isoformat()
            records.append({
                "time": r["time"],
                "open": float(r.get("Open", 0)),
                "high": float(r.get("High", 0)),
                "low": float(r.get("Low", 0)),
                "close": float(r.get("Close", 0)),
                "volume": float(r.get("Volume", 0)),
            })
        return {"ticker": ticker.upper(), "points": records}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
