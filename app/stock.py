from fastapi import APIRouter, HTTPException, Query
import yfinance as yf
import pandas as pd

router = APIRouter(prefix="/api", tags=["stock"])

@router.get("/stock")
def stock(symbol: str = Query(...), interval: str = Query("1d"), years: int = Query(1), adjusted: bool = Query(True)):
    end = pd.Timestamp.today().normalize()
    if interval == "1h" and years > 2:
        years = 2
    start = end - pd.DateOffset(years=int(years))
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=adjusted, progress=False)
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail="no_data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "count": len(df),
        "t": [int(ts.value//10**6) for ts in df.index],
        "o": [float(x) for x in df["Open"].tolist()],
        "h": [float(x) for x in df["High"].tolist()],
        "l": [float(x) for x in df["Low"].tolist()],
        "c": [float(x) for x in df["Close"].tolist()],
        "v": [float(x) for x in df.get("Volume", pd.Series(index=df.index).fillna(0)).tolist()],
    }
