from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np
import requests

app = FastAPI(title="Trading Backend")

# ==== CORS ====
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trading-frontend-coje.onrender.com",
        "https://tool.market-vision-pro.com",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Hilfsindikatoren ----
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series, window=20, mult=2.0):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return ma, ma + mult * sd, ma - mult * sd

def stoch(high, low, close, k=14, d=3):
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    k_line = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_line = k_line.rolling(d).mean()
    return k_line, d_line

# ---- Datenabruf mit Fallback ----
def fetch_data(ticker, period="1mo", interval="1d"):
    # Erst yfinance
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if not df.empty:
            return df
    except Exception:
        pass

    # Fallback: TwelveData-API (kostenlos, aber limitiert)
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval={interval}&outputsize=100&apikey=demo"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "values" in data:
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime")
            df = df.rename(columns={
                "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
            })
            df.set_index("datetime", inplace=True)
            df = df.astype(float)
            return df
    except Exception:
        pass

    return pd.DataFrame()

@app.get("/")
def root():
    return {"status": "ok"}

from fastapi.responses import JSONResponse
from datetime import datetime
import yfinance as yf

@app.get("/api/stock", response_class=JSONResponse)
def get_stock(ticker: str, period: str = "6mo", interval: str = "1d"):
    """
    Liefert OHLC-Daten (und funktioniert auch ohne period/interval-Parameter,
    dank Defaults period=6mo, interval=1d).
    """
    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )

        if data is None or data.empty:
            return JSONResponse(
                {"error": "Keine Daten gefunden", "ticker": ticker},
                status_code=404
            )

        out = {
            "index": [
                d.strftime("%Y-%m-%d %H:%M:%S") if isinstance(d, datetime) else str(d)
                for d in data.index
            ],
            "Open":  [float(x) for x in data["Open"].astype(float).round(2)],
            "High":  [float(x) for x in data["High"].astype(float).round(2)],
            "Low":   [float(x) for x in data["Low"].astype(float).round(2)],
            "Close": [float(x) for x in data["Close"].astype(float).round(2)],
            "Adj Close": [float(x) for x in data.get("Adj Close", data["Close"]).astype(float).round(2)],
            "Volume": [int(x) for x in data["Volume"].fillna(0).astype(int)],
        }
        return out
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
