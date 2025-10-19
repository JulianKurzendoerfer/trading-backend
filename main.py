from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = FastAPI(title="Trading Backend")

# ---- CORS erlauben (Frontend + Domain) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trading-frontend-coje.onrender.com",
        "https://tool.market-vision-pro.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Hilfsfunktionen / Indikatoren ----
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def bollinger(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma, sma + num_std * std, sma - num_std * std

def stochastic(df, k=14, d=3):
    low_min = df['Low'].rolling(window=k).min()
    high_max = df['High'].rolling(window=k).max()
    k_percent = 100 * (df['Close'] - low_min) / (high_max - low_min)
    d_percent = k_percent.rolling(window=d).mean()
    return k_percent, d_percent

# ---- API-Route ----
@app.get("/api/stock")
async def get_stock(ticker: str, period: str = "1mo", interval: str = "1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        if data.empty:
            return JSONResponse(content={"error": "Keine Daten gefunden"})
        df = data.copy()
        df['RSI'] = rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = macd(df['Close'])
        df['BB_Mid'], df['BB_Upper'], df['BB_Lower'] = bollinger(df['Close'])
        df['Stoch_K'], df['Stoch_D'] = stochastic(df)
        return JSONResponse(content=df.tail(60).to_dict())
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

