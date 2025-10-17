# ===== Trading Backend (FastAPI) =====
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta

# =====================
# App-Setup
# =====================
app = FastAPI(title="Trading Backend")

# CORS freischalten (Frontend darf API ansprechen)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trading-frontend-coje.onrender.com",  # Render-Frontend
        "https://tool.market-vision-pro.com"           # deine eigene Domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# Test-Endpunkte
# =====================
@app.get("/")
def root():
    return {"status": "ok", "message": "Trading Backend läuft 🚀"}

# =====================
# Aktien-Daten abrufen
# =====================
@app.get("/api/stock")
def fetch_ohlc(ticker, period, interval):
    t = yf.Ticker(ticker)
    # 1) Primär über history()
    df = t.history(period=period, interval=interval, actions=False, auto_adjust=False)
    if df is None or df.empty:
        # 2) Period-Fallbacks (yfinance mag "1mo" manchmal nicht)
        period_map = {"1mo": "30d", "3mo": "90d", "6mo": "180d", "1y": "365d"}
        alt_period = period_map.get(period, period)
        df = t.history(period=alt_period, interval=interval, actions=False, auto_adjust=False)
    if df is None or df.empty:
        # 3) Notnagel: download()
        df = yf.download(
            ticker, period=period, interval=interval,
            progress=False, threads=False, auto_adjust=False
        )
    return df if df is not None else pd.DataFrame()

@app.get("/api/stock")
def get_stock(ticker: str = "AAPL", period: str = "1mo", interval: str = "1d"):
    try:
        ticker = ticker.upper().strip()
        data = fetch_ohlc(ticker, period, interval)
        if data.empty:
            return JSONResponse(content={"error": "Keine Daten gefunden"}, status_code=404)

        # Indikatoren berechnen (unverändert):
        data["RSI"] = calculate_rsi(data)
        data["MACD"], data["MACD_signal"] = calculate_macd(data)
        data["BB_MA"], data["BB_upper"], data["BB_lower"] = calculate_bollinger(data)
        data["Stoch_K"], data["Stoch_D"] = calculate_stochastic(data)

        data.reset_index(inplace=True)
        return data.to_dict(orient="records")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
