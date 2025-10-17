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
def get_stock(ticker: str = "AAPL", period: str = "1y", interval: str = "1d"):
    """
    Lädt historische Kursdaten mit yfinance.
    Beispiel: /api/stock?ticker=TSLA&period=6mo&interval=1d
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return JSONResponse(content={"error": "Keine Daten gefunden"}, status_code=404)

        data.reset_index(inplace=True)
        return data.to_dict(orient="records")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
