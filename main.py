from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.signal import argrelextrema
from plotly.subplots import make_subplots
import plotly.graph_objects as go

app = FastAPI(title="Trading Backend")

# ===== CORS Einstellungen =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trading-frontend-coje.onrender.com",  # dein Render-Frontend
        "https://tool.market-vision-pro.com"           # deine eigene Domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Beispielroute =====
@app.get("/")
def home():
    return {"message": "Hello from Trading Backend!"}

# ===== Beispiel: Kursdaten abrufen =====
@app.get("/stock/{ticker}")
def get_stock_data(ticker: str, period: str = "1y", interval: str = "1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            return JSONResponse(content={"error": "No data found"}, status_code=404)
        data.reset_index(inplace=True)
        return data.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
