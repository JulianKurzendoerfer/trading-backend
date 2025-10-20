# main.py
import os
import time
from datetime import datetime, timedelta
from typing import Literal, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import requests

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# -------------------------------------------------
# Konfiguration
# -------------------------------------------------
FRONTEND_ORIGINS = [
    "https://trading-frontend-coje.onrender.com",
    "https://tool.market-vision-pro.com",  # falls du die Domain nutzt
]

# Stabilere yfinance-Session (User-Agent gegen Blocks)
_requests_session = requests.Session()
_requests_session.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/127.0 Safari/537.36"
})

# yfinance-Defaults
YF_KWARGS = dict(
    auto_adjust=True,  # Dividenden/Splits korrigiert – stabilere Charts
    prepost=False,
    progress=False,
    threads=True,
    session=_requests_session,
)

# Unterstützte Kombis – wir probieren diese Reihenfolge nacheinander
RANGE_MATRIX: Tuple[Tuple[str, str], ...] = (
    ("30d", "1d"),
    ("1mo", "1d"),
    ("3mo", "1d"),
    ("6mo", "1d"),
    ("1y", "1d"),
)

# -------------------------------------------------
# FastAPI
# -------------------------------------------------
app = FastAPI(title="Trading Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS + ["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=JSONResponse)
def root():
    # leichte Health-Response, damit Render beim Aufwecken schnell was hat
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


# -------------------------------------------------
# Hilfen
# -------------------------------------------------
def _to_records(df: pd.DataFrame) -> dict:
    """
    Konvertiert OHLCV + Indikatoren in einfaches, JSON-serialisierbares Dict.
    Index -> ISO-Strings.
    """
    out = {}
    if df is None or df.empty:
        return out

    # Zeitachse
    out["index"] = [pd.Timestamp(x).isoformat() for x in df.index.to_pydatetime()]

    # OHLCV – nur vorhandene Spalten mitnehmen
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            # floats -> normal rounden (2 Nachkommastellen reichen)
            out[col] = [None if pd.isna(v) else float(round(v, 6)) for v in df[col].tolist()]

    # Indikatoren, wenn vorhanden
    for col in ["RSI", "MACD", "MACD_signal", "Bollinger_upper", "Bollinger_lower", "Stoch_K", "Stoch_D"]:
        if col in df.columns:
            out[col] = [None if pd.isna(v) else float(round(v, 6)) for v in df[col].tolist()]

    return out


def _calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ein paar robuste, rechenleichte Indikatoren.
    Keine externen Libs, nur pandas/numpy.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return df

    close = df["Close"].astype(float)

    # RSI(14)
    window = 14
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(window).mean()
    roll_down = pd.Series(down, index=close.index).rolling(window).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    df["RSI"] = rsi

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal

    # Bollinger(20,2)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["Bollinger_upper"] = ma20 + 2 * std20
    df["Bollinger_lower"] = ma20 - 2 * std20

    # Stochastic(14)
    low14 = df["Low"].rolling(14).min() if "Low" in df.columns else close.rolling(14).min()
    high14 = df["High"].rolling(14).max() if "High" in df.columns else close.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14)
    stoch_d = stoch_k.rolling(3).mean()
    df["Stoch_K"] = stoch_k
    df["Stoch_D"] = stoch_d

    return df


def _download_tolerant(ticker: str, period: str, interval: str, max_retries: int = 2) -> pd.DataFrame:
    """
    Ein Download-Versuch mit kleinen Wartezeiten und 1–2 automatischen Retries.
    """
    for attempt in range(max_retries + 1):
        try:
            df = yf.download(ticker, period=period, interval=interval, **YF_KWARGS)
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Falls yfinance Multiindex OHLCV liefert
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [' '.join(col).strip() for col in df.columns.values]
                    # Standardisiere Spaltennamen auf typische Bezeichner
                    rename_map = {
                        "Open": "Open", "High": "High", "Low": "Low", "Close": "Close",
                        "Adj Close": "Adj Close", "Volume": "Volume"
                    }
                    # falls die Namen in der Form 'Close' oder 'Close close' usw. vorliegen
                    for col in list(df.columns):
                        base = col.split()[-1].capitalize()
                        if base in rename_map:
                            df.rename(columns={col: rename_map[base]}, inplace=True)

                return df
        except Exception:
            pass
        time.sleep(1.0 + attempt * 1.0)
    return pd.DataFrame()


def fetch_data_resilient(ticker: str, requested_period: str, requested_interval: str) -> Tuple[pd.DataFrame, Tuple[str, str]]:
    """
    Versuche zuerst mit der gewünschten Kombi, dann mit Fallback-Kombis aus RANGE_MATRIX.
    """
    tried = set()
    combos = [(requested_period, requested_interval)] + list(RANGE_MATRIX)
    for period, interval in combos:
        key = f"{period}|{interval}"
        if key in tried:
            continue
        tried.add(key)

        df = _download_tolerant(ticker, period=period, interval=interval, max_retries=2)
        if not df.empty:
            # Index säubern
            df = df.sort_index()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df.dropna(axis=0, subset=[df.index.name])

            # Indikatoren
            df = _calc_indicators(df)
            return df, (period, interval)

    return pd.DataFrame(), ("", "")


# -------------------------------------------------
# API
# -------------------------------------------------
@app.get("/api/stock", response_class=JSONResponse)
def get_stock(
    ticker: str = Query(..., description="z.B. AAPL"),
    period: str = Query("1mo", description="z.B. 30d, 1mo, 3mo, 6mo, 1y"),
    interval: str = Query("1d", description="z.B. 1d, 1h, 1wk"),
):
    ticker = ticker.upper().strip()

    df, used = fetch_data_resilient(ticker, period, interval)
    if df.empty:
        return JSONResponse({"error": "Keine Daten gefunden", "ticker": ticker}, status_code=502)

    payload = {
        "ticker": ticker,
        "used_period": used[0],
        "used_interval": used[1],
        "data": _to_records(df),
    }
    return JSONResponse(payload)
