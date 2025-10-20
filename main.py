from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from pandas_datareader import data as pdr

# --------------------------
# FastAPI App + CORS
# --------------------------
app = FastAPI(title="Trading Backend")

# ➜ Trage hier deine Frontend-URL(s) ein; auf Render ist das z. B.:
# https://trading-frontend-coje.onrender.com
ALLOWED_ORIGINS = [
    "https://trading-frontend-coje.onrender.com",
    # Optional (eigene Domain später):
    # "https://tool.market-vision-pro.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS + ["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Kleiner In-Memory-Cache (5 Min)
# --------------------------
CACHE: Dict[Tuple[str, Optional[str], Optional[str]], Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 300  # 5 Minuten

def _cache_get(key):
    item = CACHE.get(key)
    if not item:
        return None
    if datetime.utcnow() > item["exp"]:
        CACHE.pop(key, None)
        return None
    return item["val"]

def _cache_set(key, val):
    CACHE[key] = {"val": val, "exp": datetime.utcnow() + timedelta(seconds=CACHE_TTL_SECONDS)}

# --------------------------
# Utils
# --------------------------
def make_session() -> requests.Session:
    """Requests-Session mit UA, damit Yahoo weniger zickt."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    })
    return s

def df_to_payload(df: pd.DataFrame) -> Dict[str, Any]:
    """Standardisiertes JSON für Frontend. Index als ISO-String."""
    out = {}
    # Index vereinheitlichen
    if not isinstance(df.index, pd.DatetimeIndex):
        # Stooq kommt oft mit Date-Index; konvertieren
        df.index = pd.to_datetime(df.index)

    out["index"] = [d.strftime("%Y-%m-%d %H:%M:%S") for d in df.index]

    # Normale OHLCV Felder, nur wenn vorhanden
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            # NaN → None, damit JSON sauber ist
            vals = df[col].replace({np.nan: None}).tolist()
            out[col] = vals

    # Ein paar leichte Indikatoren (optional, falls Frontend sie anzeigt)
    close = df["Close"] if "Close" in df.columns else None
    if close is not None:
        try:
            sma20 = close.rolling(20).mean().replace({np.nan: None}).tolist()
            sma50 = close.rolling(50).mean().replace({np.nan: None}).tolist()
            out["SMA20"] = sma20
            out["SMA50"] = sma50
        except Exception:
            pass

    return out

# --------------------------
# Datenquellen
# --------------------------
def fetch_stooq_daily(ticker: str) -> Optional[pd.DataFrame]:
    """Stooq liefert Daily-Daten. Wir versuchen Ticker & Ticker.US."""
    for sym in [ticker, f"{ticker}.US"]:
        try:
            df = pdr.DataReader(sym, "stooq")
            if df is not None and not df.empty:
                df = df.sort_index()  # Stooq kommt oft absteigend
                # Spaltennamen harmonisieren
                for alt, std in [("Open", "Open"), ("High", "High"), ("Low", "Low"),
                                 ("Close", "Close"), ("Volume", "Volume")]:
                    if alt not in df.columns and std in df.columns:
                        pass  # ist ok
                return df
        except Exception:
            continue
    return None

def fetch_yfinance(ticker: str, period: str, interval: str, session: requests.Session) -> Optional[pd.DataFrame]:
    """Ein Aufruf an Yahoo mit Session/UA; None bei leeren Frames."""
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            progress=False,
            session=session,
            auto_adjust=False,
            threads=False,
        )
        if df is not None and not df.empty:
            # YF liefert MultiIndex in manchen Intervallen; flachziehen
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [" ".join(col).strip() for col in df.columns.values]
            # Einheitliche Namen soweit möglich
            for col in ["Open","High","Low","Close","Adj Close","Volume"]:
                if col not in df.columns:
                    # Versuche Varianten zu mappen (z.B. 'Adj Close' fehlt manchmal)
                    pass
            return df
    except Exception:
        return None
    return None

def fetch_with_fallbacks(ticker: str, req_period: Optional[str], req_interval: Optional[str]) -> Dict[str, Any]:
    """
    1) Stooq (Daily) zuerst.
    2) Falls leer, Yahoo mit mehreren Kombis oder der gewünschten Kombi.
    """
    # 1) Stooq (stabil, daily)
    df = fetch_stooq_daily(ticker)
    if df is not None and not df.empty:
        payload = df_to_payload(df)
        payload.update({"used_source": "stooq", "used_period": "1y", "used_interval": "1d"})
        return payload

    # 2) Yahoo-Fallbacks
    session = make_session()

    # Wenn der Client period/interval mitgibt, probieren wir das zuerst…
    combos = []
    if req_period or req_interval:
        combos.append((req_period or "1y", req_interval or "1d"))

    # …danach bewährte Kombinationen
    combos += [
        ("1mo", "1d"),
        ("3mo", "1d"),
        ("6mo", "1d"),
        ("1y", "1d"),
        ("5d", "1h"),  # etwas intraday
        ("1d", "15m"),
    ]

    tried = []
    for period, interval in combos:
        period = period or "1y"
        interval = interval or "1d"
        tried.append((period, interval))
        df = fetch_yfinance(ticker, period, interval, session=session)
        if df is not None and not df.empty:
            payload = df_to_payload(df)
            payload.update({"used_source": "yfinance", "used_period": period, "used_interval": interval})
            return payload

    return {"error": "Keine Daten gefunden", "ticker": ticker, "tried": tried}

# --------------------------
# Routes
# --------------------------
@app.get("/", response_class=JSONResponse)
def root():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}

@app.get("/api/stock", response_class=JSONResponse)
def api_stock(
    request: Request,
    ticker: str = Query(...),
    period: Optional[str] = Query(None),
    interval: Optional[str] = Query(None),
):
    key = (ticker.upper(), period, interval)

    cached = _cache_get(key)
    if cached is not None:
        return cached

    data = fetch_with_fallbacks(ticker.upper(), period, interval)
    if "error" in data:
        # Kein 500, sondern 200 mit Fehlertext → Frontend kann sinnvoll reagieren
        return JSONResponse(data)

    _cache_set(key, data)
    return data
