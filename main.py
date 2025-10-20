from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import logging
import threading
import requests

# -------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("backend")

# -------- App + CORS -------
app = FastAPI(title="Trading Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trading-frontend-coje.onrender.com",
        "https://tool.market-vision-pro.com",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- HTTP Session mit UA (gegen leere Yahoo-Frames) --------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})

# -------- Mini-Cache (RAM) --------
_cache = {}              # key: (ticker, period, interval) -> (ts, payload)
CACHE_TTL = 60           # Sekunden

def _cache_get(key):
    item = _cache.get(key)
    if not item:
        return None
    ts, payload = item
    if time.time() - ts > CACHE_TTL:
        return None
    return payload

def _cache_set(key, payload):
    _cache[key] = (time.time(), payload)

# -------- robuste Datenbeschaffung --------
# Kombis in der Reihenfolge, die in der Praxis am häufigsten erfolgreich sind
CANDIDATES = [
    ("1mo", "1d"),
    ("30d", "1d"),
    ("3mo", "1d"),
    ("6mo", "1d"),
    ("1y", "1d"),
    ("60d", "1h"),
    ("5d", "15m"),
]

def _download_yf(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Erst Ticker.history (nutzt Session sicher), dann fallback auf yf.download.
    Beide Wege mit derselben Session+UA.
    """
    try:
        t = yf.Ticker(ticker, session=SESSION)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception as e:
        log.warning(f"Ticker.history failed: {e}")

    # Fallback: yf.download (neuere yfinance akzeptiert session=)
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            session=SESSION
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception as e:
        log.warning(f"yf.download failed: {e}")

    return pd.DataFrame()

def _ensure_df(df: pd.DataFrame) -> pd.DataFrame:
    # Spalten harmonisieren
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    cols = {c.lower(): c for c in df.columns}
    want = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    out = {}
    for name in want:
        key = name.lower()
        if key in cols:
            out[name] = df[cols[key]]
    dfo = pd.DataFrame(out)
    if "Datetime" in dfo.columns:
        dfo.set_index("Datetime", inplace=True)
    if not isinstance(dfo.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            dfo.index = pd.to_datetime(df["Date"])
        else:
            dfo.index = pd.to_datetime(df.index)
    dfo.index = dfo.index.tz_localize(None) if dfo.index.tz is not None else dfo.index
    return dfo

def _calc_indicators(dfo: pd.DataFrame) -> pd.DataFrame:
    if dfo.empty:
        return dfo
    # RSI (14)
    delta = dfo["Close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll = 14
    roll_up = up.rolling(roll, min_periods=roll).mean()
    roll_down = down.rolling(roll, min_periods=roll).mean()
    rs = roll_up / (roll_down + 1e-12)
    dfo["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = dfo["Close"].ewm(span=12, adjust=False).mean()
    ema26 = dfo["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    dfo["MACD"] = macd
    dfo["MACD_Signal"] = signal
    dfo["MACD_Hist"] = macd - signal

    # Bollinger (20, 2)
    ma20 = dfo["Close"].rolling(20, min_periods=20).mean()
    std20 = dfo["Close"].rolling(20, min_periods=20).std(ddof=0)
    dfo["BB_Middle"] = ma20
    dfo["BB_Upper"] = ma20 + 2 * std20
    dfo["BB_Lower"] = ma20 - 2 * std20

    # Stochastic (14)
    low14 = dfo["Low"].rolling(14, min_periods=14).min()
    high14 = dfo["High"].rolling(14, min_periods=14).max()
    stoch_k = 100 * (dfo["Close"] - low14) / (high14 - low14 + 1e-12)
    dfo["Stoch_K"] = stoch_k
    dfo["Stoch_D"] = stoch_k.rolling(3, min_periods=3).mean()
    return dfo

def fetch_any(ticker: str, period: str | None, interval: str | None):
    """
    Holt Daten robust:
    - benutzt gewünschte Kombi, wenn gesetzt,
    - sonst probiert mehrere Kandidaten,
    - mit Retries + Delay,
    - gibt außerdem used_period/used_interval zurück.
    """
    tried = []
    combos = [(period, interval)] if period and interval else CANDIDATES

    for per, inv in combos:
        if not per or not inv:
            continue
        tried.append((per, inv))

        # Cache?
        key = (ticker.upper(), per, inv)
        cached = _cache_get(key)
        if cached:
            return cached | {"used_period": per, "used_interval": inv, "from_cache": True}

        # 2–3 Retries gegen leere Frames
        for attempt in range(3):
            df = _download_yf(ticker, per, inv)
            dfo = _ensure_df(df)
            if not dfo.empty and {"Open","High","Low","Close"}.issubset(dfo.columns):
                dfo = _calc_indicators(dfo)
                payload = {
                    "index": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in dfo.index],
                    "Open": dfo["Open"].round(4).fillna(None).tolist(),
                    "High": dfo["High"].round(4).fillna(None).tolist(),
                    "Low": dfo["Low"].round(4).fillna(None).tolist(),
                    "Close": dfo["Close"].round(4).fillna(None).tolist(),
                    "Adj Close": dfo.get("Adj Close", dfo["Close"]).round(4).fillna(None).tolist(),
                    "Volume": dfo.get("Volume", pd.Series([None]*len(dfo))).fillna(None).astype(object).tolist(),
                    "RSI": dfo.get("RSI", pd.Series([None]*len(dfo))).round(2).fillna(None).tolist(),
                    "MACD": dfo.get("MACD", pd.Series([None]*len(dfo))).round(4).fillna(None).tolist(),
                    "MACD_Signal": dfo.get("MACD_Signal", pd.Series([None]*len(dfo))).round(4).fillna(None).tolist(),
                    "MACD_Hist": dfo.get("MACD_Hist", pd.Series([None]*len(dfo))).round(4).fillna(None).tolist(),
                    "BB_Upper": dfo.get("BB_Upper", pd.Series([None]*len(dfo))).round(4).fillna(None).tolist(),
                    "BB_Middle": dfo.get("BB_Middle", pd.Series([None]*len(dfo))).round(4).fillna(None).tolist(),
                    "BB_Lower": dfo.get("BB_Lower", pd.Series([None]*len(dfo))).round(4).fillna(None).tolist(),
                    "Stoch_K": dfo.get("Stoch_K", pd.Series([None]*len(dfo))).round(2).fillna(None).tolist(),
                    "Stoch_D": dfo.get("Stoch_D", pd.Series([None]*len(dfo))).round(2).fillna(None).tolist(),
                }
                # Cache setzen
                _cache_set(key, payload)
                log.info(f"OK {ticker} with {per}/{inv} (attempt {attempt+1}) rows={len(dfo)}")
                return payload | {"used_period": per, "used_interval": inv, "from_cache": False}

            time.sleep(0.6)  # kurzer Backoff

        log.warning(f"Empty frame for {ticker} with {per}/{inv} after retries")

    return {"error": "Keine Daten gefunden", "ticker": ticker, "tried": tried}

# -------- Routes --------
@app.get("/", response_class=JSONResponse)
def root():
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}

@app.get("/api/stock", response_class=JSONResponse)
def api_stock(
    ticker: str = Query(..., description="z.B. AAPL"),
    period: str | None = Query(None, description="z.B. 1mo, 30d, 3mo, 6mo, 1y"),
    interval: str | None = Query(None, description="z.B. 1d, 1h, 15m")
):
    payload = fetch_any(ticker, period, interval)
    # Bei Fehler → 404 (damit Frontend sauber reagieren kann)
    if "error" in payload:
        return JSONResponse(payload, status_code=404)
    return payload
