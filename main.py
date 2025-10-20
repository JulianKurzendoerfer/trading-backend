# main.py — robust: Stooq daily zuerst, dann yfinance-Fallback; Cache, CORS, 2 Worker empfohlen

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
from typing import Dict, Any, List, Tuple
from io import StringIO

# ---------- HTTP Session mit Retries ----------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "Chrome/124.0 Safari/537.36"
    })
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

SESSION = make_session()

# ---------- App + CORS ----------
app = FastAPI(title="Trading Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trading-frontend-coje.onrender.com",
        "http://localhost:5173",
        "https://tool.market-vision-pro.com",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- kleiner Memory-Cache (10 min) ----------
CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 600

def cache_get(key: str):
    item = CACHE.get(key)
    if not item:
        return None
    if (datetime.utcnow() - item["ts"]).total_seconds() > CACHE_TTL:
        CACHE.pop(key, None)
        return None
    return item["data"]

def cache_set(key: str, data: Any):
    CACHE[key] = {"ts": datetime.utcnow(), "data": data}

# ---------- Indikatoren ----------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = roll_up / roll_down
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def bollinger(series: pd.Series, window=20, num_std=2) -> Tuple[pd.Series, pd.Series]:
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return ma + num_std * std, ma - num_std * std

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window=14, d_window=3):
    lowest_low = low.rolling(k_window).min()
    highest_high = high.rolling(k_window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_window).mean()
    return k, d

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RSI"] = rsi(out["Close"]).round(2)
    macd_line, sig = macd(out["Close"])
    out["MACD"] = macd_line.round(4)
    out["MACD_signal"] = sig.round(4)
    up, lo = bollinger(out["Close"])
    out["BB_upper"] = up.round(4)
    out["BB_lower"] = lo.round(4)
    k, d = stochastic(out["High"], out["Low"], out["Close"])
    out["Stoch_K"] = k.round(2)
    out["Stoch_D"] = d.round(2)
    return out

# ---------- Datenquellen ----------
def dl_stooq_csv(ticker: str) -> pd.DataFrame:
    """Daily OHLC von Stooq (stabil, kein Key)."""
    for code in [f"{ticker}.US", ticker]:
        url = f"https://stooq.com/q/d/l/?s={code.lower()}&i=d"
        try:
            r = SESSION.get(url, timeout=15)
            if not r.ok or "Date,Open,High,Low,Close,Volume" not in r.text:
                continue
            df = pd.read_csv(StringIO(r.text))
            if df.empty:
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            continue
    return pd.DataFrame()

def dl_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """yfinance; bei 429/Fehler leere DF zurück (wir loggen nur)."""
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            session=SESSION,
            threads=False,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.rename(columns=str.title)
            if "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
            if "Date" not in df.columns:
                df = df.reset_index()
            if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            return df
    except Exception:
        pass
    return pd.DataFrame()

def candidate_matrix(req_period: str | None, req_interval: str | None) -> List[Tuple[str, str]]:
    combos: List[Tuple[str, str]] = []
    if req_period and req_interval:
        combos.append((req_period, req_interval))
    combos += [
        ("1mo", "1d"),
        ("30d", "1d"),
        ("3mo", "1d"),
        ("6mo", "1d"),
        ("1y", "1d"),
        ("60d", "1h"),
        ("5d", "15m"),
    ]
    seen, uniq = set(), []
    for p, i in combos:
        k = f"{p}|{i}"
        if k not in seen:
            seen.add(k)
            uniq.append((p, i))
    return uniq

def df_to_payload(df: pd.DataFrame, meta: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "index": df["Date"].astype(str).tolist(),
        "Open": df["Open"].round(4).tolist(),
        "High": df["High"].round(4).tolist(),
        "Low": df["Low"].round(4).tolist(),
        "Close": df["Close"].round(4).tolist(),
        "Volume": df["Volume"].astype(float).tolist(),
        "RSI": df["RSI"].fillna(None).tolist(),
        "MACD": df["MACD"].fillna(None).tolist(),
        "MACD_signal": df["MACD_signal"].fillna(None).tolist(),
        "BB_upper": df["BB_upper"].fillna(None).tolist(),
        "BB_lower": df["BB_lower"].fillna(None).tolist(),
        "Stoch_K": df["Stoch_K"].fillna(None).tolist(),
        "Stoch_D": df["Stoch_D"].fillna(None).tolist(),
    }
    payload.update(meta)
    return payload

# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}

@app.get("/api/stock")
def get_stock(
    ticker: str = Query(..., description="z. B. AAPL"),
    period: str | None = Query(None),
    interval: str | None = Query(None),
):
    t = ticker.strip().upper()
    cache_key = f"{t}|{period}|{interval}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    tried: List[Tuple[str, str]] = []

    # 1) ZUERST: Stooq (Daily) – vermeidet Yahoo-429
    stooq_df = dl_stooq_csv(t)
    if not stooq_df.empty:
        df = add_indicators(stooq_df)
        meta = {"source": "stooq", "used_period": "max", "used_interval": "1d"}
        payload = df_to_payload(df, meta)
        cache_set(cache_key, payload)
        return payload

    # 2) Dann yfinance – mehrere period/interval-Kombis
    for p, iv in candidate_matrix(period, interval):
        df_yf = dl_yfinance(t, p, iv)
        if not df_yf.empty:
            df = add_indicators(df_yf)
            meta = {"source": "yfinance", "used_period": p, "used_interval": iv}
            payload = df_to_payload(df, meta)
            cache_set(cache_key, payload)
            return payload
        tried.append((p, iv))

    # 3) Nichts gefunden
    return JSONResponse({"error": "Keine Daten gefunden", "ticker": t, "tried": tried}, status_code=502)
