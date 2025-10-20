from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests, io, time, logging
from datetime import datetime, timedelta

import yfinance as yf

app = FastAPI(title="Trading Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log = logging.getLogger("uvicorn.error")

# --- kleine Utils ------------------------------------------------------------

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/119.0 Safari/537.36"}

def _period_to_days(p:str) -> int:
    p = (p or "").lower()
    return {
        "1mo": 31, "3mo": 93, "6mo": 186, "1y": 370, "2y": 740, "5y": 1850,
        "30d": 30, "90d": 90, "180d": 180,
    }.get(p, 370)

def _trim_to_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    days = _period_to_days(period)
    since = pd.Timestamp.utcnow().normalize() - timedelta(days=days)
    if "Date" in df.columns:
        m = pd.to_datetime(df["Date"]) >= since
        return df.loc[m].reset_index(drop=True)
    if "index" in df.columns:
        m = pd.to_datetime(df["index"]) >= since
        return df.loc[m].reset_index(drop=True)
    return df

# --- Stooq (Daily) -----------------------------------------------------------

def fetch_stooq_daily(ticker: str) -> pd.DataFrame:
    """
    Holt Daily-Daten von stooq.com (CSV). Für US-Ticker benötigt Stooq ein '.us'-Suffix.
    """
    t = ticker.lower()
    if "." not in t:  # wenn kein Suffix, US annehmen
        t = f"{t}.us"
    url = f"https://stooq.com/q/d/l/?s={t}&i=d"
    log.info(f"stooq GET {url}")
    r = requests.get(url, headers=UA, timeout=15)
    if r.status_code != 200 or not r.text or "No data" in r.text:
        raise RuntimeError(f"stooq empty ({r.status_code})")
    df = pd.read_csv(io.StringIO(r.text))
    # Erwartete Spalten: Date,Open,High,Low,Close,Volume
    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        raise RuntimeError("stooq malformed")
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df

# --- yfinance (flexibel) -----------------------------------------------------

def fetch_yf_any(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Versucht mehrere period/interval-Kombis, bis Daten kommen.
    """
    combos = [
        (period or "1y", interval or "1d"),
        ("6mo", "1d"),
        ("3mo", "1d"),
        ("1mo", "1d"),
        ("30d", "1d"),
        ("5d", "1h"),
        ("60d", "1h"),
        ("5d", "30m"),
        ("5d", "15m"),
    ]
    tried = []
    for per, itv in combos:
        tried.append((per, itv))
        for attempt in range(3):
            try:
                log.info(f"yfinance history {ticker} period={per} interval={itv} attempt={attempt+1}")
                df = yf.Ticker(ticker).history(period=per, interval=itv, auto_adjust=False, repair=True, prepost=False)
                if not df.empty:
                    df = df.reset_index().rename(columns={"Date": "index"})
                    return df, (per, itv), tried
            except Exception as e:
                log.warning(f"yfinance error {per}/{itv}: {e}")
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"yfinance returned empty for all combos | tried={tried}")

# --- API ---------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}

@app.get("/api/stock")
def get_stock(
    ticker: str = Query(...),
    period: str = Query("1y"),
    interval: str = Query("1d"),
):
    """
    Liefert OHLC + Meta.
    - Daily (1d): bevorzugt Stooq, danach yfinance.
    - Intraday: yfinance.
    """
    try:
        src = None
        used = None

        if (interval or "").lower() == "1d":
            # 1) Stooq zuerst (sehr stabil)
            try:
                df = fetch_stooq_daily(ticker)
                src = "stooq"
                used = (period, "1d")
            except Exception as se:
                log.warning(f"stooq fallback to yfinance: {se}")
                # 2) yfinance als Fallback
                df, used, tried = fetch_yf_any(ticker, period, "1d")
                src = "yfinance"
        else:
            # Intraday & rest: yfinance
            df, used, tried = fetch_yf_any(ticker, period, interval)
            src = "yfinance"

        # auf gewünschte Periode begrenzen (bei Stooq sinnvoll)
        df = _trim_to_period(df, period)

        # vereinheitlichte Ausgabe
        if "index" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "index"})
        out = {
            "index": df.get("index", pd.Series(dtype=str)).astype(str).tolist(),
            "Open":  df.get("Open",  pd.Series()).astype(float).fillna("").tolist(),
            "High":  df.get("High",  pd.Series()).astype(float).fillna("").tolist(),
            "Low":   df.get("Low",   pd.Series()).astype(float).fillna("").tolist(),
            "Close": df.get("Close", pd.Series()).astype(float).fillna("").tolist(),
            "Volume": df.get("Volume", pd.Series()).fillna("").tolist(),
            "meta": {
                "ticker": ticker,
                "source": src,
                "used_period": used[0] if used else period,
                "used_interval": used[1] if used else interval,
            },
        }
        return out

    except Exception as e:
        log.error(f"API error for {ticker}: {e}")
        return JSONResponse(
            status_code=502,
            content={
                "ok": False,
                "error": str(e),
                "ticker": ticker,
                "used_period": period,
                "used_interval": interval,
            },
        )
