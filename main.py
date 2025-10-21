from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests, io, time, logging
from datetime import timedelta

import yfinance as yf

app = FastAPI(title="Trading Backend")

# CORS weit offen (Frontend auf Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log = logging.getLogger("uvicorn.error")

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/119.0 Safari/537.36"}

def _period_to_days(p:str) -> int:
    p = (p or "").lower()
    return {
        "1mo": 31, "3mo": 93, "6mo": 186, "1y": 370, "2y": 740, "5y": 1850,
        "30d": 30, "90d": 90, "180d": 180,
    }.get(p, 370)

def _trim_to_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Schneidet auf gewünschte Periode – **immer in UTC** vergleichen,
    um tz-naive vs. tz-aware Fehler sicher zu vermeiden."""
    days = _period_to_days(period)
    since_utc = (pd.Timestamp.now(tz="UTC").normalize() - timedelta(days=days))

    col = None
    if "Date" in df.columns:
        col = "Date"
    elif "index" in df.columns:
        col = "index"

    if col:
        s = pd.to_datetime(df[col], utc=True, errors="coerce")
        mask = s >= since_utc
        df = df.loc[mask].reset_index(drop=True)

        # Wir behalten eine einheitliche Namensgebung bei
        if col != "index":
            df = df.rename(columns={col: "index"})
        else:
            # sicherstellen, dass Spaltenname wirklich "index" ist (falls Multi-Index o.ä.)
            df = df.rename(columns={"index": "index"})
    return df

def fetch_stooq_daily(ticker: str) -> pd.DataFrame:
    """Daily OHLC von stooq.com (sehr stabil). Für US-Ticker '.us' Suffix."""
    t = ticker.lower()
    if "." not in t:
        t = f"{t}.us"
    url = f"https://stooq.com/q/d/l/?s={t}&i=d"
    log.info(f"stooq GET {url}")
    r = requests.get(url, headers=UA, timeout=15)
    if r.status_code != 200 or not r.text or "No data" in r.text:
        raise RuntimeError(f"stooq empty ({r.status_code})")
    df = pd.read_csv(io.StringIO(r.text))
    # Erwartet: Date,Open,High,Low,Close,Volume
    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        raise RuntimeError("stooq malformed")
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df

def fetch_yf_any(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """yfinance mit Fallback-Kombis und Retries."""
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
                df = yf.Ticker(ticker).history(
                    period=per, interval=itv,
                    auto_adjust=False, repair=True, prepost=False
                )
                if not df.empty:
                    df = df.reset_index().rename(columns={"Date": "index"})
                    return df, (per, itv), tried
            except Exception as e:
                log.warning(f"yfinance error {per}/{itv}: {e}")
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"yfinance returned empty for all combos | tried={tried}")

@app.get("/")
def root():
    return {"status": "ok", "ts": pd.Timestamp.utcnow().isoformat()}

@app.get("/api/stock")
def get_stock(
    ticker: str = Query(...),
    period: str = Query("1y"),
    interval: str = Query("1d"),
):
    """OHLC + Meta. Daily über Stooq (falls möglich), sonst yfinance."""
    try:
        src = None
        used = None

        if (interval or "").lower() == "1d":
            try:
                df = fetch_stooq_daily(ticker)
                src = "stooq"
                used = (period, "1d")
            except Exception as se:
                log.warning(f"stooq fallback to yfinance: {se}")
                df, used, _ = fetch_yf_any(ticker, period, "1d")
                src = "yfinance"
        else:
            df, used, _ = fetch_yf_any(ticker, period, interval)
            src = "yfinance"

        # robust auf Periode trimmen (UTC)
        df = _trim_to_period(df, period)

        # Ausgabestruktur vereinheitlichen
        cols = df.columns
        def col(name):
            return name if name in cols else None

        out = {
            "index":  pd.to_datetime(df[col("index")] if col("index") else pd.Series([], dtype="datetime64[ns]"),
                                     utc=True, errors="coerce").astype(str).tolist(),
            "Open":   (df[col("Open")]   if col("Open")   else pd.Series([], dtype="float64")).astype(float).fillna("").tolist(),
            "High":   (df[col("High")]   if col("High")   else pd.Series([], dtype="float64")).astype(float).fillna("").tolist(),
            "Low":    (df[col("Low")]    if col("Low")    else pd.Series([], dtype="float64")).astype(float).fillna("").tolist(),
            "Close":  (df[col("Close")]  if col("Close")  else pd.Series([], dtype="float64")).astype(float).fillna("").tolist(),
            "Volume": (df[col("Volume")] if col("Volume") else pd.Series([], dtype="float64")).fillna("").tolist(),
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

# ==== AUTH attach (router + guard + CORS) ====
from fastapi.middleware.cors import CORSMiddleware
from auth import router as auth_router, guard_middleware
import os

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS","https://tool.market-vision-pro.com").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.middleware("http")(guard_middleware)
@app.get("/health")
async def health():
    return {"ok": True}
