# main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import yfinance as yf


# ---------------------------
# FastAPI + CORS
# ---------------------------
app = FastAPI(title="Trading Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trading-frontend-coje.onrender.com",
        "https://tool.market-vision-pro.com",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Helpers
# ---------------------------
def _is_df_ok(df: Optional[pd.DataFrame]) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """Index -> ISO Datetime-String, tz-frei (stabile JSON-Keys)."""
    out = df.copy()
    try:
        out.index = pd.to_datetime(out.index)
        out.index = out.index.tz_localize(None)
        out.index = out.index.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    return out


def _as_ohlc_json(df: pd.DataFrame) -> Dict[str, Any]:
    cols = {
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj Close" if "Adj Close" in df.columns else "Close",
        "Volume": "Volume",
    }
    out: Dict[str, Any] = {}
    for key, col in cols.items():
        if col in df.columns:
            s = df[col]
            if key != "Volume":
                s = s.round(4)
            else:
                s = s.fillna(0).astype(int)
            out[key] = s.dropna().to_dict()
    return out


def _compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """RSI(14), MACD(12,26,9), Bollinger(20,2), Stoch(14,3)."""
    out: Dict[str, Any] = {}
    d = df.copy()

    if not _is_df_ok(d):
        return out

    close = d["Close"]

    # RSI 14
    try:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        roll = 14
        avg_gain = gain.rolling(roll, min_periods=roll).mean()
        avg_loss = loss.rolling(roll, min_periods=roll).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.round(2)
        out["RSI"] = rsi.dropna().to_dict()
    except Exception:
        pass

    # MACD 12/26 + Signal 9
    try:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        out["MACD"] = {
            "macd": macd.round(4).dropna().to_dict(),
            "signal": signal.round(4).dropna().to_dict(),
            "hist": hist.round(4).dropna().to_dict(),
        }
    except Exception:
        pass

    # Bollinger 20, 2
    try:
        m = close.rolling(20, min_periods=20).mean()
        s = close.rolling(20, min_periods=20).std(ddof=0)
        upper = (m + 2 * s).round(4)
        lower = (m - 2 * s).round(4)
        mid = m.round(4)
        out["Bollinger"] = {
            "upper": upper.dropna().to_dict(),
            "middle": mid.dropna().to_dict(),
            "lower": lower.dropna().to_dict(),
        }
    except Exception:
        pass

    # Stochastic 14,3
    try:
        low14 = d["Low"].rolling(14, min_periods=14).min()
        high14 = d["High"].rolling(14, min_periods=14).max()
        stoch_k = ((close - low14) / (high14 - low14) * 100).rolling(3, min_periods=3).mean()
        stoch_d = stoch_k.rolling(3, min_periods=3).mean()
        out["Stoch"] = {
            "%K": stoch_k.round(2).dropna().to_dict(),
            "%D": stoch_d.round(2).dropna().to_dict(),
        }
    except Exception:
        pass

    return out


def _period_fallbacks(period: str) -> list[str]:
    """
    Alternative Perioden, falls Yahoo für den gewünschten Zeitraum leer liefert.
    """
    mapping = {
        "5d": ["10d", "14d", "1mo"],
        "1mo": ["30d", "45d", "2mo"],
        "3mo": ["90d", "120d", "6mo"],
        "6mo": ["180d", "365d", "1y"],
        "1y": ["365d", "18mo", "2y"],
        "2y": ["730d", "3y"],
    }
    return mapping.get(period, ["30d", "90d", "180d"])


def fetch_with_fallbacks(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Robust: download -> history -> alternative Perioden.
    """
    # 1) download
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if _is_df_ok(df):
        return df

    # 2) history
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        if _is_df_ok(df):
            return df
    except Exception:
        pass

    # 3) alternative Perioden probieren
    for alt in _period_fallbacks(period):
        try:
            df = yf.download(
                tickers=ticker,
                period=alt,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if _is_df_ok(df):
                return df
        except Exception:
            continue

    return pd.DataFrame()  # leer => kein Erfolg


# ---------------------------
# Endpoints
# ---------------------------
@app.get("/", response_class=JSONResponse)
def root():
    return {"status": "ok", "ts": datetime.utcnow().isoformat() + "Z"}


@app.get("/healthz", response_class=JSONResponse)
def healthz():
    return {"status": "ok"}


@app.get("/api/stock", response_class=JSONResponse)
def api_stock(
    ticker: str = Query(..., description="Symbol, z. B. AAPL"),
    period: str = Query("1mo", description="z. B. 5d, 1mo, 3mo, 6mo, 1y"),
    interval: str = Query("1d", description="z. B. 1d, 1h, 30m, 15m"),
    indicators: bool = Query(True, description="RSI/MACD/Bollinger/Stoch mitliefern"),
) -> Any:
    try:
        df = fetch_with_fallbacks(ticker, period, interval)
        if not _is_df_ok(df):
            return JSONResponse({"error": "Keine Daten gefunden", "ticker": ticker})

        df = _clean_index(df)

        payload: Dict[str, Any] = {
            "meta": {
                "ticker": ticker.upper(),
                "period": period,
                "interval": interval,
            },
            "ohlc": _as_ohlc_json(df),
        }

        if indicators:
            payload["indicators"] = _compute_indicators(df)

        return payload

    except Exception as e:
        return JSONResponse({"error": str(e), "ticker": ticker})
