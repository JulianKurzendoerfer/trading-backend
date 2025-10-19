from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI(title="Trading Backend")

# ==== CORS: erlaube dein Frontend & deine Domain ====
ALLOWED_ORIGINS = [
    "https://trading-frontend-coje.onrender.com",
    "https://tool.market-vision-pro.com",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS + ["*"],   # notfalls alles erlauben
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ Hilfsfunktionen: Indikatoren --------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period, min_periods=period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period, min_periods=period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, window=20, mult=2.0):
    ma = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std()
    upper = ma + mult * sd
    lower = ma - mult * sd
    return ma, upper, lower

def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k=14, d=3):
    lowest_low = low.rolling(k, min_periods=k).min()
    highest_high = high.rolling(k, min_periods=k).max()
    k_line = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d_line = k_line.rolling(d, min_periods=d).mean()
    return k_line, d_line

# ------------ Robustes Laden mit Fallbacks --------------
def fetch_prices(ticker: str, period: str, interval: str) -> pd.DataFrame:
    # 1) Standard: history()
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        if not df.empty:
            return df
    except Exception:
        pass

    # 2) Fallback: download()
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            prepost=True,
            threads=False,
            progress=False,
        )
        if not df.empty:
            return df
    except Exception:
        pass

    # 3) Alternative Perioden probieren (30d<->1mo, 1y, 6mo)
    alt_periods = []
    if period == "1mo":
        alt_periods = ["30d", "3mo", "6mo", "1y"]
    elif period == "30d":
        alt_periods = ["1mo", "3mo", "6mo", "1y"]
    else:
        alt_periods = ["1mo", "30d", "3mo", "6mo", "1y"]

    for p in alt_periods:
        for fn in ("history", "download"):
            try:
                if fn == "history":
                    t = yf.Ticker(ticker)
                    df = t.history(period=p, interval=interval, auto_adjust=False)
                else:
                    df = yf.download(
                        tickers=ticker,
                        period=p,
                        interval=interval,
                        auto_adjust=False,
                        prepost=True,
                        threads=False,
                        progress=False,
                    )
                if not df.empty:
                    return df
            except Exception:
                continue

    # 4) Letzter Rettungsanker
    try:
        df = yf.download(tickers=ticker, period="1y", interval="1d", progress=False, prepost=True, threads=False)
        if not df.empty:
            return df
    except Exception:
        pass

    # Nichts gefunden
    return pd.DataFrame()

# ------------ API --------------
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/api/stock")
def get_stock(
    ticker: str = Query(..., description="z.B. AAPL"),
    period: str = Query("1mo", description="z.B. 1mo, 30d, 3mo, 6mo, 1y, 5y"),
    interval: str = Query("1d", description="z.B. 1d, 1h, 30m, 15m, 5m, 1m"),
):
    df = fetch_prices(ticker, period, interval)

    if df is None or df.empty:
        return JSONResponse({"error": "Keine Daten gefunden", "ticker": ticker, "period": period, "interval": interval}, status_code=200)

    # yfinance gibt je nach Weg Multiindex/Spalten – normieren:
    if isinstance(df.columns, pd.MultiIndex):
        # Typisch: ('Open','AAPL') etc.
        df.columns = [' '.join([c for c in col if c]).strip() for col in df.columns.values]
    # Versuche Standardnamen sicherzustellen:
    rename_map = {
        'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Adj Close': 'Adj Close', 'Volume': 'Volume',
        'open':'Open','high':'High','low':'Low','close':'Close','adjclose':'Adj Close','volume':'Volume'
    }
    df = df.rename(columns=rename_map)

    needed = ["Open","High","Low","Close","Volume"]
    for col in needed:
        if col not in df.columns:
            # wenn Spaltennamen z.B. "AAPL Close" heißen:
            matches = [c for c in df.columns if col.lower() in c.lower()]
            if matches:
                df[col] = df[matches[0]]
            else:
                df[col] = np.nan

    df = df.dropna(subset=["Close"])

    # Indikatoren
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["RSI"] = rsi(close, 14)
    macd_line, signal_line, macd_hist = macd(close, 12, 26, 9)
    df["MACD"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = macd_hist
    ma, upper, lower = bollinger(close, 20, 2.0)
    df["BB_mid"] = ma
    df["BB_upper"] = upper
    df["BB_lower"] = lower
    k_line, d_line = stoch(high, low, close, 14, 3)
    df["Stoch_K"] = k_line
    df["Stoch_D"] = d_line

    # JSON-freundlich machen
    df = df.fillna(method="ffill").dropna()
    out = {
        "index": [i.isoformat() if hasattr(i, "isoformat") else str(i) for i in df.index.to_list()],
        "Open": df["Open"].round(6).tolist(),
        "High": df["High"].round(6).tolist(),
        "Low": df["Low"].round(6).tolist(),
        "Close": df["Close"].round(6).tolist(),
        "Volume": [int(x) if pd.notna(x) else None for x in df["Volume"].tolist()],
        "RSI": df["RSI"].round(4).tolist(),
        "MACD": df["MACD"].round(6).tolist(),
        "MACD_signal": df["MACD_signal"].round(6).tolist(),
        "MACD_hist": df["MACD_hist"].round(6).tolist(),
        "BB_mid": df["BB_mid"].round(6).tolist(),
        "BB_upper": df["BB_upper"].round(6).tolist(),
        "BB_lower": df["BB_lower"].round(6).tolist(),
        "Stoch_K": df["Stoch_K"].round(4).tolist(),
        "Stoch_D": df["Stoch_D"].round(4).tolist(),
        "meta": {"ticker": ticker, "period": period, "interval": interval, "rows": int(df.shape[0])},
    }
    return JSONResponse(out, status_code=200)
