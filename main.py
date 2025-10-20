from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import io
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------------
# Render/CORS: Frontend erlauben
# -----------------------------
FRONTEND_ORIGINS = [
    "https://trading-frontend-coje.onrender.com",
    "http://localhost:5173",
]
app = FastAPI(title="Trading Backend (robust)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# HTTP Session mit Retries + UA
# -----------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,                # 3 Versuche
        backoff_factor=0.6,     # 0.6s, 1.2s, 2.4s …
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    })
    return s

SESSION = make_session()

# -----------------------------
# Mini In-Memory Cache (10 min)
# -----------------------------
CACHE_TTL = 600  # Sekunden
CACHE: Dict[str, Dict[str, Any]] = {}  # key -> {ts,data}

def cache_key(ticker: str, period: str|None, interval: str|None) -> str:
    return f"{ticker.upper()}|{period or '-'}|{interval or '-'}"

def cache_get(key: str):
    hit = CACHE.get(key)
    if not hit: return None
    if (datetime.utcnow() - hit["ts"]).total_seconds() > CACHE_TTL:
        CACHE.pop(key, None); return None
    return hit["data"]

def cache_put(key: str, data: Dict[str, Any]):
    CACHE[key] = {"ts": datetime.utcnow(), "data": data}

# -----------------------------
# Utils
# -----------------------------
def period_to_days(p: str) -> int:
    # grobe Umrechnung für Filter
    table = {"1mo":31, "3mo":93, "6mo":186, "1y":365, "2y":730, "5y":1825}
    return table.get(p, 365)

def to_payload(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return {
        "index": [d.strftime("%Y-%m-%d %H:%M:%S") for d in df.index],
        "Open":  [float(x) for x in df["Open"].astype(float).tolist()],
        "High":  [float(x) for x in df["High"].astype(float).tolist()],
        "Low":   [float(x) for x in df["Low"].astype(float).tolist()],
        "Close": [float(x) for x in df["Close"].astype(float).tolist()],
    }

# -----------------------------
# Datenquellen
# -----------------------------
def fetch_from_stooq_daily(ticker: str) -> pd.DataFrame:
    """
    Stooq liefert stabile Daily-Daten als CSV.
    """
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
    r = SESSION.get(url, timeout=15)
    if r.status_code != 200 or "Date,Open,High,Low,Close,Volume" not in r.text:
        raise RuntimeError("stooq empty or unexpected format")
    df = pd.read_csv(io.StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df.rename(columns=str.strip)[["Open","High","Low","Close"]].dropna()
    return df

def fetch_from_yfinance_with_fallbacks(ticker: str,
                                       prefer_period: str|None,
                                       prefer_interval: str|None) -> Tuple[pd.DataFrame, str, str, List[Tuple[str,str]]]:
    """
    Versucht mehrere (period, interval)-Kombis; gibt verwendete Kombi + Liste der Versuche zurück.
    """
    tried: List[Tuple[str,str]] = []
    combos: List[Tuple[str,str]] = []

    if prefer_period and prefer_interval:
        combos.append((prefer_period, prefer_interval))

    # robuste Daily-Kombis (intraday ist auf Render/Yahoo oft flaky)
    for p in ["1mo","3mo","6mo","1y","2y"]:
        combos.append((p, "1d"))

    seen = set()
    combos = [c for c in combos if not (c in seen or seen.add(c))]

    last_err = None
    for p,i in combos:
        tried.append((p,i))
        try:
            df = yf.download(
                ticker, period=p, interval=i,
                progress=False, auto_adjust=False, prepost=False, threads=False
            )
            # yfinance gibt bei manchen Fehlern ein DataFrame mit Spalten aber ohne Zeilen
            if isinstance(df, pd.DataFrame) and not df.empty:
                cols = [c for c in ["Open","High","Low","Close"] if c in df.columns]
                if len(cols) == 4:
                    out = df[cols].dropna()
                    if not out.empty:
                        return out, p, i, tried
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("yfinance returned empty for all combos")

# -----------------------------
# Orchestrierung: Quelle wählen
# -----------------------------
def obtain_data(ticker: str, period: str|None, interval: str|None) -> Dict[str, Any]:
    """
    1) Für Daily zuerst Stooq (stabil, schnell).
    2) Falls (interval != 1d) oder Stooq leer: auf yfinance mit Fallbacks.
    3) Ergebnis inkl. Metadaten (source, used_period, used_interval).
    """
    meta: Dict[str, Any] = {
        "source": None, "used_period": period, "used_interval": interval, "ts": datetime.utcnow().isoformat()
    }

    # Versuch 1: Stooq (nur Daily)
    if interval in (None, "", "1d"):
        try:
            df = fetch_from_stooq_daily(ticker)
            if period:
                days = period_to_days(period)
                start = datetime.utcnow() - timedelta(days=days)
                df = df[df.index >= pd.Timestamp(start.date())]
            if not df.empty:
                meta["source"] = "stooq"
                meta["used_interval"] = "1d"
                if period is None:
                    meta["used_period"] = "1y"  # grobe Angabe
                return {"payload": to_payload(df), "meta": meta}
        except Exception:
            pass  # weiter zu yfinance

    # Versuch 2: yfinance (mit Fallbacks)
    df, used_p, used_i, tried = fetch_from_yfinance_with_fallbacks(ticker, period, interval)
    meta.update({"source": "yfinance", "used_period": used_p, "used_interval": used_i, "tried": tried})
    return {"payload": to_payload(df), "meta": meta}

# -----------------------------
# FastAPI Endpoints
# -----------------------------
@app.get("/", response_class=JSONResponse)
def root():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}

@app.get("/api/stock", response_class=JSONResponse)
def api_stock(
    ticker: str = Query(..., description="z.B. AAPL"),
    period: str | None = Query(None, description="z.B. 1mo, 3mo, 6mo, 1y …"),
    interval: str | None = Query(None, description="z.B. 1d (empfohlen)"),
):
    key = cache_key(ticker, period, interval)
    cached = cache_get(key)
    if cached:
        return cached

    try:
        result = obtain_data(ticker.strip(), period, interval)
        data = {"ok": True, **result}
        cache_put(key, data)
        return data
    except Exception as e:
        # saubere Fehlermeldung zurückgeben
        return JSONResponse(
            {"ok": False, "error": str(e), "ticker": ticker, "used_period": period, "used_interval": interval},
            status_code=502
        )
