from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from datetime import timedelta

app = FastAPI(title="Trading Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ========= Helpers & Indicators =========

def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()

def macd_tv(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    macd_sig  = ema(macd_line, signal)
    macd_hist = macd_line - macd_sig
    return macd_line, macd_sig, macd_hist

def rsi_wilder(close: pd.Series, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(0, 100)

def stochastic_full(high, low, close, k_period=14, k_smooth=3, d_period=3):
    hh = high.rolling(k_period, min_periods=k_period).max()
    ll = low.rolling(k_period, min_periods=k_period).min()
    denom = (hh - ll).replace(0, np.nan)
    raw_k = 100 * (close - ll) / denom
    k_slow = raw_k.rolling(k_smooth, min_periods=k_smooth).mean()
    d_slow = k_slow.rolling(d_period, min_periods=d_period).mean()
    return k_slow.clip(0, 100), d_slow.clip(0, 100)

def stoch_rsi(close, period=14, k=3, d=3):
    r = rsi_wilder(close, period)
    r_min = r.rolling(period, min_periods=period).min()
    r_max = r.rolling(period, min_periods=period).max()
    denom = (r_max - r_min).replace(0, np.nan)
    st_rsi = 100 * (r - r_min) / denom
    k_line = st_rsi.rolling(k, min_periods=k).mean()
    d_line = k_line.rolling(d, min_periods=d).mean()
    return k_line.clip(0, 100), d_line.clip(0, 100)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA9"]  = ema(out["Close"], 9)
    out["EMA21"] = ema(out["Close"], 21)
    out["EMA50"] = ema(out["Close"], 50)

    sma20 = out["Close"].rolling(20, min_periods=20).mean()
    std20 = out["Close"].rolling(20, min_periods=20).std()
    out["BB_basis"] = sma20
    out["BB_upper"] = sma20 + 2*std20
    out["BB_lower"] = sma20 - 2*std20

    out["RSI"] = rsi_wilder(out["Close"], 14)
    m_line, m_sig, m_hist = macd_tv(out["Close"], 12, 26, 9)
    out["MACD"], out["MACD_sig"], out["MACD_hist"] = m_line, m_sig, m_hist

    k_s, d_s = stochastic_full(out["High"], out["Low"], out["Close"], 14, 3, 3)
    out["%K"], out["%D"] = k_s, d_s

    sk, sd = stoch_rsi(out["Close"], 14, 3, 3)
    out["ST_RSI_K"], out["ST_RSI_D"] = sk, sd
    return out

def fetch_data(ticker: str, interval: str, period: str, adj: bool = True) -> pd.DataFrame:
    """
    Holt frische Daten per yfinance. Für 1h schneiden wir die evtl. letzte unvollständige Kerze ab.
    """
    df = yf.download(
        ticker, period=period, interval=interval, auto_adjust=adj, progress=False
    )
    if df is None or len(df) == 0:
        raise ValueError(f"Keine Daten empfangen für {ticker} ({period}/{interval}).")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)

    # letzte unvollständige 1h-Kerze entfernen
    if interval == "1h" and len(df) > 1:
        last = df.index[-1]
        prev = df.index[-2]
        if (last - prev) < timedelta(hours=1):
            df = df.iloc[:-1]
    return df

def fmt_title(ticker: str, period: str, interval: str):
    return f"{ticker.upper()} – {period} / {interval}"

def find_trend_levels(df: pd.DataFrame, window=10, tol=0.01, use_relative=True):
    p = pd.to_numeric(df["Close"], errors="coerce").values
    lows_idx  = argrelextrema(p, np.less_equal, order=window)[0]
    highs_idx = argrelextrema(p, np.greater_equal, order=window)[0]
    if (lows_idx.size + highs_idx.size) == 0:
        return np.array([], int), np.array([], int), np.array([], float), np.array([], float), np.array([], float)
    lvls_raw = np.sort(np.concatenate([p[lows_idx], p[highs_idx]]))
    cl, ct = [], []
    for l in lvls_raw:
        if not cl:
            cl.append(float(l)); ct.append(1); continue
        gap_ok = (abs(l - cl[-1]) <= (tol * cl[-1])) if use_relative else (abs(l - cl[-1]) <= tol)
        if gap_ok:
            cl[-1] = (cl[-1]*ct[-1] + float(l)) / (ct[-1] + 1); ct[-1] += 1
        else:
            cl.append(float(l)); ct.append(1)
    counts = np.array(ct, dtype=float)
    strength = (counts - counts.min()) / (counts.max() - counts.min() + 1e-9)
    return lows_idx, highs_idx, np.array(cl, dtype=float), counts, strength

def add_trend_panel(fig, df, row):
    x = df.index
    close_line = pd.to_numeric(df["Close"], errors="coerce")
    fig.add_trace(go.Scatter(x=x, y=close_line, name="Close", mode="lines",
                             line=dict(width=1.6), connectgaps=True), row=row, col=1)
    lows, highs, lvls, ct, strength = find_trend_levels(df, window=10, tol=0.01, use_relative=True)
    # Levels als horizontale Linien
    if lvls.size:
        min_w, max_w = 1.5, 4.5
        for lvl, cnt, s in zip(lvls, ct, strength):
            width = float(min_w + (max_w - min_w) * float(s))
            fig.add_hline(y=float(lvl), line_width=width, line_color="rgba(120,120,220,0.85)",
                          opacity=0.65, layer="below", row=row, col=1)
    # aktuelle Preislinie
    if len(close_line):
        fig.add_hline(y=float(close_line.iloc[-1]), line_dash="dash", line_color="orange",
                      opacity=0.9, layer="below", row=row, col=1)

def build_figure(df: pd.DataFrame, interval: str, theme: str = "light") -> go.Figure:
    # Reihenfolge: Price, MACD, RSI(größer), Stoch, Stoch RSI, Trend
    row_heights = [0.95, 0.55, 0.75, 0.55, 0.55, 0.60]
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights)

    # 1) Price + BB + kompakte EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             mode="lines", line=dict(width=1.8)), row=1, col=1)
    for col, nm, dash in [
        ("BB_upper", "BB Upper", "dot"),
        ("BB_basis", "BB Basis", "dash"),
        ("BB_lower", "BB Lower", "dot"),
    ]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=nm,
                                 mode="lines", line=dict(width=1.2, dash=dash)), row=1, col=1)
    for col, nm in [("EMA9","EMA9"),("EMA21","EMA21"),("EMA50","EMA50")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=nm, mode="lines", line=dict(width=1.2)),
                      row=1, col=1)

    # 2) MACD (kräftigere Farben)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD hist", opacity=0.35), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines",
                             line=dict(width=1.8, color="#d35400")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_sig"], name="Signal", mode="lines",
                             line=dict(width=1.6, color="#2980b9")), row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#999", row=2, col=1)

    # 3) RSI (größer)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI (14)", mode="lines"), row=3, col=1)
    for y in (70, 50, 30):
        fig.add_hline(y=y, line_width=1, line_dash="dash", line_color="#aaa", row=3, col=1)
    fig.update_yaxes(range=[0,100], row=3, col=1)

    # 4) Stochastic
    fig.add_trace(go.Scatter(x=df.index, y=df["%K"], name="Stoch %K", mode="lines"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["%D"], name="Stoch %D", mode="lines"), row=4, col=1)
    for y in (80, 50, 20):
        fig.add_hline(y=y, line_width=1, line_dash="dash", line_color="#aaa", row=4, col=1)
    fig.update_yaxes(range=[0,100], row=4, col=1)

    # 5) Stoch RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["ST_RSI_K"], name="Stoch RSI %K", mode="lines"), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ST_RSI_D"], name="Stoch RSI %D", mode="lines"), row=5, col=1)
    for y in (80, 50, 20):
        fig.add_hline(y=y, line_width=1, line_dash="dash", line_color="#aaa", row=5, col=1)
    fig.update_yaxes(range=[0,100], row=5, col=1)

    # 6) Trend-Panel
    add_trend_panel(fig, df, row=6)

    # Layout
    fig.update_layout(
        template="plotly_white" if theme == "light" else "plotly_dark",
        height=1050,
        margin=dict(l=40, r=10, t=40, b=20),
        hovermode="x unified",
        showlegend=True,
    )
    # X-Achsen
    fmt = "%Y-%m-%d %H:%M" if interval=="1h" else "%Y-%m-%d"
    for r in range(1, 7):
        fig.update_xaxes(matches="x", row=r, col=1, hoverformat=fmt, showgrid=True)
    return fig

# ========= API =========

@app.get("/api/indicators")
def indicators(ticker: str = "AAPL", period: str = "1y", interval: str = "1d", adj: bool = True):
    df = fetch_data(ticker, interval, period, adj)
    df = compute_indicators(df)
    payload = {
        "series": [{"date": str(d), "close": float(v)} for d, v in zip(df.index, df["Close"])],
        "indicators": {
            "ema9": df["EMA9"].dropna().tolist(),
            "ema21": df["EMA21"].dropna().tolist(),
            "ema50": df["EMA50"].dropna().tolist(),
        }
    }
    resp = JSONResponse(payload)
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp

# ========= UI (Form + Plot auf EINER Seite) =========

FORM_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Trading Dashboard</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    h1 {{ margin: 0 0 16px 0; }}
    form {{ display: grid; grid-template-columns: repeat(6, max-content); gap: 10px 16px; align-items: center; }}
    label {{ font-weight: 600; }}
    input[type=text], select {{ padding: 6px 8px; border-radius: 8px; border: 1px solid #ccc; }}
    button {{ padding: 8px 14px; border-radius: 10px; border: 0; background:#111; color:#fff; cursor:pointer; }}
    .hint {{ color:#666; font-size: 12px; margin-top: 6px; }}
    .wrap {{ max-width: 1300px; }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Trading Dashboard</h1>
  <form method="get" action="/">
    <label for="ticker">Ticker</label>
    <input type="text" id="ticker" name="ticker" value="{ticker}" placeholder="AAPL" />
    <label for="period">Zeitraum</label>
    <select id="period" name="period">
      {period_options}
    </select>
    <label for="interval">Intervall</label>
    <select id="interval" name="interval">
      {interval_options}
    </select>
    <label for="theme">Theme</label>
    <select id="theme" name="theme">
      <option value="light" {sel_light}>light</option>
      <option value="dark"  {sel_dark}>dark</option>
    </select>
    <button type="submit">Aktualisieren</button>
  </form>

  {plot_html}
</div>
</body>
</html>
"""

PERIOD_CHOICES = ["6mo","1y","2y","5y","10y","max"]
INTERVAL_CHOICES = ["1h","1d","1wk"]

def _options_html(current: str, choices: list[str]):
    out = []
    for c in choices:
        sel = "selected" if c == current else ""
        out.append(f'<option value="{c}" {sel}>{c}</option>')
    return "\n".join(out)

@app.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    ticker: str = "AAPL",
    period: str = "1y",
    interval: str = "1d",
    theme: str = "light"
):
    # Plot auf derselben Seite rendern
    try:
        df = fetch_data(ticker.strip(), interval, period, True)
        df = compute_indicators(df)
        fig = build_figure(df, interval=interval, theme=theme)
        fig.update_layout(title=fmt_title(ticker, period, interval))
        plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception as e:
        plot_html = f'<p style="color:#c00"><b>Fehler:</b> {str(e)}</p>'

    html = FORM_TEMPLATE.format(
        ticker=ticker,
        period_options=_options_html(period, PERIOD_CHOICES),
        interval_options=_options_html(interval, INTERVAL_CHOICES),
        sel_light="selected" if theme=="light" else "",
        sel_dark="selected" if theme=="dark" else "",
        plot_html=plot_html
    )
    resp = HTMLResponse(html)
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp
