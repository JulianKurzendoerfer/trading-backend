from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd, requests, time, io
try:
    import yfinance as yf
except Exception:
    yf = None

APP_ORIGIN = "https://tool.market-vision-pro.com"
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=[APP_ORIGIN], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_cache, _TTL = {}, 120
def _now(): return int(time.time())
def _get(k): v=_cache.get(k); return None if not v or _now()-v["ts"]>_TTL else v["data"]
def _set(k,d): _cache[k]={"ts":_now(),"data":d}

def _to_points(df):
    if df is None or df.empty: return []
    tcol = next((c for c in ["Date","Datetime","date","Time","time"] if c in df.columns), None)
    if not tcol: return []
    if tcol!="Datetime" and "Adj Close" in df.columns: df=df.rename(columns={"Adj Close":"Close"})
    df=df.reset_index(drop=True)
    rec=[]
    for _,r in df.iterrows():
        t=r.get(tcol)
        ts=t.isoformat() if hasattr(t,"isoformat") else str(t)
        try: ts=pd.to_datetime(ts).isoformat()
        except Exception: continue
        rec.append({"time":ts,"open":float(r.get("Open",0) or 0),"high":float(r.get("High",0) or 0),"low":float(r.get("Low",0) or 0),"close":float(r.get("Close",r.get("Close*",0)) or 0),"volume":float(r.get("Volume",0) or 0)})
    return rec

def _stooq_df(ticker):
    try:
        r=requests.get(f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d",
                       headers={"User-Agent":"Mozilla/5.0","Accept":"text/csv","Cache-Control":"no-cache"},timeout=12)
        if r.status_code!=200 or "Date,Open,High,Low,Close,Volume" not in r.text: return None
        df=pd.read_csv(io.StringIO(r.text))
        if df is None or df.empty or "Date" not in df.columns: return None
        df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
        return df
    except Exception:
        return None

def _yf_df(ticker, period, interval):
    if yf is None: return None
    try:
        data=yf.Ticker(ticker).history(period=period,interval=interval,auto_adjust=False,actions=False)
        if data is None or data.empty: return None
        data=data.reset_index()
        if "Date" in data.columns: data["Date"]=pd.to_datetime(data["Date"],errors="coerce")
        return data
    except Exception:
        return None

@app.get("/")
def root(): return {"ok":True,"endpoints":["/api/health","/api/stock?ticker=AAPL&period=1y&interval=1d"]}

@app.get("/api/health")
def health(): return {"ok":True,"ts":datetime.utcnow().isoformat()}

@app.head("/api/health")
def health_head(): return Response(status_code=200)

@app.get("/api/stock")
def stock(ticker:str, period:str="1y", interval:str="1d"):
    key=f"{ticker}|{period}|{interval}"
    c=_get(key)
    if c is not None: return c
    df=_stooq_df(ticker) or _yf_df(ticker,period,interval)
    if df is None or df.empty:
        payload={"ticker":ticker.upper(),"points":[]}; _set(key,payload); return payload
    payload={"ticker":ticker.upper(),"points":_to_points(df)}; _set(key,payload); return payload
