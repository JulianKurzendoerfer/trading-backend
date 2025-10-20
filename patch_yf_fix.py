import yfinance as yf, socket
print("Testing yfinance access...")
try:
    data = yf.download("AAPL", period="1mo", interval="1d", progress=False)
    if data.empty:
        print("❌ Yahoo returned empty.")
    else:
        print("✅ yfinance working locally, rows:", len(data))
except Exception as e:
    print("⚠️ Exception:", e)
