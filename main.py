from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from stock import router as stock_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tool.market-vision-pro.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
@app.get("/health")
def health():
    return {"ok": True}

app.include_router(stock_router, prefix="/api")
