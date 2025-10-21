from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app import schemas
from app.models import User
from app.security import hash_pw, verify_pw, create_access_token, decode_token

router = APIRouter(prefix="/api/auth", tags=["auth"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/register", response_model=schemas.TokenOut)
def register(payload: schemas.RegisterIn, db: Session = Depends(get_db)):
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="password_too_short")
    exists = db.query(User).filter(User.email == payload.email).first()
    if exists:
        raise HTTPException(status_code=400, detail="email_exists")
    u = User(email=payload.email, password_hash=hash_pw(payload.password))
    db.add(u)
    db.commit()
    db.refresh(u)
    token = create_access_token({"sub": str(u.id), "email": u.email})
    return {"access_token": token, "token_type": "bearer"}

@router.post("/login", response_model=schemas.TokenOut)
def login(payload: schemas.LoginIn, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == payload.email).first()
    if not u or not verify_pw(payload.password, u.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_credentials")
    token = create_access_token({"sub": str(u.id), "email": u.email})
    return {"access_token": token, "token_type": "bearer"}

@router.get("/me", response_model=schemas.MeOut)
def me(request: Request, db: Session = Depends(get_db)):
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing_token")
    token = auth.split(" ", 1)[1]
    data = decode_token(token)
    if not data or "sub" not in data:
        raise HTTPException(status_code=401, detail="invalid_token")
    u = db.get(User, int(data["sub"]))
    if not u:
        raise HTTPException(status_code=401, detail="user_not_found")
    return {"id": u.id, "email": u.email}
