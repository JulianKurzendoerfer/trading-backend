import os, time, jwt
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, constr
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from auth_db import SessionLocal, init_db
from auth_models import User

router = APIRouter(prefix="/api/auth", tags=["auth"])

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
JWT_SECRET = os.getenv("JWT_SECRET", "devsecret-change-me")
ALGO = "HS256"
ACCESS_MIN = int(os.getenv("ACCESS_TOKEN_MIN", "20"))
REFRESH_DAYS = int(os.getenv("REFRESH_TOKEN_DAYS", "7"))
COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN", ".market-vision-pro.com")
OPEN_SIGNUPS = os.getenv("OPEN_SIGNUPS", "false").lower() == "true"
ADMIN_INVITE_CODE = os.getenv("ADMIN_INVITE_CODE", "")

def db():
    d = SessionLocal()
    try:
        yield d
    finally:
        d.close()

class RegisterIn(BaseModel):
    email: EmailStr
    password: constr(min_length=8)
    invite_code: str | None = None

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

def _jwt(payload: dict, minutes: int = 20):
    payload = {**payload, "exp": datetime.utcnow() + timedelta(minutes=minutes), "iat": datetime.utcnow()}
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGO)

def _refresh_jwt(payload: dict, days: int = 7):
    payload = {**payload, "exp": datetime.utcnow() + timedelta(days=days), "iat": datetime.utcnow(), "typ": "refresh"}
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGO)

def set_refresh_cookie(resp: Response, token: str):
    resp.set_cookie(
        "refresh_token", token,
        max_age=REFRESH_DAYS*24*3600, httponly=True, secure=True, samesite="none",
        domain=COOKIE_DOMAIN, path="/"
    )

@router.post("/register")
def register(body: RegisterIn, session: Session = Depends(db)):
    if not OPEN_SIGNUPS and body.invite_code != ADMIN_INVITE_CODE:
        raise HTTPException(status_code=403, detail="signup closed")
    if session.query(User).filter(User.email == body.email.lower()).first():
        raise HTTPException(status_code=409, detail="exists")
    u = User(email=body.email.lower(), password_hash=pwd.hash(body.password))
    session.add(u); session.commit()
    return {"ok": True}

@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, response: Response, session: Session = Depends(db)):
    u = session.query(User).filter(User.email == body.email.lower()).first()
    if not u or not pwd.verify(body.password, u.password_hash):
        raise HTTPException(status_code=401, detail="invalid")
    access = _jwt({"sub": str(u.id), "email": u.email}, minutes=ACCESS_MIN)
    refresh = _refresh_jwt({"sub": str(u.id)}, days=REFRESH_DAYS)
    set_refresh_cookie(response, refresh)
    return {"access_token": access}

@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("refresh_token", domain=COOKIE_DOMAIN, path="/")
    return {"ok": True}

@router.get("/refresh", response_model=TokenOut)
def refresh(request: Request):
    tok = request.cookies.get("refresh_token")
    if not tok:
        raise HTTPException(status_code=401, detail="no cookie")
    try:
        payload = jwt.decode(tok, JWT_SECRET, algorithms=[ALGO])
    except Exception:
        raise HTTPException(status_code=401, detail="bad cookie")
    access = _jwt({"sub": payload["sub"]}, minutes=ACCESS_MIN)
    resp = Response()
    resp.media_type = "application/json"
    resp.body = ('{"access_token":"%s","token_type":"bearer"}' % access).encode()
    return resp

bearer = HTTPBearer(auto_error=False)

def require_user(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    if not creds or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="no token")
    try:
        jwt.decode(creds.credentials, JWT_SECRET, algorithms=[ALGO])
    except Exception:
        raise HTTPException(status_code=401, detail="bad token")
    return True

async def guard_middleware(request: Request, call_next):
    p = request.url.path
    if p.startswith("/api") and not (p.startswith("/api/auth") or p.startswith("/api/health")):
        auth = request.headers.get("authorization", "")
        ok = False
        if auth.lower().startswith("bearer "):
            try:
                jwt.decode(auth.split(" ",1)[1], JWT_SECRET, algorithms=[ALGO]); ok=True
            except Exception: ...
        if not ok:
            return Response('{"detail":"unauthorized"}', status_code=401, media_type="application/json")
    return await call_next(request)

init_db()
