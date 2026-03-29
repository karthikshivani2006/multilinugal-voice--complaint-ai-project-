import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import Session

import models  # noqa: F401
from config import settings
from database import Base, engine, SessionLocal
from models.admin import Admin
from routes.admin import router as admin_router
from routes.ai import router as ai_router
from routes.chat import router as chat_router
from routes.complaint import router as complaint_router
from utils.file_utils import ensure_upload_dirs
from utils.security import hash_password


app = FastAPI(title="CyberGuard AI Backend", version="1.0.0")
Base.metadata.create_all(bind=engine)
# seed_admin() # Optional if needed now

# Ensure upload folders exist before serving static files.
try:
    ensure_upload_dirs()
    if os.path.exists(settings.upload_dir):
        app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")
except Exception as e:
    print(f"Static mount failed: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:8081",
        "http://localhost:8082",
        "http://localhost:5173",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
        "http://127.0.0.1:8082",
        "http://127.0.0.1:5173",
    ],
    # Allow local dev origins on arbitrary ports (Vite, preview, static servers).
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    ensure_sqlite_columns()
    seed_admin()

def health():
    return {"status": "ok"}


def seed_admin() -> None:
    db: Session = SessionLocal()
    try:
        admin = db.query(Admin).filter(Admin.email == settings.admin_email).first()
        if not admin:
            db.add(Admin(email=settings.admin_email, password=hash_password(settings.admin_password)))
            db.commit()
    finally:
        db.close()


def ensure_sqlite_columns() -> None:
    if not str(settings.database_url).startswith("sqlite"):
        return

    required_columns = {
        "suspect_vpa": "TEXT",
        "suspect_phone": "TEXT",
        "suspect_bank_account": "TEXT",
    }

    try:
        with engine.begin() as conn:
            rows = conn.execute(text("PRAGMA table_info(complaints)"))
            existing = {row[1] for row in rows}
            for col, col_type in required_columns.items():
                if col not in existing:
                    conn.exec_driver_sql(f"ALTER TABLE complaints ADD COLUMN {col} {col_type}")
    except Exception as e:
        print(f"SQLite schema patch skipped: {e}")


@app.get("/")
def root():
    return {
        "message": "CyberGuard AI backend is running",
        "supported_languages": [
            "English",
            "Hindi",
            "Konkani",
            "Kannada",
            "Dogri",
            "Bodo",
            "Urdu",
            "Tamil",
            "Kashmiri",
            "Assamese",
            "Bengali",
            "Marathi",
            "Sindhi",
            "Maithili",
            "Punjabi",
            "Malayalam",
            "Manipuri",
            "Telugu",
            "Sanskrit",
            "Nepali",
            "Santali",
            "Gujarati",
            "Odia",
        ],
    }


app.include_router(chat_router)
app.include_router(complaint_router)
app.include_router(admin_router)
app.include_router(ai_router)
