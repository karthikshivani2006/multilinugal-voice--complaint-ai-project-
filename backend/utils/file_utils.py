from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from config import settings


def ensure_upload_dirs() -> None:
    base = Path(settings.upload_dir)
    (base / "evidence").mkdir(parents=True, exist_ok=True)
    (base / "id_proofs").mkdir(parents=True, exist_ok=True)


def save_upload_file(upload: UploadFile, sub_dir: str) -> str:
    """Save upload file and return absolute path."""
    ensure_upload_dirs()
    safe_name = upload.filename or "file"
    extension = Path(safe_name).suffix
    final_name = f"{uuid4().hex}{extension}"
    final_path = Path(settings.upload_dir) / sub_dir / final_name

    with final_path.open("wb") as f:
        f.write(upload.file.read())

    # Return absolute path for file operations
    return str(final_path.resolve())


def get_relative_file_path(absolute_path: str) -> str:
    """Convert absolute path to relative path for serving via /uploads mount."""
    # Extract the part after "uploads/" from the absolute path
    # E.g., "/path/to/uploads/evidence/abc.pdf" -> "uploads/evidence/abc.pdf"
    path_str = absolute_path.replace("\\", "/")
    
    # Find where "uploads/" starts
    if "uploads/" in path_str:
        idx = path_str.find("uploads/")
        return path_str[idx:]
    
    # Fallback: return as-is if "uploads/" not found
    return path_str
