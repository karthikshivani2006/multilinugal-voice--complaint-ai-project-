from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models.admin import Admin
from models.complaint import Complaint
from schemas.admin import AdminLoginRequest, AdminLoginResponse
from schemas.complaint import ComplaintStatusUpdateRequest
from services.auth_service import auth_service, get_current_admin
from services.translation_service import translation_service
from utils.file_utils import get_relative_file_path


router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/login", response_model=AdminLoginResponse)
def admin_login(payload: AdminLoginRequest, db: Session = Depends(get_db)):
    token = auth_service.login_admin(db, email=payload.email, password=payload.password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    return AdminLoginResponse(access_token=token)


@router.get("/complaints")
def get_all_complaints(
    db: Session = Depends(get_db),
    _admin: Admin = Depends(get_current_admin),
):
    complaints = db.query(Complaint).order_by(Complaint.created_at.desc()).all()

    result = []
    for c in complaints:
        complaint_type_en = c.complaint_type
        description_en = c.description
        evidence_files = []
        try:
            if (c.language or "English") != "English":
                complaint_type_en = translation_service.to_english(c.complaint_type or "") or c.complaint_type
                description_en = translation_service.to_english(c.description or "") or c.description
        except Exception:
            complaint_type_en = c.complaint_type
            description_en = c.description

        for e in c.evidence_items:
            extracted = e.extracted_text or ""
            extracted_en = extracted
            try:
                if extracted and (c.language or "English") != "English":
                    extracted_en = translation_service.to_english(extracted) or extracted
            except Exception:
                extracted_en = extracted

            # Convert absolute file path to relative URL path
            relative_path = get_relative_file_path(e.file_path)
            file_url = "/" + relative_path if not relative_path.startswith("/") else relative_path
            evidence_files.append(
                {
                    "file_path": e.file_path,
                    "file_url": file_url,
                    "extracted_text": extracted_en,
                    "extracted_text_original": extracted,
                }
            )

        result.append(
            {
                "ticket_id": c.ticket_id,
                "name": c.user.name if c.user else "",
                "email": c.user.email if c.user else "",
                "phone": c.user.phone if c.user else "",
                "complaint_type": complaint_type_en,
                "description": description_en,
                "status": c.status,
                "language": c.language,
                "created_at": c.created_at,
                "amount_lost": c.amount_lost,
                "transaction_id": c.transaction_id,
                "platform": c.platform,
                "suspect_details": c.suspect_details,
                "suspect_vpa": c.suspect_vpa,
                "suspect_phone": c.suspect_phone,
                "suspect_bank_account": c.suspect_bank_account,
                "date_time": c.date_time,
                "evidence_files": evidence_files,
            }
        )

    return result


@router.put("/update-status")
def update_status(
    payload: ComplaintStatusUpdateRequest,
    db: Session = Depends(get_db),
    _admin: Admin = Depends(get_current_admin),
):
    complaint = db.query(Complaint).filter(Complaint.ticket_id == payload.ticket_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    complaint.status = payload.status
    db.add(complaint)
    db.commit()

    return {"message": "Status updated", "ticket_id": complaint.ticket_id, "status": complaint.status}
