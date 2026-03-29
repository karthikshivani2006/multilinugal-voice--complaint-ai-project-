import mimetypes
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from models.complaint import Complaint
from models.evidence import Evidence
from schemas.complaint import ComplaintCreateRequest, ComplaintCreateResponse, ComplaintTrackResponse, ComplaintCreateWithPartialIDRequest
from services.complaint_service import complaint_service
from services.email_service import email_service
from services.gemini_service import gemini_service
from services.translation_service import translation_service
from services.validation_service import validation_service
from services.enhanced_validation import validation_service as enhanced_validation
from services.llm_selector import get_active_llm_name, get_llm_client
from utils.file_utils import save_upload_file


router = APIRouter(prefix="/complaint", tags=["Complaint"])
logger = logging.getLogger(__name__)


@router.post("/create", response_model=ComplaintCreateResponse)
def create_complaint(payload: ComplaintCreateRequest, db: Session = Depends(get_db)):
    # Use enhanced validation with detailed error reporting
    is_valid, validation_errors = enhanced_validation.validate_complaint_data(payload.model_dump())
    if not is_valid:
        logger.warning("Complaint validation failed: %s", validation_errors)
        raise HTTPException(status_code=422, detail=validation_errors)

    # Log which LLM is active for this request
    active_llm = get_active_llm_name()
    logger.info(f"Processing complaint with LLM: {active_llm}")

    user = complaint_service.upsert_user(db, payload)
    complaint = complaint_service.create_complaint(db, user_id=user.id, payload=payload)

    success_message = "Complaint created successfully"
    localized_message = success_message
    localized_summary = complaint_service.build_user_summary(payload, complaint.ticket_id)
    admin_summary_en = complaint_service.build_admin_english_summary(payload, complaint.ticket_id)

    # Non-critical integrations should never block complaint creation.
    try:
        localized_summary = translation_service.translate(localized_summary, payload.language) or localized_summary
        admin_summary_en = translation_service.to_english(admin_summary_en) or admin_summary_en
        localized_message = translation_service.translate(success_message, payload.language) or success_message
    except Exception as exc:
        logger.warning("Translation failed for ticket %s: %s", complaint.ticket_id, exc)

    try:
        if payload.email and payload.email.strip() and payload.email.lower() not in {"n/a", "na", "none", "skip"}:
            email_service.send_user_ticket_email(payload.email, complaint.ticket_id, localized_summary)
        email_service.send_admin_complaint_email(settings.admin_email, complaint.ticket_id, admin_summary_en)
    except Exception as exc:
        logger.warning("Email dispatch failed for ticket %s: %s", complaint.ticket_id, exc)

    return ComplaintCreateResponse(
        ticket_id=complaint.ticket_id,
        status=complaint.status,
        created_at=complaint.created_at,
        message=localized_message,
    )


@router.post("/validate-id")
    def validate_id_proof(
        full_name: str = Form(""),
        phone_number: str = Form(""),
        email: str = Form(""),
        address: str = Form(""),
        file: UploadFile = File(...),
    ):
        """Validate ID proof and show extracted vs provided details with option to proceed with missing fields."""
        logger.info(f"🔍 ID proof validation started for {file.filename}")

        file_path = save_upload_file(file, sub_dir="id_proofs")
        mime_type = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"

        analysis = {}
        try:
            analysis = gemini_service.analyze_id_proof(file_path=file_path, mime_type=mime_type)
            logger.info(f"✅ ID proof analyzed: extraction_status={analysis.get('extraction_status')}")
        except Exception as exc:
            logger.error(f"❌ ID proof analysis error: {exc}", exc_info=True)
            analysis = {
                "extraction_status": "ERROR",
                "missing_fields": ["name", "phone", "email"],
                "extracted_text": f"[ID Proof: {file.filename}]"
            }

        # Get extracted values
        extracted_name = (analysis.get("name") or "").strip()
        extracted_phone = (analysis.get("phone") or "").strip()
        extracted_email = (analysis.get("email") or "").strip()
        extracted_address = (analysis.get("address") or "").strip()
        missing_fields = analysis.get("missing_fields", [])
        extraction_status = analysis.get("extraction_status", "UNKNOWN")

        # Normalization functions for comparison
        def norm_text(v: str) -> str:
            return " ".join((v or "").strip().lower().split())

        def norm_phone(v: str) -> str:
            digits = "".join(ch for ch in (v or "") if ch.isdigit())
            if len(digits) == 12 and digits.startswith("91"):
                digits = digits[2:]
            return digits[-10:] if len(digits) >= 10 else digits

        # Detect mismatches
        mismatches: list[str] = []
        if full_name.strip() and extracted_name and norm_text(full_name) != norm_text(extracted_name):
            mismatches.append("full_name")
        if phone_number.strip() and extracted_phone and norm_phone(phone_number) != norm_phone(extracted_phone):
            mismatches.append("phone_number")
        if email.strip() and extracted_email and norm_text(email) != norm_text(extracted_email):
            mismatches.append("email")
        if address.strip() and extracted_address and norm_text(address) not in norm_text(extracted_address) and norm_text(extracted_address) not in norm_text(address):
            mismatches.append("address")

        # Determine what user-provided data we have
        user_provided = {
            "name": full_name.strip() if full_name.strip() else None,
            "phone": phone_number.strip() if phone_number.strip() else None,
            "email": email.strip() if email.strip() else None,
            "address": address.strip() if address.strip() else None,
        }

        # Identify unclear/missing fields that need user input
        unclear_fields = []
        for field in missing_fields:
            if not user_provided.get(field):
                unclear_fields.append(field)

        logger.info(f"📊 ID proof validation summary: extraction_status={extraction_status}, "
                   f"unclear_fields={unclear_fields}, mismatches={mismatches}")

        return {
            "file_path": file_path,
            "analysis": analysis,
            "extracted": {
                "name": extracted_name,
                "phone": extracted_phone,
                "email": extracted_email,
                "address": extracted_address,
                "document_type": (analysis.get("document_type") or "").strip(),
                "id_number": (analysis.get("id_number") or "").strip(),
            },
            "user_provided": user_provided,
            "extraction_status": extraction_status,
            "missing_fields": missing_fields,
            "unclear_fields": unclear_fields,
            "mismatch_fields": mismatches,
            "message": (
                f"ID proof processed with {len(missing_fields)} unclear fields. "
                f"You can provide manual details for missing fields and proceed."
            ) if missing_fields else "ID proof details verified successfully.",
            "proceed_allowed": True,  # Allow proceeding regardless of extraction quality
            "proceed_recommended": len(mismatches) == 0 and len(missing_fields) == 0,
        }


@router.post("/create-with-partial-id", response_model=ComplaintCreateResponse)
def create_complaint_with_partial_id(payload: ComplaintCreateWithPartialIDRequest, db: Session = Depends(get_db)):
    """Create complaint when ID proof extraction is partial/unclear. Merges extracted + manual data."""
    logger.info(f"🆕 Creating complaint with partial ID proof")

    # Merge extracted data with manual input (manual overrides extracted)
    final_name = (payload.full_name or payload.extracted_name or "").strip()
    final_phone = (payload.phone_number or payload.extracted_phone or "").strip()
    final_email = (payload.email or payload.extracted_email or "").strip()
    final_address = (payload.address or payload.extracted_address or "").strip()

    if not final_name:
        raise HTTPException(status_code=422, detail="Name required - provide manually or ensure ID proof has visible name")
    if not final_phone:
        raise HTTPException(status_code=422, detail="Phone required - provide manually or ensure ID proof has visible phone")

    # Build merged complaint request
    merged_request = ComplaintCreateRequest(
        full_name=final_name,
        phone_number=final_phone,
        email=final_email if final_email else None,
        address=final_address if final_address else None,
        complaint_type=payload.complaint_type,
        description=payload.description,
        date_time=payload.date_time,
        amount_lost=payload.amount_lost,
        transaction_id=payload.transaction_id,
        platform=payload.platform,
        suspect_details=payload.suspect_details,
        suspect_vpa=payload.suspect_vpa,
        suspect_phone=payload.suspect_phone,
        suspect_bank_account=payload.suspect_bank_account,
        language=payload.language,
    )

    # Validate
    is_valid, validation_errors = enhanced_validation.validate_complaint_data(merged_request.model_dump())
    if not is_valid:
        raise HTTPException(status_code=422, detail=validation_errors)

    active_llm = get_active_llm_name()
    logger.info(f"✅ Processing with LLM: {active_llm}")

    # Create complaint
    user = complaint_service.upsert_user(db, merged_request)
    complaint = complaint_service.create_complaint(db, user_id=user.id, payload=merged_request)

    localized_summary = complaint_service.build_user_summary(merged_request, complaint.ticket_id)
    admin_summary_en = complaint_service.build_admin_english_summary(merged_request, complaint.ticket_id)

    try:
        localized_summary = translation_service.translate(localized_summary, payload.language) or localized_summary
    except Exception as e:
        logger.debug(f"Translation error (non-critical): {e}")

    try:
        email_service.send_complaint_notification(
            name=user.full_name,
            email=user.email,
            ticket_id=complaint.ticket_id,
            summary=localized_summary,
            admin_summary=admin_summary_en,
        )
    except Exception as e:
        logger.debug(f"Email error (non-critical): {e}")

    logger.info(f"✅ Complaint created (partial ID mode): {complaint.ticket_id}")

    return ComplaintCreateResponse(
        ticket_id=complaint.ticket_id,
        status="pending",
        created_at=complaint.created_at,
        message=f"Complaint created successfully with Ticket ID: {complaint.ticket_id}. "
                f"Some details used from ID proof, others provided manually.",
    )


@router.post("/upload")
def upload_evidence(
    ticket_id: str = Form(...),
    file_kind: str = Form("evidence"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Ticket not found")

    # Validate file before processing
    is_valid, error_msg = enhanced_validation.validate_file_upload(
        filename=file.filename or "unknown",
        file_size=file.size or 0,
        mime_type=file.content_type or "application/octet-stream"
    )
    if not is_valid:
        logger.warning("File validation failed for ticket %s: %s", ticket_id, error_msg)
        raise HTTPException(status_code=422, detail=error_msg)

    sub_dir = "id_proofs" if file_kind == "id_proof" else "evidence"
    file_path = save_upload_file(file, sub_dir=sub_dir)
    logger.info(f"📁 File saved for ticket {ticket_id}: {file.filename} → {file_path}")

    mime_type = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"
    extracted_text = ""
    
    try:
        logger.info(f"🔍 Starting evidence analysis for {ticket_id}: {file.filename}")
        extracted_text = gemini_service.analyze_evidence(file_path=file_path, mime_type=mime_type)
        logger.info(f"✅ Analysis completed: {len(extracted_text)} chars extracted")
    except Exception as exc:
        logger.error(f"❌ Evidence analysis error for {ticket_id}: {exc}", exc_info=True)
        # Use fallback
        extracted_text = f"[Evidence: {file.filename}]"

    # Ensure extracted_text is never empty
    if not extracted_text or not extracted_text.strip():
        extracted_text = f"[Evidence File: {file.filename} ({mime_type})]"
        logger.warning(f"Empty extracted_text for {ticket_id}, using fallback")

    evidence = Evidence(
        complaint_id=complaint.id,
        file_path=file_path,
        extracted_text=extracted_text,
    )
    db.add(evidence)
    db.commit()
    db.refresh(evidence)
    
    logger.info(f"💾 Evidence stored for ticket {ticket_id}: evidence_id={evidence.id}, text_len={len(evidence.extracted_text)}")

    return {
        "ticket_id": complaint.ticket_id,
        "evidence_id": evidence.id,
        "file_path": evidence.file_path,
        "extracted_text": evidence.extracted_text,
    }


@router.get("/{ticket_id}", response_model=ComplaintTrackResponse)
def track_complaint(ticket_id: str, db: Session = Depends(get_db)):
    complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    return ComplaintTrackResponse(
        ticket_id=complaint.ticket_id,
        status=complaint.status,
        complaint_type=complaint.complaint_type,
        description=complaint.description,
        language=complaint.language,
        created_at=complaint.created_at,
    )
