from sqlalchemy.orm import Session

from models.complaint import Complaint
from models.user import User
from schemas.complaint import ComplaintCreateRequest
from utils.ticket_service import generate_ticket_id


class ComplaintService:
    def _normalize_optional(self, value: str | None, default: str = "N/A") -> str:
        text = (value or "").strip()
        if not text or text.lower() in {"n/a", "na", "none", "skip"}:
            return default
        return text

    def _build_fallback_email(self, payload: ComplaintCreateRequest) -> str:
        phone_digits = "".join(ch for ch in (payload.phone_number or "") if ch.isdigit())
        suffix = phone_digits[-10:] if phone_digits else "unknown"
        return f"noemail_{suffix}@cyberguard.local"

    def upsert_user(self, db: Session, payload: ComplaintCreateRequest) -> User:
        normalized_email = self._normalize_optional(payload.email, "")
        normalized_address = self._normalize_optional(payload.address)
        target_email = normalized_email or self._build_fallback_email(payload)

        user = db.query(User).filter(User.email == target_email).first()
        if user:
            user.name = payload.full_name
            user.phone = payload.phone_number
            user.address = normalized_address
            db.add(user)
            db.commit()
            db.refresh(user)
            return user

        user = User(
            name=payload.full_name,
            email=target_email,
            phone=payload.phone_number,
            address=normalized_address,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def create_complaint(self, db: Session, user_id: int, payload: ComplaintCreateRequest) -> Complaint:
        complaint = Complaint(
            ticket_id=generate_ticket_id(),
            user_id=user_id,
            complaint_type=payload.complaint_type,
            description=payload.description,
            status="pending",
            language=payload.language,
            date_time=self._normalize_optional(payload.date_time),
            amount_lost=payload.amount_lost,
            transaction_id=payload.transaction_id,
            platform=payload.platform,
            suspect_details=payload.suspect_details,
            suspect_vpa=payload.suspect_vpa,
            suspect_phone=payload.suspect_phone,
            suspect_bank_account=payload.suspect_bank_account,
        )
        db.add(complaint)
        db.commit()
        db.refresh(complaint)
        return complaint

    def build_user_summary(self, payload: ComplaintCreateRequest, ticket_id: str) -> str:
        return (
            f"Ticket ID: {ticket_id}. "
            f"Type: {payload.complaint_type}. "
            f"Description: {payload.description}. "
            f"Date and Time: {payload.date_time}."
        )

    def build_admin_english_summary(self, payload: ComplaintCreateRequest, ticket_id: str) -> str:
        return (
            f"Ticket ID: {ticket_id}\n"
            f"Name: {payload.full_name}\n"
            f"Phone: {payload.phone_number}\n"
            f"Email: {self._normalize_optional(payload.email)}\n"
            f"Address: {self._normalize_optional(payload.address)}\n"
            f"Type: {payload.complaint_type}\n"
            f"Date and Time: {self._normalize_optional(payload.date_time)}\n"
            f"Description: {payload.description}\n"
            f"Amount Lost: {payload.amount_lost or 'N/A'}\n"
            f"Transaction ID/UTR: {payload.transaction_id or 'N/A'}\n"
            f"Platform: {payload.platform or 'N/A'}\n"
            f"Suspect Details: {payload.suspect_details or 'N/A'}\n"
            f"Original Language: {payload.language}"
        )


complaint_service = ComplaintService()
