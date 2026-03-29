from datetime import datetime
from pydantic import BaseModel, Field


class ComplaintCreateRequest(BaseModel):
    full_name: str
    phone_number: str
    email: str | None = None
    address: str | None = None
    complaint_type: str
    date_time: str | None = None
    description: str
    amount_lost: str | None = None
    transaction_id: str | None = None
    platform: str | None = None
    suspect_details: str | None = None
    suspect_vpa: str | None = None
    suspect_phone: str | None = None
    suspect_bank_account: str | None = None
    language: str = "English"


class ComplaintCreateRequest(BaseModel):
    """Standard complaint creation with full details."""
    full_name: str
    phone_number: str
    email: str | None = None
    address: str | None = None
    complaint_type: str
    date_time: str | None = None
    description: str
    amount_lost: str | None = None
    transaction_id: str | None = None
    platform: str | None = None
    suspect_details: str | None = None
    suspect_vpa: str | None = None
    suspect_phone: str | None = None
    suspect_bank_account: str | None = None
    language: str = "English"


class ComplaintCreateWithPartialIDRequest(BaseModel):
    """Complaint creation allowing partial/unclear ID proof data with manual override."""
    # Required basic details
    complaint_type: str
    description: str
    language: str = "English"
    
    # User's manual input (optional - can use extracted data if available)
    full_name: str | None = None
    phone_number: str | None = None
    email: str | None = None
    address: str | None = None
    
    # ID Proof extracted data (merged with manual if user provides)
    extracted_name: str | None = None
    extracted_phone: str | None = None
    extracted_email: str | None = None
    extracted_address: str | None = None
    extracted_id_number: str | None = None
    extracted_document_type: str | None = None
    
    # Additional optional complaint details
    date_time: str | None = None
    amount_lost: str | None = None
    transaction_id: str | None = None
    platform: str | None = None
    suspect_details: str | None = None
    suspect_vpa: str | None = None
    suspect_phone: str | None = None
    suspect_bank_account: str | None = None
    
    # Optional metadata
    id_proof_id_number: str | None = None
    id_proof_document_type: str | None = None
class ComplaintStatusUpdateRequest(BaseModel):
    ticket_id: str = Field(..., min_length=4)
    status: str = Field(..., pattern="^(pending|reviewing|resolved)$")
