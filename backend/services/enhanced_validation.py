"""
Enhanced Input Validation Service with comprehensive checks.
"""
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Validation constants
PHONE_PATTERN = r'^[6-9]\d{9}$'
EMAIL_PATTERN = r'^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$'
UPI_PATTERN = r'^[a-zA-Z0-9.\-_]{3,}@[a-zA-Z]{2,}$'
BANK_ACCOUNT_PATTERN = r'^\d{9,18}$'
IFSC_PATTERN = r'^[A-Z]{4}0[A-Z0-9]{6}$'

# Field length limits
FIELD_LIMITS = {
    'full_name': (2, 100),
    'email': (5, 254),
    'phone': (10, 10),  # Fixed length for Indian numbers
    'address': (5, 500),
    'description': (10, 2000),
    'complaint_type': (5, 100),
    'date_time': (5, 100),
    'platform': (2, 100),
    'amount_lost': (1, 15),  # String representation of number
    'transaction_id': (3, 100),
    'suspect_vpa': (3, 100),
    'suspect_phone': (10, 10),
    'suspect_bank_account': (9, 18),
    'suspect_details': (5, 500),
}

# Sensitive data patterns to detect
SUSPICIOUS_PATTERNS = {
    'script_tags': r'<script|javascript:|on\w+\s*=',
    'sql_injection': r'(\bor\b|\bunion\b|\bselect\b|\bdrop\b|\binsert\b)',
    'command_injection': r'[\`$(){}[]|&;]',
}


class ValidationService:
    """Comprehensive input validation for complaint system."""

    @staticmethod
    def validate_phone(phone: str) -> tuple[bool, Optional[str]]:
        """Validate Indian phone number."""
        if not phone:
            return False, "Phone number is required"
        
        # Remove spaces and dashes
        cleaned = re.sub(r'[\s\-]', '', phone)
        
        if not re.match(PHONE_PATTERN, cleaned):
            return False, f"Invalid phone format. Expected 10-digit Indian number starting with 6-9, got: {phone}"
        
        return True, None

    @staticmethod
    def validate_email(email: str) -> tuple[bool, Optional[str]]:
        """Validate email address."""
        if not email:
            return True, None  # Optional field
        
        # Check for N/A markers
        if email.lower().strip() in {"n/a", "na", "none", "skip"}:
            return True, None
        
        if len(email) > FIELD_LIMITS['email'][1]:
            return False, f"Email too long (max {FIELD_LIMITS['email'][1]} chars)"
        
        if not re.match(EMAIL_PATTERN, email, re.IGNORECASE):
            return False, f"Invalid email format: {email}"
        
        return True, None

    @staticmethod
    def validate_string_field(field_name: str, value: str) -> tuple[bool, Optional[str]]:
        """Validate string field with length limits."""
        if not value:
            return False, f"{field_name} is required"
        
        if field_name not in FIELD_LIMITS:
            return True, None  # No limit defined
        
        min_len, max_len = FIELD_LIMITS[field_name]
        value_len = len(value.strip())
        
        if value_len < min_len:
            return False, f"{field_name} too short (min {min_len} chars)"
        
        if value_len > max_len:
            return False, f"{field_name} too long (max {max_len} chars)"
        
        return True, None

    @staticmethod
    def validate_amount(amount: Optional[str]) -> tuple[bool, Optional[str]]:
        """Validate financial amount."""
        if not amount or amount.lower().strip() in {"n/a", "na", "none"}:
            return True, None
        
        try:
            amount_float = float(amount)
            if amount_float < 0:
                return False, "Amount cannot be negative"
            if amount_float > 999999999:  # Max ~$1B
                return False, "Amount seems unrealistic (exceeds 999,999,999)"
            return True, None
        except ValueError:
            return False, f"Invalid amount format: {amount}"

    @staticmethod
    def validate_upi(upi: Optional[str]) -> tuple[bool, Optional[str]]:
        """Validate UPI address format."""
        if not upi or upi.lower().strip() in {"n/a", "na", "none"}:
            return True, None
        
        if not re.match(UPI_PATTERN, upi, re.IGNORECASE):
            return False, f"Invalid UPI format. Expected user@bank (e.g., user@okhdfcbank), got: {upi}"
        
        return True, None

    @staticmethod
    def validate_bank_account(account: Optional[str]) -> tuple[bool, Optional[str]]:
        """Validate bank account number."""
        if not account or account.lower().strip() in {"n/a", "na", "none"}:
            return True, None
        
        # Remove spaces
        cleaned = re.sub(r'\s+', '', account)
        
        if not re.match(BANK_ACCOUNT_PATTERN, cleaned):
            return False, f"Invalid bank account. Expected 9-18 digits, got: {account}"
        
        return True, None

    @staticmethod
    def check_injection_risks(text: str) -> List[str]:
        """Detect potential code injection attempts."""
        risks = []
        
        for risk_type, pattern in SUSPICIOUS_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                risks.append(risk_type)
        
        return risks

    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input by removing dangerous characters."""
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        # Remove excess whitespace
        text = ' '.join(text.split())
        return text.strip()

    @staticmethod
    def validate_file_upload(filename: str, file_size: int, mime_type: str) -> tuple[bool, Optional[str]]:
        """Validate file upload (size, type, name)."""
        # Size validation (max 50MB per file)
        MAX_SIZE = 50 * 1024 * 1024
        if file_size > MAX_SIZE:
            return False, f"File too large (max 50MB, got {file_size / 1024 / 1024:.1f}MB)"
        
        # Allowed MIME types
        ALLOWED_TYPES = {
            'image/jpeg', 'image/png', 'image/webp', 'image/gif',
            'video/mp4', 'video/mpeg', 'video/quicktime',
            'application/pdf', 'text/plain',
            'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        
        if mime_type not in ALLOWED_TYPES:
            return False, f"File type not allowed: {mime_type}"
        
        # Filename validation
        if not filename or len(filename) > 255:
            return False, "Invalid filename"
        
        # Check for dangerous extensions
        dangerous_exts = {'.exe', '.sh', '.bat', '.cmd', '.com', '.pif', '.scr'}
        file_ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        
        if file_ext in dangerous_exts:
            return False, f"File type not allowed: {file_ext}"
        
        return True, None

    @staticmethod
    def validate_complaint_data(data: dict) -> tuple[bool, List[str]]:
        """Comprehensive validation of complaint data."""
        errors = []
        
        # Required fields
        required = ['full_name', 'phone_number', 'complaint_type', 'description']
        for field in required:
            if not data.get(field):
                errors.append(f"{field} is required")
        
        # Phone validation
        phone = data.get('phone_number', '')
        if phone:
            valid, error = ValidationService.validate_phone(phone)
            if not valid:
                errors.append(error)
        
        # Email validation
        email = data.get('email', '')
        if email:
            valid, error = ValidationService.validate_email(email)
            if not valid:
                errors.append(error)
        
        # Amount validation
        amount = data.get('amount_lost')
        if amount:
            valid, error = ValidationService.validate_amount(amount)
            if not valid:
                errors.append(error)
        
        # UPI validation
        upi = data.get('suspect_vpa')
        if upi:
            valid, error = ValidationService.validate_upi(upi)
            if not valid:
                errors.append(error)
        
        # Bank account validation
        account = data.get('suspect_bank_account')
        if account:
            valid, error = ValidationService.validate_bank_account(account)
            if not valid:
                errors.append(error)
        
        # String field limits
        string_fields = ['description', 'address', 'suspect_details']
        for field in string_fields:
            value = data.get(field, '')
            if value:
                valid, error = ValidationService.validate_string_field(field, value)
                if not valid:
                    errors.append(error)
        
        # Check for injection attempts
        all_text = ' '.join(str(v) for v in data.values() if isinstance(v, str))
        injection_risks = ValidationService.check_injection_risks(all_text)
        if injection_risks:
            errors.append(f"Suspicious content detected: {', '.join(injection_risks)}")
        
        return len(errors) == 0, errors


# Export singleton instance
validation_service = ValidationService()
