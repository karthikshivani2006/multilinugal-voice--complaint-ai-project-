"""
Comprehensive Backend Tests for CyberGuard AI
"""
import pytest
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.enhanced_validation import validation_service
from services.llm_selector import get_llm_service, get_active_llm_name


class TestValidationService:
    """Test input validation."""

    def test_valid_phone(self):
        """Test valid Indian phone numbers."""
        valid, error = validation_service.validate_phone("9876543210")
        assert valid and error is None, f"Should accept valid phone, got: {error}"

    def test_invalid_phone_format(self):
        """Test invalid phone formats."""
        test_cases = [
            ("1234567890", "Should reject 4-5 starting digits"),
            ("12345", "Should reject short numbers"),
            ("abcdefghij", "Should reject non-digits"),
        ]
        for phone, description in test_cases:
            valid, error = validation_service.validate_phone(phone)
            assert not valid, f"{description}"

    def test_phone_with_spaces(self):
        """Test phone with spaces/dashes."""
        valid, error = validation_service.validate_phone("9876 543 210")
        assert valid, f"Should accept phone with spaces, got error: {error}"

    def test_valid_email(self):
        """Test valid email addresses."""
        valid, error = validation_service.validate_email("user@example.com")
        assert valid, f"Should accept valid email, got: {error}"

    def test_email_na_marker(self):
        """Test N/A email marker."""
        for na_variant in ["n/a", "N/A", "na", "none"]:
            valid, error = validation_service.validate_email(na_variant)
            assert valid, f"Should accept N/A variant '{na_variant}'"

    def test_invalid_email(self):
        """Test invalid emails."""
        test_cases = [
            "notanemail",
            "user@",
            "@example.com",
            "user@.com",
        ]
        for email in test_cases:
            valid, error = validation_service.validate_email(email)
            assert not valid, f"Should reject invalid email: {email}"

    def test_valid_amount(self):
        """Test valid financial amounts."""
        test_cases = ["100", "5000.50", "1000000"]
        for amount in test_cases:
            valid, error = validation_service.validate_amount(amount)
            assert valid, f"Should accept valid amount '{amount}', got: {error}"

    def test_invalid_amount(self):
        """Test invalid amounts."""
        test_cases = [
            ("-100", "Negative amount"),
            ("999999999999", "Unrealistic amount"),
            ("abc", "Non-numeric"),
        ]
        for amount, description in test_cases:
            valid, error = validation_service.validate_amount(amount)
            assert not valid, f"{description}: {error}"

    def test_valid_upi(self):
        """Test valid UPI addresses."""
        test_cases = [
            "user@okhdfcbank",
            "customer@ibl",
            "merchant@ybl",
        ]
        for upi in test_cases:
            valid, error = validation_service.validate_upi(upi)
            assert valid, f"Should accept valid UPI '{upi}', got: {error}"

    def test_invalid_upi(self):
        """Test invalid UPI addresses."""
        test_cases = ["notaupi", "@bank", "user@"]
        for upi in test_cases:
            valid, error = validation_service.validate_upi(upi)
            assert not valid, f"Should reject invalid UPI: {upi}"

    def test_valid_bank_account(self):
        """Test valid bank account numbers."""
        test_cases = ["123456789", "123456789012345678"]
        for account in test_cases:
            valid, error = validation_service.validate_bank_account(account)
            assert valid, f"Should accept valid account '{account}', got: {error}"

    def test_injection_detection(self):
        """Test injection risk detection."""
        test_cases = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE complaints; --",
            "$(rm -rf /)",
        ]
        for payload in test_cases:
            risks = validation_service.check_injection_risks(payload)
            assert len(risks) > 0, f"Should detect injection in: {payload}"

    def test_sanitize_input(self):
        """Test input sanitization."""
        dirty = "  Hello   World  \t\n  "
        clean = validation_service.sanitize_input(dirty)
        assert clean == "Hello World", f"Should sanitize input: {clean}"

    def test_file_upload_validation(self):
        """Test file upload validation."""
        # Valid file
        valid, error = validation_service.validate_file_upload(
            "evidence.pdf",
            1024 * 1024,  # 1MB
            "application/pdf"
        )
        assert valid, f"Should accept valid file: {error}"

        # Too large
        valid, error = validation_service.validate_file_upload(
            "large.pdf",
            100 * 1024 * 1024,  # 100MB
            "application/pdf"
        )
        assert not valid, "Should reject too large file"

        # Invalid type
        valid, error = validation_service.validate_file_upload(
            "malware.exe",
            1024,
            "application/x-msdownload"
        )
        assert not valid, "Should reject .exe files"

    def test_comprehensive_complaint_validation(self):
        """Test full complaint data validation."""
        valid_data = {
            'full_name': 'John Doe',
            'phone_number': '9876543210',
            'email': 'john@example.com',
            'complaint_type': 'UPI Fraud',
            'description': 'I was scammed via UPI',
        }

        valid, errors = validation_service.validate_complaint_data(valid_data)
        assert valid and len(errors) == 0, f"Valid data should pass: {errors}"

    def test_complaint_validation_with_errors(self):
        """Test complaint validation with invalid data."""
        invalid_data = {
            'full_name': '',  # Empty
            'phone_number': '1234567890',  # Invalid
            'email': 'notanemail',  # Invalid
            'complaint_type': 'UPI Fraud',
            'description': 'x',  # Too short
            'amount_lost': '-100',  # Negative
        }

        valid, errors = validation_service.validate_complaint_data(invalid_data)
        assert not valid and len(errors) > 0, f"Invalid data should fail validation"


class TestLLMSelector:
    """Test LLM service selection and fallback."""

    def test_get_active_llm(self):
        """Test that a service is available."""
        try:
            llm = get_llm_service()
            assert llm is not None, "Should return an LLM service"
            print(f"✅ Active LLM: {get_active_llm_name()}")
        except Exception as e:
            pytest.skip(f"No LLM available: {e}")

    def test_llm_name(self):
        """Test LLM name detection."""
        name = get_active_llm_name()
        assert name in ["gemini-2.0-flash", "ollama-llama2", "unknown"], \
            f"Unexpected LLM name: {name}"


if __name__ == "__main__":
    # Run tests
    print("=" * 70)
    print("CyberGuard Backend Test Suite")
    print("=" * 70 + "\n")

    pytest.main([__file__, "-v", "--tb=short"])
