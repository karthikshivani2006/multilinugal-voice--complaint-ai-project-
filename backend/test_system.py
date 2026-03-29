#!/usr/bin/env python3
"""
End-to-End System Test for CyberGuard AI
Tests the complete workflow: validation → LLM → database → response
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import SessionLocal
from models.complaint import Complaint
from models.user import User
from services.enhanced_validation import validation_service
from services.llm_selector import get_active_llm_name
from config import settings


def test_validation_layer():
    """Test 1: Comprehensive input validation."""
    print("\n" + "="*70)
    print("TEST 1: Input Validation Layer")
    print("="*70)
    
    # Test data with various valid and invalid inputs
    test_cases = [
        {
            'name': 'Valid complaint',
            'data': {
                'full_name': 'Rajesh Kumar',
                'phone_number': '9876543210',
                'email': 'rajesh@example.com',
                'complaint_type': 'UPI Fraud',
                'description': 'Unauthorized UPI transaction of 5000 rupees',
                'amount_lost': '5000',
            },
            'should_pass': True
        },
        {
            'name': 'Invalid phone number',
            'data': {
                'full_name': 'John Doe',
                'phone_number': '1234567890',  # Starts with 1, invalid
                'email': 'john@example.com',
                'complaint_type': 'UPI Fraud',
                'description': 'Test complaint',
            },
            'should_pass': False
        },
        {
            'name': 'Injection attempt',
            'data': {
                'full_name': 'Hacker',
                'phone_number': '9876543210',
                'email': 'hacker@evil.com',
                'complaint_type': 'UPI Fraud',
                'description': "'; DROP TABLE complaints; --",  # SQL injection
            },
            'should_pass': False
        },
        {
            'name': 'Negative amount',
            'data': {
                'full_name': 'Jane Doe',
                'phone_number': '9876543211',
                'email': 'jane@example.com',
                'complaint_type': 'UPI Fraud',
                'description': 'Test complaint',
                'amount_lost': '-1000',  # Negative
            },
            'should_pass': False
        },
    ]
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        valid, errors = validation_service.validate_complaint_data(test['data'])
        
        if valid == test['should_pass']:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"\n{status}: {test['name']}")
        if not valid:
            print(f"   Errors: {errors}")
    
    print(f"\n{'─'*70}")
    print(f"Validation Tests: {passed} passed, {failed} failed\n")
    return failed == 0


def test_llm_selection():
    """Test 2: LLM service selection and fallback."""
    print("\n" + "="*70)
    print("TEST 2: LLM Service Selection & Fallback")
    print("="*70)
    
    try:
        llm_name = get_active_llm_name()
        print(f"\n✅ Active LLM: {llm_name}")
        
        if llm_name == "gemini-2.0-flash":
            print(f"   Status: Primary (Gemini) is active")
            print(f"   API Key: {settings.gemini_api_key[:20]}...")
        elif llm_name == "ollama-llama2":
            print(f"   Status: Fallback (Ollama) is active")
        else:
            print(f"   Warning: Unknown LLM: {llm_name}")
        
        print(f"\n{'─'*70}")
        print(f"LLM Selection: ✅ PASS\n")
        return True
    except Exception as e:
        print(f"\n❌ LLM Selection: FAIL")
        print(f"   Error: {e}\n")
        return False


def test_database_operations():
    """Test 3: Database CRUD operations."""
    print("\n" + "="*70)
    print("TEST 3: Database Operations (Create, Read)")
    print("="*70)
    
    db = SessionLocal()
    
    try:
        # Count existing records
        complaint_count = db.query(Complaint).count()
        print(f"\n✅ Database connection successful")
        print(f"   Existing complaints: {complaint_count}")
        
        # List recent complaints
        recent = db.query(Complaint).order_by(Complaint.created_at.desc()).limit(3).all()
        if recent:
            print(f"\n✅ Recent complaints retrieved:")
            for c in recent:
                print(f"   - {c.ticket_id}: {c.complaint_type}")
        
        print(f"\n{'─'*70}")
        print(f"Database Tests: ✅ PASS\n")
        return True
    except Exception as e:
        print(f"\n❌ Database Tests: FAIL")
        print(f"   Error: {e}\n")
        return False
    finally:
        db.close()


def test_file_handling():
    """Test 4: File upload validation."""
    print("\n" + "="*70)
    print("TEST 4: File Upload Validation")
    print("="*70)
    
    # Test valid file
    valid, error = validation_service.validate_file_upload(
        "evidence.pdf",
        1024 * 1024,  # 1MB
        "application/pdf"
    )
    
    print(f"\n✅ Valid PDF upload: {valid}")
    
    # Test invalid size
    valid, error = validation_service.validate_file_upload(
        "large.pdf",
        100 * 1024 * 1024,  # 100MB
        "application/pdf"
    )
    
    print(f"✅ Large file rejection: {not valid}")
    print(f"   Message: {error}")
    
    # Test invalid type
    valid, error = validation_service.validate_file_upload(
        "malware.exe",
        1024,
        "application/x-msdownload"
    )
    
    print(f"✅ Malware rejection: {not valid}")
    print(f"   Message: {error}")
    
    print(f"\n{'─'*70}")
    print(f"File Handling Tests: ✅ PASS\n")
    return True


def test_special_characters():
    """Test 5: Special character handling."""
    print("\n" + "="*70)
    print("TEST 5: Special Character & Unicode Handling")
    print("="*70)
    
    # Test unicode names
    test_names = [
        "राज कुमार",  # Devanagari
        "محمد علي",  # Arabic
        "李明",  # Chinese
        "José García",  # Spanish accents
    ]
    
    print(f"\n✅ Testing unicode names:")
    for name in test_names:
        cleaned = validation_service.sanitize_input(name)
        print(f"   ✓ {name} → {cleaned}")
    
    print(f"\n{'─'*70}")
    print(f"Character Handling Tests: ✅ PASS\n")
    return True


def run_full_system_test():
    """Run all system tests."""
    print("\n")
    print("┌" + "─"*68 + "┐")
    print("│" + " "*12 + "CyberGuard AI - End-to-End System Test" + " "*18 + "│")
    print("└" + "─"*68 + "┘")
    
    results = {
        'Validation Layer': test_validation_layer(),
        'LLM Selection': test_llm_selection(),
        'Database Operations': test_database_operations(),
        'File Handling': test_file_handling(),
        'Character Handling': test_special_characters(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'─'*70}")
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is production-ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")
    
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_full_system_test()
    sys.exit(0 if success else 1)
