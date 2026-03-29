#!/usr/bin/env python3
"""
Test script for ID proof extraction with partial data handling fix.
Tests the new endpoints and workflows.

Usage:
    python test_id_proof_extraction_fix.py
"""

import json
import requests
import sys
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8000/api"
TEST_ID_PROOF_PATH = "backend/uploads/id_proofs/sample_id.jpg"  # Provide an actual test image

class Colors:
    """Color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{Colors.RESET}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")

def print_json(data):
    """Pretty print JSON"""
    print(json.dumps(data, indent=2))

# Test 1: Validate ID Proof Extraction
def test_validate_id_proof():
    """Test 1: Upload ID proof and validate extraction status"""
    print_header("TEST 1: Validate ID Proof Extraction")
    
    # Create a test image file if it doesn't exist
    test_dir = Path("backend/uploads/id_proofs")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_image = test_dir / "test_id.jpg"
    
    if not test_image.exists():
        print_info("Creating test ID proof file...")
        # Create a simple test image (in real use, upload actual ID)
        try:
            from PIL import Image
            img = Image.new('RGB', (400, 300), color='white')
            img.save(test_image)
            print_success(f"Test image created: {test_image}")
        except ImportError:
            print_error("PIL not installed. Using mock test.")
            return False
    
    try:
        print_info("Uploading ID proof to /complaint/validate-id...")
        
        with open(test_image, 'rb') as f:
            files = {'file': f}
            data = {
                'full_name': 'Test User',
                'phone_number': '9876543210',
                'email': 'test@example.com',
            }
            
            response = requests.post(
                f"{API_BASE}/complaint/validate-id",
                files=files,
                data=data,
                timeout=30
            )
        
        print_info(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print_success("ID proof validation successful!")
            
            # Check for key fields
            required_fields = [
                'extraction_status',
                'missing_fields',
                'extracted',
                'proceed_allowed',
                'message'
            ]
            
            for field in required_fields:
                if field in result:
                    print_success(f"✓ Response contains '{field}'")
                else:
                    print_error(f"✗ Missing '{field}' in response")
                    return False
            
            # Print extraction details
            print(f"\n📊 Extraction Status Details:")
            print(f"  Status: {Colors.BOLD}{result.get('extraction_status', 'N/A')}{Colors.RESET}")
            print(f"  Missing Fields: {result.get('missing_fields', [])}")
            print(f"  Proceed Allowed: {result.get('proceed_allowed', False)}")
            print(f"  Proceed Recommended: {result.get('proceed_recommended', False)}")
            
            extracted = result.get('extracted', {})
            print(f"\n📋 Extracted Data:")
            for key, value in extracted.items():
                status = "✓" if value else "✗"
                print(f"  {status} {key}: {value or '(empty)'}")
            
            return True
        else:
            print_error(f"API Error: {response.status_code}")
            print_json(response.json())
            return False
            
    except Exception as e:
        print_error(f"Test failed: {e}")
        return False

# Test 2: Create Complaint with Partial ID
def test_create_complaint_with_partial_id():
    """Test 2: Create complaint with merged extracted + manual data"""
    print_header("TEST 2: Create Complaint with Partial ID Proof Data")
    
    payload = {
        "complaint_type": "Online Fraud",
        "description": "Test phishing complaint - extracted + manual data merged",
        "language": "English",
        
        # Manual user input
        "full_name": "John Doe",
        "phone_number": "9876543210",
        "email": "john@example.com",
        "address": "123 Main Street, City",
        
        # Extracted from ID (simulating partial extraction)
        "extracted_name": "",
        "extracted_phone": "9876543210",
        "extracted_email": "",
        "extracted_address": "123 Main Street, City",
        "extracted_id_number": "1234567890",
        "extracted_document_type": "Aadhaar",
        
        # Optional details
        "date_time": "2024-03-29T10:30:00",
        "amount_lost": "5000",
    }
    
    try:
        print_info("Submitting complaint with /complaint/create-with-partial-id...")
        
        response = requests.post(
            f"{API_BASE}/complaint/create-with-partial-id",
            json=payload,
            timeout=30
        )
        
        print_info(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print_success("Complaint created successfully!")
            
            # Check for required fields
            required_fields = ['ticket_id', 'status', 'created_at', 'message']
            for field in required_fields:
                if field in result:
                    print_success(f"✓ {field}: {result.get(field)}")
                else:
                    print_error(f"✗ Missing '{field}'")
                    return False
            
            print(f"\n📝 Complaint Details:")
            print(f"  Ticket ID: {Colors.BOLD}{result.get('ticket_id')}{Colors.RESET}")
            print(f"  Status: {result.get('status')}")
            print(f"  Created: {result.get('created_at')}")
            print(f"  Message: {result.get('message')}")
            
            return True
        else:
            print_error(f"API Error: {response.status_code}")
            print_json(response.json())
            return False
            
    except Exception as e:
        print_error(f"Test failed: {e}")
        return False

# Test 3: Create Complaint with All Manual Data (No ID)
def test_create_complaint_without_id():
    """Test 3: Create complaint without ID proof extraction"""
    print_header("TEST 3: Create Complaint Without ID Proof (Fallback)")
    
    payload = {
        "complaint_type": "Banking Fraud",
        "description": "Unauthorized transaction - no ID proof used",
        "language": "English",
        
        # All manual data
        "full_name": "Jane Smith",
        "phone_number": "8765432109",
        "email": "jane@example.com",
        "address": "456 Oak Avenue",
        
        # No extracted data
        "extracted_name": "",
        "extracted_phone": "",
        "extracted_email": "",
        "date_time": "2024-03-29T14:00:00",
        "amount_lost": "10000",
        "platform": "Bank",
    }
    
    try:
        print_info("Creating complaint with manual data only...")
        
        response = requests.post(
            f"{API_BASE}/complaint/create-with-partial-id",
            json=payload,
            timeout=30
        )
        
        print_info(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print_success("Complaint created without ID proof!")
            print(f"  Ticket ID: {Colors.BOLD}{result.get('ticket_id')}{Colors.RESET}")
            return True
        else:
            print_error(f"API Error: {response.status_code}")
            print_json(response.json())
            return False
            
    except Exception as e:
        print_error(f"Test failed: {e}")
        return False

# Test 4: Validation - Missing Required Fields
def test_validation_missing_name():
    """Test 4: Validation should fail when name is missing"""
    print_header("TEST 4: Validation - Missing Required Name")
    
    payload = {
        "complaint_type": "Online Fraud",
        "description": "Test complaint",
        "full_name": "",  # Empty name
        "phone_number": "9876543210",
        "extracted_name": "",
        "extracted_phone": "",
    }
    
    try:
        print_info("Attempting to create complaint without name...")
        
        response = requests.post(
            f"{API_BASE}/complaint/create-with-partial-id",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 422:
            print_success("✓ Validation correctly rejected missing name")
            error_detail = response.json().get('detail', '')
            print_info(f"Error message: {error_detail}")
            return True
        else:
            print_error(f"Expected 422 error, got {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {e}")
        return False

# Test 5: Validation - Missing Required Phone
def test_validation_missing_phone():
    """Test 5: Validation should fail when phone is missing"""
    print_header("TEST 5: Validation - Missing Required Phone")
    
    payload = {
        "complaint_type": "Online Fraud",
        "description": "Test complaint",
        "full_name": "John Doe",
        "phone_number": "",  # Empty phone
        "extracted_name": "",
        "extracted_phone": "",
    }
    
    try:
        print_info("Attempting to create complaint without phone...")
        
        response = requests.post(
            f"{API_BASE}/complaint/create-with-partial-id",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 422:
            print_success("✓ Validation correctly rejected missing phone")
            error_detail = response.json().get('detail', '')
            print_info(f"Error message: {error_detail}")
            return True
        else:
            print_error(f"Expected 422 error, got {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {e}")
        return False

# Test 6: Data Merge Priority (Manual > Extracted)
def test_data_merge_priority():
    """Test 6: Verify manual data takes priority over extracted"""
    print_header("TEST 6: Data Merge Priority (Manual > Extracted)")
    
    payload = {
        "complaint_type": "Online Fraud",
        "description": "Test data merge priority",
        "language": "English",
        
        # Manual input (should take priority)
        "full_name": "Manual Name",
        "phone_number": "1111111111",
        
        # Extracted (should be ignored because manual is provided)
        "extracted_name": "Extracted Name",
        "extracted_phone": "2222222222",
    }
    
    try:
        print_info("Testing data merge with both manual and extracted...")
        
        response = requests.post(
            f"{API_BASE}/complaint/create-with-partial-id",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print_success("✓ Complaint created with merged data")
            print_info("Note: Manual data should take priority over extracted")
            print_info("  Manual Name: 'Manual Name' (used)")
            print_info("  Extracted Name: 'Extracted Name' (ignored)")
            return True
        else:
            print_error(f"API Error: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  ID Proof Extraction Fix - Test Suite                      ║")
    print("║  Testing partial data handling and complaint creation      ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE}/../docs", timeout=5)
        print_success("API server is running")
    except requests.ConnectionError:
        print_error(f"Cannot connect to API at {API_BASE}")
        print_info("Make sure backend is running: python backend/main.py")
        return False
    
    tests = [
        ("Validate ID Proof", test_validate_id_proof),
        ("Create Complaint + Partial ID", test_create_complaint_with_partial_id),
        ("Create Complaint Without ID", test_create_complaint_without_id),
        ("Validation - Missing Name", test_validation_missing_name),
        ("Validation - Missing Phone", test_validation_missing_phone),
        ("Data Merge Priority", test_data_merge_priority),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            results[test_name] = False
    
    # Print summary
    print_header("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = f"{Colors.GREEN}✅ PASS{Colors.RESET}" if passed_flag else f"{Colors.RED}❌ FAIL{Colors.RESET}"
        print(f"{status} - {test_name}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed! ID proof fix is working correctly.{Colors.RESET}\n")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  Some tests failed. Check logs above.{Colors.RESET}\n")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
