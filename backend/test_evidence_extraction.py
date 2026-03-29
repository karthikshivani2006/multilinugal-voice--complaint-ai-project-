#!/usr/bin/env python3
"""
Test script to verify evidence extraction is working properly
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.gemini_service import gemini_service
from services.enhanced_validation import validation_service
from config import settings


def test_evidence_extraction():
    """Test evidence text extraction with various file types."""
    print("\n" + "="*70)
    print("TEST: Evidence File Text Extraction")
    print("="*70)
    
    # Create a test text file
    test_file = Path("/tmp/test_evidence.txt")
    test_content = """
    COMPLAINT DETAILS:
    Transaction Date: 2024-01-15
    Amount: ₹5000
    Transaction ID: UTR-2024-001-ABC123
    
    Account Details:
    Account Holder: Rajesh Kumar
    Account Number: 1234567890123456
    IFSC Code: SBIN0002345
    
    Suspicious Activity:
    - Unauthorized transaction
    - Recipient account unknown
    - No authorization given
    
    Bank Reference: BRN-2024-001-XYZ
    """
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    print(f"\n1️⃣  Created test file: {test_file}")
    print(f"   Size: {test_file.stat().st_size} bytes")
    
    # Test file validation
    print(f"\n2️⃣  Validating file upload...")
    is_valid, error = validation_service.validate_file_upload(
        filename="test_evidence.txt",
        file_size=test_file.stat().st_size,
        mime_type="text/plain"
    )
    print(f"   Valid: {is_valid}")
    if not is_valid:
        print(f"   Error: {error}")
    else:
        print(f"   ✅ File validation passed")
    
    # Test evidence extraction
    print(f"\n3️⃣  Extracting evidence text...")
    
    if not settings.gemini_api_key or settings.gemini_api_key == "dev_key_placeholder":
        print(f"   ⚠️  Gemini API key not configured")
        print(f"   Using fallback extraction")
    else:
        print(f"   Using Gemini API extraction")
    
    try:
        extracted_text = gemini_service.analyze_evidence(
            file_path=str(test_file),
            mime_type="text/plain"
        )
        
        print(f"\n4️⃣  Extraction Result:")
        print(f"   Status: ✅ SUCCESS")
        print(f"   Text Length: {len(extracted_text)} characters")
        print(f"   First 200 chars:\n   {extracted_text[:200]}")
        
        # Validate extracted text
        if extracted_text and len(extracted_text.strip()) > 0:
            print(f"\n   ✅ Extracted text is not empty")
            
            # Check for key indicators
            indicators = ["Transaction", "₹", "Account", "2024", "Amount"]
            found = [ind for ind in indicators if ind in extracted_text]
            if found:
                print(f"   ✅ Found key indicators: {', '.join(found)}")
            else:
                print(f"   ℹ️  No specific indicators found (content may be generic)")
        else:
            print(f"   ❌ Extracted text is empty!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ EXTRACTION FAILED:")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
            print(f"\n✓ Cleaned up test file")


def test_fallback_extraction():
    """Test that fallback extraction works when API is unavailable."""
    print("\n" + "="*70)
    print("TEST: Fallback Extraction (No API)")
    print("="*70)
    
    # Temporarily disable Gemini API
    original_key = settings.gemini_api_key
    settings.gemini_api_key = "dev_key_placeholder"
    
    test_file = Path("/tmp/test_fallback.pdf")
    test_file.write_text("Test PDF content")
    
    try:
        print(f"\n1️⃣  Testing fallback with {test_file.name}")
        
        extracted = gemini_service.analyze_evidence(
            file_path=str(test_file),
            mime_type="application/pdf"
        )
        
        print(f"\n2️⃣  Fallback Result:")
        print(f"   Extracted: {extracted}")
        
        if extracted and len(extracted) > 0:
            print(f"   ✅ Fallback returned non-empty text")
            return True
        else:
            print(f"   ❌ Fallback returned empty text")
            return False
            
    finally:
        settings.gemini_api_key = original_key
        if test_file.exists():
            test_file.unlink()


def test_mime_type_detection():
    """Test MIME type detection for various file types."""
    print("\n" + "="*70)
    print("TEST: MIME Type Detection & Validation")
    print("="*70)
    
    test_cases = [
        ("document.pdf", 1024*1024, "application/pdf", True),
        ("image.png", 5*1024*1024, "image/png", True),
        ("image.jpg", 3*1024*1024, "image/jpeg", True),
        ("file.exe", 1024, "application/x-msdownload", False),
        ("video.mp4", 100*1024*1024, "video/mp4", False),  # Too large
        ("document.doc", 1024*1024, "application/msword", True),  # Valid document type
    ]
    
    print(f"\nTesting {len(test_cases)} file types:\n")
    passed = 0
    
    for filename, size, mime_type, expected_valid in test_cases:
        is_valid, error = validation_service.validate_file_upload(
            filename=filename,
            file_size=size,
            mime_type=mime_type
        )
        
        status = "✅" if is_valid == expected_valid else "❌"
        result = "PASS" if is_valid == expected_valid else "FAIL"
        
        print(f"{status} {filename:20} | Size: {size/(1024*1024):6.1f}MB | {result}")
        if is_valid != expected_valid:
            print(f"   Expected: {expected_valid}, Got: {is_valid}")
            print(f"   Error: {error}")
        else:
            passed += 1
    
    print(f"\n{'─'*70}")
    print(f"MIME Type Tests: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def main():
    """Run all evidence extraction tests."""
    print("\n")
    print("┌" + "─"*68 + "┐")
    print("│" + " "*10 + "Evidence File Text Extraction - Test Suite" + " "*16 + "│")
    print("└" + "─"*68 + "┘")
    
    results = {
        'Evidence Extraction': test_evidence_extraction(),
        'Fallback Extraction': test_fallback_extraction(),
        'MIME Type Detection': test_mime_type_detection(),
    }
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n{'─'*70}")
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All evidence extraction tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
