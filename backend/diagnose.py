#!/usr/bin/env python3
"""
Diagnostic script to verify evidence extraction and file serving
"""
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import SessionLocal
from models.complaint import Complaint
from models.evidence import Evidence
from utils.file_utils import get_relative_file_path
from config import settings

def check_complaints_and_evidence():
    """Check database for complaints and evidence files."""
    db = SessionLocal()
    
    try:
        complaints = db.query(Complaint).order_by(Complaint.created_at.desc()).all()
        print(f"\n📊 Found {len(complaints)} complaints in database\n")
        
        if not complaints:
            print("⚠️  No complaints in database. Skipping evidence check.\n")
            return
        
        for c in complaints:
            print(f"🎫 Ticket: {c.ticket_id}")
            print(f"   Type: {c.complaint_type}")
            print(f"   Evidence items: {len(c.evidence_items)}")
            
            for i, ev in enumerate(c.evidence_items, 1):
                print(f"\n   📄 Evidence {i}:")
                print(f"      File path (stored): {ev.file_path}")
                
                # Convert to relative path
                rel_path = get_relative_file_path(ev.file_path)
                print(f"      Relative path: {rel_path}")
                
                # Build serving URL
                file_url = "/" + rel_path if not rel_path.startswith("/") else rel_path
                print(f"      Serving URL: {file_url}")
                
                # Check if file physically exists
                file_exists = Path(ev.file_path).exists()
                print(f"      File exists: {'✓' if file_exists else '✗'}")
                
                # Check extracted text
                text_length = len(ev.extracted_text or "")
                print(f"      Extracted text length: {text_length} chars")
                if text_length > 0 and text_length < 200:
                    print(f"      Extracted text: {ev.extracted_text}")
            print()
    
    finally:
        db.close()

def check_upload_directory():
    """Check if upload directory exists and is accessible."""
    print("\n📁 Upload Directory Check:")
    print(f"   Configured path: {settings.upload_dir}")
    
    upload_path = Path(settings.upload_dir)
    print(f"   Exists: {'✓' if upload_path.exists() else '✗'}")
    
    if upload_path.exists():
        print(f"   Is directory: {'✓' if upload_path.is_dir() else '✗'}")
        
        evidence_dir = upload_path / "evidence"
        id_dir = upload_path / "id_proofs"
        
        print(f"\n   Subdirectories:")
        print(f"      evidence/: {'✓' if evidence_dir.exists() else '✗'}")
        print(f"      id_proofs/: {'✓' if id_dir.exists() else '✗'}")
        
        if evidence_dir.exists():
            files = list(evidence_dir.glob("*"))
            print(f"      Files in evidence/: {len(files)}")
            for f in files[:5]:
                print(f"         - {f.name}")

def check_configuration():
    """Check critical configuration."""
    print("\n⚙️  Configuration Check:")
    print(f"   Gemini API Key: {'✓ Valid' if settings.gemini_api_key and settings.gemini_api_key != 'dev_key_placeholder' else '⚠️  Using placeholder'}")
    print(f"   Database URL: {settings.database_url}")
    print(f"   Upload directory: {settings.upload_dir}")

if __name__ == "__main__":
    print("=" * 60)
    print("CyberGuard Evidence & File Serving Diagnostics")
    print("=" * 60)
    
    try:
        check_configuration()
        check_upload_directory()
        check_complaints_and_evidence()
        
        print("\n" + "=" * 60)
        print("Diagnostics Complete")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
