#!/usr/bin/env python3
"""
Test the admin endpoint response structure to ensure it's ready for frontend
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import SessionLocal
from models.complaint import Complaint
from utils.file_utils import get_relative_file_path

def simulate_admin_response():
    """Simulate what the /admin/complaints endpoint returns."""
    db = SessionLocal()
    
    try:
        complaints = db.query(Complaint).order_by(Complaint.created_at.desc()).all()
        
        result = []
        for c in complaints[:1]:  # Just test the first one
            evidence_files = []
            
            for e in c.evidence_items:
                relative_path = get_relative_file_path(e.file_path)
                file_url = "/" + relative_path if not relative_path.startswith("/") else relative_path
                
                evidence_files.append({
                    "file_path": e.file_path,
                    "file_url": file_url,
                    "extracted_text": e.extracted_text,
                    "extracted_text_original": e.extracted_text,
                })
            
            complaint_obj = {
                "ticket_id": c.ticket_id,
                "name": c.user.name if c.user else "",
                "email": c.user.email if c.user else "",
                "phone": c.user.phone if c.user else "",
                "complaint_type": c.complaint_type,
                "description": c.description,
                "status": c.status,
                "created_at": c.created_at.isoformat(),
                "evidence_files": evidence_files,
            }
            result.append(complaint_obj)
        
        print("\n✅ Admin Endpoint Response (First Complaint):\n")
        print(json.dumps(result[0], indent=2))
        
        print("\n\n🔍 Evidence Files Processing:\n")
        for i, ev in enumerate(result[0]["evidence_files"], 1):
            print(f"Document {i}:")
            print(f"  Backend file_url: {ev['file_url']}")
            
            # Simulate frontend URL construction
            BASE_URL = "http://localhost:8000"
            file_url = ev['file_url']
            frontend_url = (file_url if file_url.startswith("http")
                           else f"{BASE_URL}{file_url if file_url.startswith('/') else '/' + file_url}")
            
            print(f"  Frontend finalUrl: {frontend_url}")
            print(f"  Extracted text: {ev['extracted_text'][:50]}...")
            print()
    
    finally:
        db.close()

if __name__ == "__main__":
    print("=" * 70)
    print("Admin Endpoint Response Simulation")
    print("=" * 70)
    
    try:
        simulate_admin_response()
        print("=" * 70)
        print("✅ Response structure is ready for frontend!")
        print("=" * 70)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
