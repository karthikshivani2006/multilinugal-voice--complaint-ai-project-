#!/usr/bin/env python3
"""
Re-extract evidence for existing records using the real Gemini API
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import SessionLocal
from models.evidence import Evidence
from services.gemini_service import gemini_service
import mimetypes
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def re_extract_evidence():
    """Re-extract evidence using the configured Gemini API."""
    db = SessionLocal()
    
    try:
        # Find evidence records
        evidence_list = db.query(Evidence).all()
        
        print(f"\nFound {len(evidence_list)} evidence records to re-extract\n")
        
        for i, ev in enumerate(evidence_list, 1):
            print(f"Processing {i}/{len(evidence_list)}: {Path(ev.file_path).name}")
            
            file_path = ev.file_path
            if not Path(file_path).exists():
                print(f"  ⚠️  File not found: {file_path}")
                continue
            
            # Get MIME type
            mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            
            try:
                # Extract evidence using Gemini
                extracted_text = gemini_service.analyze_evidence(file_path, mime_type)
                
                if extracted_text:
                    old_len = len(ev.extracted_text or "")
                    new_len = len(extracted_text)
                    
                    ev.extracted_text = extracted_text
                    db.add(ev)
                    
                    print(f"  ✅ Extracted: {old_len} → {new_len} chars")
                    if new_len <= 150:
                        print(f"     Text: {extracted_text}")
                    else:
                        print(f"     Text: {extracted_text[:150]}...")
                else:
                    print(f"  ⚠️  No extraction result")
            
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            print()
        
        # Commit changes
        db.commit()
        print(f"✅ Successfully updated all evidence records!")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        sys.exit(1)
    
    finally:
        db.close()

if __name__ == "__main__":
    print("=" * 70)
    print("Re-extracting Evidence with Real Gemini API")
    print("=" * 70)
    
    re_extract_evidence()
    
    print("=" * 70)
    print("✅ Complete! Evidence extraction now uses real Gemini API")
    print("=" * 70)
