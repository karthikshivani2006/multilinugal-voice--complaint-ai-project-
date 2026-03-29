#!/usr/bin/env python3
"""
Fix existing evidence records by adding fallback extracted text.
This updates evidence records that have empty extracted_text fields.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import SessionLocal
from models.evidence import Evidence

def fix_empty_evidence_text():
    """Update evidence records with empty extracted_text with fallback summaries."""
    db = SessionLocal()
    
    try:
        # Find all evidence with empty extracted_text
        empty_evidence = db.query(Evidence).filter(
            (Evidence.extracted_text == None) | (Evidence.extracted_text == "")
        ).all()
        
        print(f"Found {len(empty_evidence)} evidence records with empty extracted_text\n")
        
        updated_count = 0
        for ev in empty_evidence:
            # Create fallback summary from file info
            file_name = Path(ev.file_path).name
            file_size_mb = (Path(ev.file_path).stat().st_size / 1024 / 1024) if Path(ev.file_path).exists() else 0
            
            # Determine file type
            suffix = Path(ev.file_path).suffix.lower()
            file_type = {
                '.mp4': 'video',
                '.avi': 'video',
                '.mov': 'video',
                '.mkv': 'video',
                '.webp': 'image',
                '.jpg': 'image',
                '.jpeg': 'image',
                '.png': 'image',
                '.pdf': 'document',
                '.txt': 'document',
                '.doc': 'document',
                '.docx': 'document',
            }.get(suffix, 'file')
            
            # Create fallback text with useful information
            fallback_text = f"[{file_type.title()} Evidence: {file_name} ({file_size_mb:.1f}MB)]"
            
            ev.extracted_text = fallback_text
            db.add(ev)
            
            print(f"✓ Updated: {file_name}")
            print(f"  Fallback: {fallback_text}\n")
            
            updated_count += 1
        
        if updated_count > 0:
            db.commit()
            print(f"\n✅ Successfully updated {updated_count} evidence records!")
        else:
            print("✓ All evidence records already have extracted_text")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        sys.exit(1)
    
    finally:
        db.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Fixing Empty Evidence Text Records")
    print("=" * 60 + "\n")
    
    fix_empty_evidence_text()
    
    print("=" * 60)
    print("Complete!")
    print("=" * 60)
