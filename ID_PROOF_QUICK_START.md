# ID Proof Extraction - Quick Start Guide

## The Problem (Before Fix)

❌ When ID proof extraction wasn't clear → Complaint blocked  
❌ No way to proceed with manual details  
❌ User frustrated and unable to file complaint

## The Solution (After Fix)

✅ **Smart extraction** with automatic retry (3 attempts)  
✅ **Shows what was extracted** vs what's missing  
✅ **Users provide missing details manually**  
✅ **System merges both** and creates complaint successfully  
✅ **No blocking** - always allows user to proceed

---

## For Users - How to File Complaint Now

### Option 1: Clear ID Proof ✓
1. Upload your ID (Aadhaar, PAN, Driving License, Passport)
2. System extracts your name, phone, email automatically
3. Review pre-filled details ✓
4. Submit complaint directly
5. ✅ Done! Ticket ID: CG-2024-XXXXX

### Option 2: Blurry/Partial ID Proof 🔶
1. Upload your ID
2. System shows what it couldn't extract (e.g., "Email not visible")
3. You provide missing details manually
4. System merges: extracted data + your manual input
5. Submit complaint
6. ✅ Done! Ticket ID: CG-2024-XXXXX

### Option 3: Can't Read ID or No ID Available ❌
1. Skip ID proof upload
2. Manually enter your name, phone, email, address
3. Fill complaint details
4. Submit
5. ✅ Done! Ticket ID: CG-2024-XXXXX

---

## For Developers - What Changed

### Backend Updates

**File: `backend/services/gemini_service.py`**
- `analyze_id_proof()` now returns:
  - `extraction_status`: SUCCESS | PARTIAL_SUCCESS | UNCLEAR | ERROR
  - `missing_fields`: ['name', 'phone', 'email'] (what's missing)
  - `confidence`: HIGH | MEDIUM | LOW
  - 3-attempt retry with exponential backoff (1s, 2s, 4s)
  - Rate limit detection and auto-retry

**File: `backend/routes/complaint.py`**
- `validate_id_proof()` endpoint updated:
  - Returns detailed extraction status
  - Shows what was extracted vs what's missing
  - Always allows proceeding (`proceed_allowed: true`)
- New endpoint: `/complaint/create-with-partial-id`
  - Accepts both extracted and manual data
  - Merges intelligently (manual overrides extracted)
  - Creates complaint with combined data

**File: `backend/schemas/complaint.py`**
- New schema: `ComplaintCreateWithPartialIDRequest`
  - Fields for manual input: `full_name`, `phone_number`, `email`
  - Fields for extracted data: `extracted_name`, `extracted_phone`, `extracted_email`
  - System uses whichever is provided

### API Endpoints

**1. Validate ID Proof**
```
POST /complaint/validate-id
Input: ID proof file + optional user data
Output: extraction_status, missing_fields, extracted data
```

**2. Create Complaint (Partial ID)**
```
POST /complaint/create-with-partial-id
Input: complaint details + merged extracted+manual ID data
Output: ticket_id, status, created_at
```

---

## Technical Flow

```
┌────────────────────────────────────────┐
│ 1. User uploads ID proof              │
└────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────┐
│ 2. Backend attempts extraction         │
│    - Try 1: Analyze with Gemini        │
│    - Try 2: Wait 1s, retry             │
│    - Try 3: Wait 2s, retry             │
│    - Fallback: Return what extracted   │
└────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────┐
│ 3. Return extraction status            │
│    - SUCCESS: All fields found         │
│    - PARTIAL: Name ✓, Email ✗          │
│    - UNCLEAR: Nothing clear            │
└────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────┐
│ 4. User provides missing details       │
│    - Pre-filled fields: keep           │
│    - Missing fields: user enters       │
└────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────┐
│ 5. User submits complaint              │
│    - Payload includes both:            │
│      • Extracted data                  │
│      • User-entered data               │
└────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────┐
│ 6. Backend merges data                 │
│    - Manual overrides extracted        │
│    - If user provided name → use it    │
│    - Else use extracted name           │
└────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────┐
│ 7. Complaint created successfully ✓   │
│    - Ticket ID: CG-2024-XXXXX          │
└────────────────────────────────────────┘
```

---

## Key Features

| Feature | Before | After |
|---------|--------|-------|
| **Retry Logic** | ❌ None | ✅ 3 attempts, exponential backoff |
| **Rate Limiting** | ❌ Blocks on error | ✅ Auto-detects & retries |
| **Partial Extract** | ❌ Blocks complaint | ✅ Shows missing fields |
| **User Override** | ❌ Not possible | ✅ Can provide manually |
| **Data Merging** | ❌ N/A | ✅ Intelligently combines |
| **Error Messages** | ❌ Generic | ✅ Specific & actionable |
| **Extraction Status** | ❌ Binary | ✅ SUCCESS/PARTIAL/UNCLEAR/ERROR |

---

## API Quick Reference

### Validate ID Proof
```bash
curl -X POST http://localhost:8000/api/complaint/validate-id \
  -F "file=@aadhaar.jpg" \
  -F "full_name=John Doe"

# Returns:
# {
#   "extraction_status": "PARTIAL_SUCCESS",
#   "missing_fields": ["email"],
#   "extracted": { "name": "John", "phone": "9876543210", "email": "" }
# }
```

### Create Complaint with Partial ID
```bash
curl -X POST http://localhost:8000/api/complaint/create-with-partial-id \
  -H "Content-Type: application/json" \
  -d '{
    "complaint_type": "Phishing",
    "description": "Received phishing email",
    "full_name": "John Doe",
    "phone_number": "9876543210",
    "email": "john@example.com",
    "extracted_phone": "9876543210",
    "extracted_email": ""
  }'

# Returns:
# {
#   "ticket_id": "CG-2024-0012345",
#   "status": "pending",
#   "message": "Complaint created successfully..."
# }
```

---

## Frontend Implementation

**3-Step Component:**

```typescript
// 1. Upload ID Proof
<IDProofUpload onComplete={handleExtraction} />

// 2. Show Extraction Status
<IDProofExtractionStatus data={extractionResult} />

// 3. Auto-fill Form + Submit
<ComplaintFormWithIDProof initialData={mergedData} />
```

**User sees:**
- ✅ Green checkmark for extracted fields
- ⚠️ Yellow highlight for missing fields
- 📝 Input boxes for missing data
- ✅ Submit button (always enabled)

---

## Deployment Notes

### No Configuration Changes Required
- System automatically uses existing Gemini API key
- Falls back to Ollama if Gemini unavailable
- Retry logic built-in

### Logging
- All extraction attempts logged
- Rate limit events logged
- Track extraction success rates via logs

### Monitoring
Monitor these metrics:
- Extraction success rate: `extraction_status = "SUCCESS"` count
- Partial extraction rate: `extraction_status = "PARTIAL_SUCCESS"` count
- Failure rate: `extraction_status = "UNCLEAR"` count
- Rate limit hits: "Rate limited, waiting" log count

---

## Error Scenarios & Handling

### Scenario 1: Blurry ID
```
extraction_status: UNCLEAR
missing_fields: ["name", "phone", "email"]
↓
Show: "ID is not clear enough. Please enter details manually."
↓
User fills all fields manually
↓
Create complaint successfully
```

### Scenario 2: API Rate Limited
```
Error: 429 Too Many Requests
↓
System: Auto-retry in 1s, then 2s, then 4s
↓
If all fail: Return extraction_status = ERROR
↓
User: Can provide data manually
```

### Scenario 3: Bad File Upload
```
Error: File not readable / corrupted
↓
extraction_status: FILE_NOT_FOUND or ERROR
↓
Show: "ID proof couldn't be read. Please try a clearer image."
↓
User: Re-upload or enter manually
```

---

## Testing Checklist

- [ ] Upload clear ID → All fields extracted
- [ ] Upload blurry ID → Shows missing fields
- [ ] Upload damaged ID → Shows "UNCLEAR"
- [ ] User provides missing fields → Form pre-filled
- [ ] Submit with partial ID → Complaint created
- [ ] Submit without ID → Works fine
- [ ] Test with different document types (Aadhaar, PAN, DL, Passport)
- [ ] Test with non-Indian documents → Graceful fallback

---

## Support Docs

- **Full Fix Details:** `ID_PROOF_EXTRACTION_FIX.md`
- **Frontend Implementation:** `FRONTEND_ID_PROOF_INTEGRATION.md`
- **API Reference:** `API_REFERENCE_ID_PROOF.md`
- **Code Location:** 
  - Backend: `backend/services/gemini_service.py`, `backend/routes/complaint.py`
  - Frontend: `frontend/src/components/IDProofUpload.tsx`, `ComplaintFormWithIDProof.tsx`

---

## Summary

✅ **Issue Fixed:** ID proof extraction no longer blocks complaint submission  
✅ **User Experience:** Clear feedback + ability to provide missing details  
✅ **Backend:** Robust retry logic + intelligent data merging  
✅ **Frontend:** Auto-filling + missing field indicators  
✅ **Result:** Complaints always created, whether ID extraction succeeds or not

**Next Step:** Integrate frontend components and test with real users!
