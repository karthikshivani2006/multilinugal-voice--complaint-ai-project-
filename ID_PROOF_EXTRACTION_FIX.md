# ID Proof Extraction - Partial Data Handling Fix

## Problem Statement

Previously, when ID proof (Aadhaar, PAN, Driving License, etc.) text extraction wasn't clear enough to extract name/phone/email, the system would block complaint submission with an error message:

> "Could not clearly extract name/phone/email from the ID proof"

This prevented users from proceeding even though they had other complaint details captured.

## Solution Overview

The system now intelligently handles **partial/unclear ID proof extraction** by:

1. ✅ **Attempting extraction** using Gemini AI with 3-retry logic
2. ✅ **Reporting what was extracted** vs what's missing/unclear
3. ✅ **Allowing user override** - users can provide missing details manually
4. ✅ **Merging data intelligently** - combining extracted data with manual input
5. ✅ **Enabling complaint submission** even with incomplete extraction

## New Extraction Status Levels

The updated `analyze_id_proof()` endpoint now returns detailed extraction status:

### Response Fields

```json
{
  "extraction_status": "PARTIAL_SUCCESS | SUCCESS | UNCLEAR | ERROR | FILE_NOT_FOUND",
  "missing_fields": ["name", "phone", "email"],
  "confidence": "HIGH | MEDIUM | LOW",
  "extracted": {
    "name": "Extracted from ID or empty",
    "phone": "Extracted from ID or empty",
    "email": "Extracted from ID or empty",
    "address": "Extracted from ID or empty",
    "document_type": "Aadhaar | PAN | etc.",
    "id_number": "ID number if visible"
  },
  "message": "ID proof processed with 3 unclear fields. You can provide manual details..."
}
```

### Status Explanations

| Status | Meaning | Action |
|--------|---------|--------|
| **SUCCESS** | All fields (name, phone, email) extracted clearly | Proceed directly |
| **PARTIAL_SUCCESS** | Some fields extracted, others missing | Provide missing fields manually |
| **UNCLEAR** | ID proof image not clear enough | Provide all details manually |
| **ERROR** | Processing failed (API error, etc.) | Retry or provide manually |
| **FILE_NOT_FOUND** | File location invalid | Re-upload ID proof |

## API Endpoints

### 1. Validate ID Proof Extraction

**Endpoint:** `POST /complaint/validate-id`

**Request:**
```form-data
full_name: (optional) User's name if already known
phone_number: (optional) User's phone if already known
email: (optional) User's email if already known
address: (optional) User's address if already known
file: (required) ID proof image/PDF
```

**Response:**
```json
{
  "file_path": "uploads/id_proofs/abc123.jpg",
  "extraction_status": "PARTIAL_SUCCESS",
  "missing_fields": ["name", "email"],
  "unclear_fields": ["name", "email"],
  "extracted": {
    "name": "",
    "phone": "9876543210",
    "email": "",
    "address": "123 Main St, City",
    "document_type": "Driving License",
    "id_number": "DL-2024-123456"
  },
  "user_provided": {
    "name": "John Doe",
    "phone": "9876543210"
  },
  "message": "ID proof processed with 2 unclear fields. You can provide manual details...",
  "proceed_allowed": true,
  "proceed_recommended": false
}
```

### 2. Create Complaint with Partial ID Proof

**NEW ENDPOINT:** `POST /complaint/create-with-partial-id`

**Purpose:** Submit complaint when ID proof extraction is partial. System merges extracted data with manual input.

**Request:**
```json
{
  "complaint_type": "Online Fraud",
  "description": "Received phishing email",
  "language": "English",
  
  "full_name": "John Doe",
  "phone_number": "9876543210",
  "email": "john@example.com",
  "address": "123 Main St",
  
  "extracted_name": "",
  "extracted_phone": "9876543210",
  "extracted_email": "",
  "extracted_address": "123 Main St",
  "extracted_id_number": "DL-2024-123456",
  "extracted_document_type": "Driving License",
  
  "date_time": "2024-03-29T10:30:00",
  "amount_lost": "5000",
  "platform": "Email"
}
```

**Response:**
```json
{
  "ticket_id": "CG-2024-001234",
  "status": "pending",
  "created_at": "2024-03-29T10:30:00",
  "message": "Complaint created successfully with Ticket ID: CG-2024-001234. Some details used from ID proof, others provided manually."
}
```

### 3. Regular Complaint Creation (Still Available)

**Endpoint:** `POST /complaint/create`

**Use when:** All details (name, phone, email) are available and ID proof validation not needed.

## User Workflow

### Scenario 1: ID Proof Has Clear Details ✅

1. User uploads ID proof
2. System extracts: name ✓, phone ✓, email ✓
3. Response: `proceed_recommended: true`
4. User submits complaint immediately
5. ✅ Complaint created

### Scenario 2: ID Proof is Blurry/Partial 🔶

1. User uploads ID proof
2. System extracts: name ✗, phone ✓, email ✗
3. Response shows: `missing_fields: ["name", "email"]`
4. **User provides missing details manually**
5. User submits with: extracted (phone) + manual (name, email)
6. ✅ Complaint created successfully

### Scenario 3: ID Proof Unreadable ❌

1. User uploads ID proof (very blurry/damaged)
2. System extracts: nothing clearly
3. Response: `extraction_status: UNCLEAR`, `missing_fields: ["name", "phone", "email"]`
4. **User provides all details manually** (since none extracted)
5. User submits: all manual details
6. ✅ Complaint created successfully

## Frontend Integration

### Step 1: Upload and Validate ID Proof

```javascript
// Step 1: Upload ID proof for validation
const validateIDProof = async (formData) => {
  const response = await fetch('/api/complaint/validate-id', {
    method: 'POST',
    body: formData  // Must include: file, optional: full_name, phone_number, email
  });
  
  const result = await response.json();
  return result;
  // Returns: {
  //   extraction_status, missing_fields, extracted, extracted_text,
  //   proceed_allowed, proceed_recommended, message
  // }
};

// Step 2: Show user what was extracted
const idValidationResult = await validateIDProof(uploadFormData);

if (idValidationResult.extraction_status === 'SUCCESS') {
  // All fields extracted - proceed directly
  console.log("✅ ID proof validated successfully");
  return showSuccessMessage();
}

if (idValidationResult.proceed_allowed) {
  // Partial extraction - show user what's missing
  const missingFields = idValidationResult.missing_fields;
  const extractedData = idValidationResult.extracted;
  
  console.log(`ℹ️  Missing fields: ${missingFields.join(', ')}`);
  console.log(`Extracted: name="${extractedData.name}", phone="${extractedData.phone}"`);
  
  // Show form with pre-filled extracted data
  showComplaintFormWithExtractedData(extractedData);
  
  // Highlight missing fields for user to fill
  missingFields.forEach(field => {
    markFieldAsRequired(field);
  });
}
```

### Step 2: Submit Complaint with Partial ID Data

```javascript
// Step 3: User fills missing fields and submits
const submitComplaintWithPartialID = async (formData) => {
  // Prepare payload: merge extracted + manual fields
  const payload = {
    complaint_type: formData.complaintType,
    description: formData.description,
    language: formData.language,
    
    // Manual user input (overrides extracted if both provided)
    full_name: formData.manualName || "",
    phone_number: formData.manualPhone || "",
    email: formData.manualEmail || "",
    address: formData.manualAddress || "",
    
    // Extracted data from ID proof
    extracted_name: formData.extractedName || "",
    extracted_phone: formData.extractedPhone || "",
    extracted_email: formData.extractedEmail || "",
    extracted_address: formData.extractedAddress || "",
    extracted_id_number: formData.extractedIDNumber || "",
    extracted_document_type: formData.extractedDocType || "",
    
    // Other complaint details
    date_time: formData.dateTime,
    amount_lost: formData.amountLost,
    platform: formData.platform
  };
  
  const response = await fetch('/api/complaint/create-with-partial-id', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  
  const result = await response.json();
  
  if (response.ok) {
    console.log(`✅ Complaint created: ${result.ticket_id}`);
    return result;
  } else {
    console.error(`❌ Error: ${result.detail}`);
    throw new Error(result.detail);
  }
};
```

## Technical Implementation Details

### Backend Changes

#### 1. Enhanced `analyze_id_proof()` 

**File:** `backend/services/gemini_service.py`

**Changes:**
- ✅ Implemented 3-attempt retry logic with exponential backoff
- ✅ Rate limit detection and automatic retry
- ✅ Added `extraction_status` field (SUCCESS, PARTIAL_SUCCESS, UNCLEAR, ERROR)
- ✅ Added `missing_fields` array listing clearly-not-extracted fields
- ✅ Added `confidence` level (HIGH, MEDIUM, LOW)
- ✅ Detailed logging at each extraction step

**Retry Logic:**
```python
for attempt in range(3):  # 3 attempts
    try:
        # Attempt extraction
        if success:
            return {extraction_status: "SUCCESS", missing_fields: [...]}
    except RateLimitError:
        if attempt < 2:
            time.sleep(2 ** attempt)  # Wait 1s, 2s, 4s
            continue
```

#### 2. Updated `validate_id_proof()` Route

**File:** `backend/routes/complaint.py`

**Changes:**
- ✅ Returns detailed extraction status and missing fields
- ✅ Separates "mismatch_fields" (doesn't match user input) from "missing_fields" (not extracted)
- ✅ Always sets `proceed_allowed: true` (system never blocks)
- ✅ Sets `proceed_recommended: true/false` based on extraction quality
- ✅ Returns both extracted and user-provided data for comparison

#### 3. New `create-with-partial-id` Endpoint

**File:** `backend/routes/complaint.py`

**Functionality:**
- ✅ Accepts both extracted and manual user data
- ✅ Merges intelligently: manual overrides extracted if both provided
- ✅ Validates minimum required fields (name, phone)
- ✅ Creates complaint with merged data
- ✅ Returns success with note about data source

#### 4. New Schema `ComplaintCreateWithPartialIDRequest`

**File:** `backend/schemas/complaint.py`

**Fields:**
- Required: `complaint_type`, `description`
- User input (manual): `full_name`, `phone_number`, `email`, `address`
- Extracted data: `extracted_name`, `extracted_phone`, `extracted_email`, etc.
- Optional: other complaint details

## Error Handling

### Minimum Required Fields

- **Name:** Must be provided (either extracted or manual)
- **Phone:** Must be provided (either extracted or manual)
- **Email:** Optional (can be provided or left empty)

### Validation Errors

```json
{
  "status": 422,
  "detail": "Name is required. Please provide manually if not extracted from ID proof."
}
```

## Testing

### Test Case 1: Full Extraction

```bash
# Upload clear ID proof with all details visible
# Expected: extraction_status = "SUCCESS", missing_fields = []

curl -X POST /complaint/validate-id \
  -F "file=@clear_aadhaar.jpg"

# Response: All fields extracted ✓
```

### Test Case 2: Partial Extraction

```bash
# Upload slightly blurry ID proof
# Expected: extraction_status = "PARTIAL_SUCCESS", missing_fields = ["email"]

curl -X POST /complaint/validate-id \
  -F "file=@blurry_dl.jpg" \
  -F "full_name=John Doe" \
  -F "phone_number=9876543210"

# Then submit with manual email
curl -X POST /complaint/create-with-partial-id \
  -H "Content-Type: application/json" \
  -d '{
    "complaint_type": "Fraud",
    "description": "...",
    "full_name": "John Doe",
    "phone_number": "9876543210",
    "email": "john@email.com",
    "extracted_name": "",
    "extracted_phone": "9876543210"
  }'
```

### Test Case 3: Unclear Extraction

```bash
# Upload very blurry/damaged ID proof
# Expected: extraction_status = "UNCLEAR", missing_fields = ["name", "phone", "email"]

curl -X POST /complaint/validate-id \
  -F "file=@damaged_passport.jpg"

# Response: No fields extracted clearly
# User must provide all details manually
```

## Configuration

No additional configuration needed. The system automatically:
- Uses Gemini 2.0 Flash for extraction with fallback to Gemini 1.5 Flash
- Implements retry logic with exponential backoff
- Detects rate limits and waits appropriately
- Logs all extraction attempts for debugging

## Troubleshooting

### Issue: "Empty field still being returned"

**Solution:** System now guarantees non-empty `extracted_text` field for every response:
- If extraction fails: `[ID Proof: filename.jpg]`
- If extraction partial: returns what was extracted + empty string for missing
- Always has at least a fallback value

### Issue: "User can't proceed with unclear extraction"

**Solution:** 
1. Use new `/create-with-partial-id` endpoint
2. Pass `extracted_*` fields as empty strings
3. Provide manual user data in regular fields
4. System merges and creates complaint successfully

### Issue: "Rate limit errors blocking submission"

**Solution:** 
- Implemented automatic retry with exponential backoff
- Up to 3 attempts with 1s, 2s, 4s waits
- System logs all rate limit events for monitoring

## Related Files

- `backend/services/gemini_service.py` - ID proof extraction logic
- `backend/routes/complaint.py` - ID proof validation and complaint creation endpoints
- `backend/schemas/complaint.py` - Data models
- Frontend components: ID proof upload and complaint form

## Summary

The system now **gracefully handles ID proof extraction failures** by:

1. ✅ Showing users exactly what was / wasn't extracted
2. ✅ Allowing manual entry of missing fields
3. ✅ Merging extracted + manual data intelligently
4. ✅ Creating complaints even with partial ID data
5. ✅ Never blocking complaint submission on poor extraction

Users can now proceed confidently even when ID proof extraction is unclear!
