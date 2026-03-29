# API Reference - ID Proof Extraction Endpoints

## Quick Reference

### Endpoint 1: Validate ID Proof

```
POST /complaint/validate-id
```

**Purpose:** Upload ID proof and get extraction status + missing fields

**Request:**
```
Content-Type: multipart/form-data

Parameters:
- file (required): ID proof image/PDF
  - Supported: PNG, JPG, PDF
  - Max size: 25MB
- full_name (optional): User's name for comparison
- phone_number (optional): User's phone for comparison  
- email (optional): User's email for comparison
- address (optional): User's address for comparison
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/complaint/validate-id \
  -F "file=@aadhaar.jpg" \
  -F "full_name=John Doe" \
  -F "phone_number=9876543210" \
  -F "email=john@example.com"
```

**Response (200 OK):**
```json
{
  "file_path": "uploads/id_proofs/abc123.jpg",
  "extraction_status": "PARTIAL_SUCCESS",
  "missing_fields": ["email"],
  "unclear_fields": ["email"],
  "extracted": {
    "name": "John Doe",
    "phone": "9876543210",
    "email": "",
    "address": "123 Main St, City",
    "document_type": "Aadhaar",
    "id_number": "1234 5678 9012"
  },
  "user_provided": {
    "name": "John Doe",
    "phone": "9876543210",
    "email": "john@example.com",
    "address": null
  },
  "analysis": {
    "document_type": "Aadhaar",
    "name": "John Doe",
    "id_number": "1234 5678 9012",
    "dob": "01-JAN-1990",
    "address": "123 Main St, City",
    "phone": "9876543210",
    "email": "",
    "confidence": "MEDIUM",
    "extraction_status": "PARTIAL_SUCCESS",
    "missing_fields": ["email"],
    "extracted_text": "[Parsed Aadhaar document with partial extraction]"
  },
  "mismatch_fields": [],
  "message": "ID proof processed with 1 unclear field. You can provide manual details for missing fields and proceed.",
  "proceed_allowed": true,
  "proceed_recommended": false
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `file_path` | string | Server path to uploaded ID proof |
| `extraction_status` | string | SUCCESS / PARTIAL_SUCCESS / UNCLEAR / ERROR / FILE_NOT_FOUND |
| `missing_fields` | array | Fields not extracted: name, phone, email, address |
| `unclear_fields` | array | Same as missing_fields (fields user needs to provide) |
| `extracted` | object | Extracted data with empty strings for missing fields |
| `extracted[name]` | string | Extracted name or empty |
| `extracted[phone]` | string | Extracted phone or empty |
| `extracted[email]` | string | Extracted email or empty |
| `extracted[address]` | string | Extracted address or empty |
| `extracted[document_type]` | string | ID type: Aadhaar, PAN, Driving License, Passport, etc. |
| `extracted[id_number]` | string | Extracted ID/doc number |
| `user_provided` | object | Data user provided in request for comparison |
| `mismatch_fields` | array | Fields that don't match user input (empty if match or user didn't provide) |
| `message` | string | Human-readable status message |
| `proceed_allowed` | boolean | Always true - system never blocks |
| `proceed_recommended` | boolean | true if high quality extraction, false if needs manual review |

**Error Responses:**

```bash
# 400: Invalid file format
{
  "detail": "File type not supported. Use PNG, JPG, or PDF"
}

# 422: Validation error
{
  "detail": "File size exceeds 25MB limit"
}

# 500: Server error
{
  "detail": "ID proof analysis failed. Please try again."
}
```

---

### Endpoint 2: Create Complaint with Partial ID Proof

```
POST /complaint/create-with-partial-id
```

**Purpose:** Submit complaint with merged extracted + manual ID proof data

**Request:**
```
Content-Type: application/json

Body: {
  "complaint_type": "Online Fraud",
  "description": "Received phishing email",
  "language": "English",
  
  // Manual user input (takes precedence over extracted)
  "full_name": "John Doe",
  "phone_number": "9876543210",
  "email": "john@example.com",
  "address": "123 Main St, City",
  
  // Extracted from ID proof (can be empty strings)
  "extracted_name": "",
  "extracted_phone": "9876543210",
  "extracted_email": "",
  "extracted_address": "123 Main St, City",
  "extracted_id_number": "1234 5678 9012",
  "extracted_document_type": "Aadhaar",
  
  // Optional complaint details
  "date_time": "2024-03-29T10:30:00",
  "amount_lost": "5000",
  "transaction_id": "TXN123456",
  "platform": "Email",
  "suspect_details": "Unknown sender",
  "suspect_vpa": "attacker@upi",
  "suspect_phone": "9999999999",
  "suspect_bank_account": "123456789012"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/complaint/create-with-partial-id \
  -H "Content-Type: application/json" \
  -d '{
    "complaint_type": "Online Fraud",
    "description": "Phishing attempt",
    "language": "English",
    "full_name": "John Doe",
    "phone_number": "9876543210",
    "extracted_name": "",
    "extracted_phone": "9876543210",
    "email": "john@example.com"
  }'
```

**Response (200 OK):**
```json
{
  "ticket_id": "CG-2024-0012345",
  "status": "pending",
  "created_at": "2024-03-29T10:30:00",
  "message": "Complaint created successfully with Ticket ID: CG-2024-0012345. Some details used from ID proof, others provided manually."
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | string | Unique complaint identifier (format: CG-YYYY-XXXXXX) |
| `status` | string | Current status: pending |
| `created_at` | string | ISO timestamp of complaint creation |
| `message` | string | Success message for user |

**Error Responses:**

```bash
# 422: Missing required fields
{
  "detail": "Name is required. Please provide manually if not extracted from ID proof."
}

# 422: Missing phone
{
  "detail": "Phone number is required. Please provide manually if not extracted from ID proof."
}

# 422: Invalid complaint type
{
  "detail": "Invalid complaint_type"
}

# 500: Server error
{
  "detail": "Failed to create complaint. Please try again."
}
```

---

## Data Merge Logic

When both extracted and manual data are provided, the system merges them as:

```
final_value = manual_value OR extracted_value OR ""

Examples:
- full_name: "John" OR "" → uses "John"
- full_name: "" OR "Jane" → uses "Jane"  
- full_name: "John" OR "Jane" → uses "John" (manual takes precedence)
```

**Field Priority:**
1. **Manual input takes precedence** - If user provides a value, it's used
2. **Extracted data as fallback** - If user doesn't provide, extracted is used
3. **Empty string if neither** - If both empty, field remains blank

---

## Field Requirements

### Required Fields
- `complaint_type` - Complaint category
- `description` - Detailed description
- `full_name` - Either from user input or ID extracted
- `phone_number` - Either from user input or ID extracted

### Optional Fields
- `email` - Can be from ID or user input
- `address` - Can be from ID or user input
- `language` - Default: "English"
- `date_time` - When complaint occurred
- `amount_lost` - Rupees (numeric)
- `transaction_id` - Transaction ID if applicable
- `platform` - Where fraud occurred (Email, UPI, Bank, etc.)
- `suspect_details` - Description of suspect
- `suspect_vpa` - Suspect's UPI/VPA
- `suspect_phone` - Suspect's phone number
- `suspect_bank_account` - Suspect's bank account

---

## Validation Rules

### Name Validation
- Minimum 3 characters
- No special characters
- Max 100 characters

### Phone Validation
- 10 digits (Indian format)
- Must start with 6-9

### Email Validation
- Valid RFC 5322 format
- Must contain @ and domain

### Complaint Type Validation
- Must be one of: Online Fraud, Banking Fraud, Phishing, E-Commerce Fraud, etc.

### Description Validation
- Minimum 10 characters
- No XSS/injection patterns

---

## Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid format) |
| 404 | Resource not found |
| 422 | Validation error (missing required fields, invalid values) |
| 500 | Server error |

---

## Extraction Status Values

### SUCCESS
- ✅ All key fields (name, phone, email) clearly extracted
- **Action:** User can submit directly
- **proceed_recommended:** true

### PARTIAL_SUCCESS
- ⚠️ Some fields extracted, others missing
- **Action:** User provides missing fields manually
- **proceed_recommended:** false
- **missing_fields:** Lists what's missing

### UNCLEAR
- ❌ ID proof too blurry/damaged to read clearly
- **Action:** User provides all details manually
- **proceed_recommended:** false
- **missing_fields:** ["name", "phone", "email"]

### ERROR
- 🔴 Processing failed (API error, rate limit, timeout)
- **Action:** Retry or provide manually
- **proceed_recommended:** false

### FILE_NOT_FOUND
- 📁 File path invalid or file deleted
- **Action:** Re-upload ID proof
- **proceed_recommended:** false

---

## Rate Limiting

- ID proof extraction is rate-limited at **5 requests per minute per IP**
- System automatically retries with exponential backoff
- Retry wait times: 1 second, 2 seconds, 4 seconds

---

## Timeout Handling

| Operation | Timeout |
|-----------|---------|
| File upload | 30 seconds |
| ID extraction | 60 seconds |
| Complaint creation | 30 seconds |

---

## Example Workflows

### Workflow 1: All Fields Extracted

```
1. POST /complaint/validate-id
   ↓ extraction_status: SUCCESS
2. Show success message
3. User clicks "Proceed"
4. POST /complaint/create-with-partial-id (no manual overrides needed)
   ↓ success with ticket_id
```

### Workflow 2: Some Fields Missing

```
1. POST /complaint/validate-id
   ↓ extraction_status: PARTIAL_SUCCESS
   ↓ missing_fields: ["email"]
2. Show "Email is missing" message
3. User fills in email field
4. POST /complaint/create-with-partial-id
   (extracted_phone="9876543210", email="john@ex.com" from user input)
   ↓ success with ticket_id
```

### Workflow 3: Extraction Failed

```
1. POST /complaint/validate-id
   ↓ extraction_status: UNCLEAR
   ↓ missing_fields: ["name", "phone", "email"]
2. Show "Please enter your details" message
3. User fills all required fields manually
4. POST /complaint/create-with-partial-id
   (All manual inputs, extracted fields empty)
   ↓ success with ticket_id
```

---

## Integration Checklist

- [ ] Frontend uploads to `/complaint/validate-id`
- [ ] Display extraction status and missing fields
- [ ] Pre-fill form with extracted data
- [ ] Allow user to override any extracted field
- [ ] Collect missing fields from user
- [ ] Submit to `/complaint/create-with-partial-id`
- [ ] Display success with ticket_id
- [ ] Handle errors with user-friendly messages

---

## Troubleshooting

### Issue: "Empty name after extraction"
- **Cause:** ID document didn't have readable name
- **Solution:** User provides name manually
- **Code:** 422 error → prompt for name input

### Issue: "Extraction taking too long"
- **Cause:** Large file or slow network
- **Solution:** Timeout after 60 seconds, allow manual entry
- **Code:** Automatic retry with exponential backoff

### Issue: "Extraction quota exceeded"  
- **Cause:** Too many requests to Gemini API
- **Solution:** 429 error triggers auto-retry with delay
- **Wait:** 1s → 2s → 4s (up to 3 attempts)

---

## Support

For issues with ID proof extraction:
1. Check extraction_status in response
2. Review missing_fields list
3. Provide missing data manually
4. If error persists, try different ID document type
5. Contact support with ticket_id if needed
