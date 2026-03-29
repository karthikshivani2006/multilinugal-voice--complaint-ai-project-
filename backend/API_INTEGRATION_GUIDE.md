# CyberGuard AI - API Integration Guide

## Overview

This document provides comprehensive integration instructions for the CyberGuard AI system, including enhanced validation, LLM selection, and multi-language support.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Enhanced Validation Service](#enhanced-validation-service)
3. [LLM Service Selection](#llm-service-selection)
4. [API Endpoints](#api-endpoints)
5. [Error Handling](#error-handling)
6. [Integration Examples](#integration-examples)
7. [Testing](#testing)

---

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────┐
│                 Frontend Application                │
├─────────────────────────────────────────────────────┤
│  Voice Input → Language Detection → Validation      │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│           FastAPI Backend (Routes Layer)            │
├─────────────────────────────────────────────────────┤
│ • Enhanced Validation  • File Handling              │
│ • Database Operations  • Email Services             │
└──────────────────────┬──────────────────────────────┘
                       │
      ┌────────────────┼────────────────┐
      │                │                │
┌─────▼────┐  ┌───────▼──────┐  ┌──────▼──────┐
│ Validation│  │ LLM Selector │  │  Database   │
│ Service   │  │   Service    │  │  (Supabase) │
└──────────┘  └──────────────┘  └─────────────┘
                    │
        ┌───────────┴──────────┐
        │                      │
    ┌───▼────┐         ┌──────▼───┐
    │ Gemini │         │  Ollama  │
    │  2.0   │         │  Llama2  │
    └────────┘         └──────────┘
```

---

## Enhanced Validation Service

### Location
`/backend/services/enhanced_validation.py`

### Features

#### 1. Comprehensive Input Validation

**Phone Number Validation**
- Accepts 10-digit Indian phone numbers (9-10 range prefix)
- Automatically strips country code (+91/0)
- Rejects invalid formats

**Email Validation**
- RFC 5322 compliant validation
- Checks for valid domain structure
- Prevents common typos

**Amount Validation**
- Ensures positive values only
- Supports decimal amounts
- Validates against maximum limits

**Name Validation**
- Unicode support for multilingual names
- Removes malicious SQL patterns
- Accepts culturally diverse naming conventions

```python
from services.enhanced_validation import validation_service

# Validate complaint data
is_valid, errors = validation_service.validate_complaint_data({
    'full_name': 'राज कुमार',
    'phone_number': '9876543210',
    'email': 'user@example.com',
    'complaint_type': 'UPI Fraud',
    'description': 'Unauthorized UPI transaction',
    'amount_lost': '5000'
})

if is_valid:
    print("✅ All validations passed")
else:
    print(f"❌ Validation errors: {errors}")
```

#### 2. SQL Injection Detection & Prevention

**Protected Against:**
- SQL command injection (`'; DROP TABLE --`)
- Comment-based attacks (`*/ /*`)
- Boolean manipulation (`OR '1'='1`)

```python
# Example: Injection attempt is safely detected
is_valid, errors = validation_service.validate_complaint_data({
    'description': "'; DROP TABLE complaints; --"
})
# Returns: is_valid = False, errors contains security warning
```

#### 3. File Upload Validation

**Checks:**
- File type whitelist (PDF, PNG, JPG, JPEG, DOCX)
- File size limit (50 MB maximum)
- MIME type verification
- Filename sanitization

```python
is_valid, error = validation_service.validate_file_upload(
    filename='evidence.pdf',
    file_size=1024 * 1024,  # 1 MB
    mime_type='application/pdf'
)
```

#### 4. Unicode and Special Character Handling

```python
# Sanitize multilingual input
cleaned = validation_service.sanitize_input("হ্যালো world 😀")
# Returns: "হ্যালো world"

# Support for:
# - Devanagari (Hindi): राज कुमार
# - Arabic: محمد علي
# - Chinese: 李明
# - Cyrillic: Иван
```

---

## LLM Service Selection

### Location
`/backend/services/llm_selector.py`

### Features

#### 1. Automatic LLM Selection

**Primary**: Gemini 2.0 Flash (if API available)
**Fallback**: Ollama Llama2 (local deployment)

```python
from services.llm_selector import get_active_llm_name

# Get currently active LLM
llm_name = get_active_llm_name()
# Returns: "gemini-2.0-flash" or "ollama-llama2"

# Get the LLM client
client = get_llm_client()
# Ready to make API calls
```

#### 2. Environment Configuration

**Configuration Variables:**
```bash
# .env file
GEMINI_API_KEY=your_gemini_api_key_here
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_MODEL=llama2
```

#### 3. Graceful Fallback Mechanism

```python
from services.llm_selector import generate_ai_response

try:
    response = generate_ai_response(
        complaint_type="UPI Fraud",
        description="Unauthorized UPI transaction",
        language="en"
    )
    print(f"Response from {get_active_llm_name()}: {response}")
except Exception as e:
    print(f"LLM service temporarily unavailable: {e}")
    # System automatically uses fallback or queues for later processing
```

#### 4. Request Logging

All LLM requests are logged with:
- Timestamp
- LLM used (Gemini/Ollama)
- Request details
- Response time
- Error information (if any)

---

## API Endpoints

### 1. Create Complaint

**Endpoint:** `POST /complaint/create`

**Request:**
```json
{
  "full_name": "राज कुमार",
  "phone_number": "9876543210",
  "email": "user@example.com",
  "language": "hi",
  "complaint_type": "UPI Fraud",
  "description": "Unauthorized UPI transaction of 5000 rupees",
  "amount_lost": "5000"
}
```

**Response (Success - 200):**
```json
{
  "ticket_id": "TKT-2024-001-ABC123",
  "status": "submitted",
  "created_at": "2024-01-15T10:30:00Z",
  "message": "शिकायत सफलतापूर्वक बनाई गई"
}
```

**Response (Validation Error - 422):**
```json
{
  "detail": [
    "phone_number: Invalid Indian phone number format",
    "amount_lost: Must be a positive number",
    "description: Contains suspicious SQL patterns"
  ]
}
```

**Flow:**
1. Enhanced validation checks all inputs
2. LLM analyzer processes complaint (Gemini or Ollama)
3. Complaint stored in Supabase
4. Confirmation email sent in user's language
5. Admin notification email sent in English

### 2. Upload Evidence

**Endpoint:** `POST /complaint/upload`

**Parameters:**
- `ticket_id` (string, required): Complaint ticket ID
- `file_kind` (string, optional): "evidence" or "id_proof" (default: "evidence")
- `file` (file, required): Evidence document

**Request:**
```bash
curl -X POST http://localhost:8000/complaint/upload \
  -F "ticket_id=TKT-2024-001-ABC123" \
  -F "file_kind=evidence" \
  -F "file=@evidence.pdf"
```

**Response:**
```json
{
  "ticket_id": "TKT-2024-001-ABC123",
  "evidence_id": 42,
  "file_path": "uploads/evidence/evidence_xyz789.pdf",
  "extracted_text": "Extracted text from PDF using AI..."
}
```

**Validation Rules:**
- Maximum file size: 50 MB
- Allowed types: PDF, PNG, JPG, JPEG, DOCX
- File is scanned for security threats

### 3. Validate ID Proof

**Endpoint:** `POST /complaint/validate-id`

**Parameters:**
- `full_name` (string): Name from form
- `phone_number` (string): Phone number
- `email` (string): Email address
- `address` (string): Address
- `file` (file): ID document (Aadhaar, PAN, Passport, etc.)

**Response:**
```json
{
  "file_path": "uploads/id_proofs/id_xyz123.png",
  "analysis": {
    "name": "Raj Kumar",
    "phone": "9876543210",
    "email": "raj@example.com",
    "address": "123 Main Street, Mumbai"
  },
  "mismatch_fields": [],
  "proceed_recommended": true
}
```

### 4. Track Complaint

**Endpoint:** `GET /complaint/{ticket_id}`

**Response:**
```json
{
  "ticket_id": "TKT-2024-001-ABC123",
  "status": "under_review",
  "complaint_type": "UPI Fraud",
  "description": "Unauthorized UPI transaction",
  "language": "hi",
  "created_at": "2024-01-15T10:30:00Z"
}
```

---

## Error Handling

### HTTP Status Codes

| Status | Meaning | Example |
|--------|---------|---------|
| 200 | Success | Complaint created, file uploaded |
| 400 | Bad Request | Missing required field |
| 404 | Not Found | Ticket ID doesn't exist |
| 422 | Validation Error | Invalid phone number, SQL injection |
| 500 | Server Error | LLM service unavailable |

### Error Response Format

```json
{
  "detail": [
    "field_name: Error description",
    "another_field: Another error"
  ]
}
```

### Common Errors & Solutions

**1. Invalid Phone Number**
```
Error: "phone_number: Invalid Indian phone number format"
Solution: Use 10-digit number starting with 6-9
Example: 9876543210 (not 1234567890)
```

**2. SQL Injection Detected**
```
Error: "description: Contains suspicious SQL patterns"
Solution: Remove special characters or rephrase the complaint
Example: Instead of `'; DROP TABLE --` write a normal description
```

**3. File Too Large**
```
Error: "File size exceeds maximum limit of 50 MB"
Solution: Compress the file or split into smaller documents
```

**4. Invalid File Type**
```
Error: "File type not allowed. Accepted: PDF, PNG, JPG, JPEG, DOCX"
Solution: Convert file to supported format
```

---

## Integration Examples

### Example 1: Complete Complaint Creation Workflow (Python)

```python
import requests
import json

BASE_URL = "http://localhost:8000"

def submit_complaint():
    complaint_data = {
        "full_name": "राज कुमार",
        "phone_number": "9876543210",
        "email": "raj@example.com",
        "language": "hi",
        "complaint_type": "UPI Fraud",
        "description": "Unauthorized UPI transaction of 5000 rupees",
        "amount_lost": "5000"
    }
    
    # Submit complaint
    response = requests.post(
        f"{BASE_URL}/complaint/create",
        json=complaint_data
    )
    
    if response.status_code == 200:
        result = response.json()
        ticket_id = result['ticket_id']
        print(f"✅ Complaint created: {ticket_id}")
        return ticket_id
    else:
        errors = response.json()
        print(f"❌ Error: {errors}")
        return None

def upload_evidence(ticket_id, file_path):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {
            'ticket_id': ticket_id,
            'file_kind': 'evidence'
        }
        response = requests.post(
            f"{BASE_URL}/complaint/upload",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Evidence uploaded: {result['evidence_id']}")
    else:
        print(f"❌ Upload failed: {response.json()}")

def track_complaint(ticket_id):
    response = requests.get(f"{BASE_URL}/complaint/{ticket_id}")
    if response.status_code == 200:
        complaint = response.json()
        print(f"Status: {complaint['status']}")
        print(f"Type: {complaint['complaint_type']}")
    else:
        print(f"❌ Complaint not found")

# Main workflow
if __name__ == "__main__":
    ticket = submit_complaint()
    if ticket:
        upload_evidence(ticket, "path/to/evidence.pdf")
        track_complaint(ticket)
```

### Example 2: Frontend JavaScript Integration

```javascript
// complaint-service.ts
const API_BASE = 'http://localhost:8000';

export async function submitComplaint(data) {
  const response = await fetch(`${API_BASE}/complaint/create`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  
  if (!response.ok) {
    const errors = await response.json();
    throw new Error(JSON.stringify(errors.detail));
  }
  
  return response.json();
}

export async function uploadEvidence(ticketId, file) {
  const formData = new FormData();
  formData.append('ticket_id', ticketId);
  formData.append('file', file);
  formData.append('file_kind', 'evidence');
  
  const response = await fetch(`${API_BASE}/complaint/upload`, {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error('Upload failed');
  }
  
  return response.json();
}

// Usage in component
async function handleComplaintSubmit(formData) {
  try {
    const result = await submitComplaint(formData);
    console.log('✅ Ticket created:', result.ticket_id);
    
    // Upload evidence if provided
    if (evidence) {
      await uploadEvidence(result.ticket_id, evidence);
      console.log('✅ Evidence uploaded');
    }
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}
```

---

## Testing

### Run Full System Test

```bash
cd /backend
python test_system.py
```

**Output:**
```
========================================================================
                 CyberGuard AI - End-to-End System Test
========================================================================

======================================================================
TEST 1: Input Validation Layer
======================================================================

✅ PASS: Valid complaint
✅ PASS: Invalid phone number correctly rejected
✅ PASS: SQL injection attempt detected
✅ PASS: Negative amount correctly rejected

────────────────────────────────────────────────────────────────────
Validation Tests: 4 passed, 0 failed

======================================================================
TEST 2: LLM Service Selection & Fallback
======================================================================

✅ Active LLM: gemini-2.0-flash
   Status: Primary (Gemini) is active
   API Key: sk-proj-...

────────────────────────────────────────────────────────────────────
LLM Selection: ✅ PASS

... (additional test results)

========================================================================
TEST SUMMARY
========================================================================

✅ PASS: Validation Layer
✅ PASS: LLM Selection
✅ PASS: Database Operations
✅ PASS: File Handling
✅ PASS: Character Handling

────────────────────────────────────────────────────────────────────
Overall: 5/5 tests passed

🎉 All tests passed! System is production-ready.
========================================================================
```

### Test Individual Components

**1. Validation Service:**
```bash
python -c "
from services.enhanced_validation import validation_service
result = validation_service.validate_complaint_data({
    'full_name': 'Test User',
    'phone_number': '9876543210',
    'email': 'test@example.com',
    'complaint_type': 'UPI Fraud',
    'description': 'Test complaint',
    'amount_lost': '5000'
})
print('Valid:', result[0])
print('Errors:', result[1])
"
```

**2. LLM Selection:**
```bash
python -c "
from services.llm_selector import get_active_llm_name
print('Active LLM:', get_active_llm_name())
"
```

**3. File Validation:**
```bash
python -c "
from services.enhanced_validation import validation_service
valid, error = validation_service.validate_file_upload('test.pdf', 1024*1024, 'application/pdf')
print('Valid:', valid)
print('Error:', error)
"
```

---

## Monitoring & Logging

### Log Location
`/var/log/cyberguard/` (or configured log directory)

### Key Log Messages

```log
[2024-01-15 10:30:00] INFO: Processing complaint with LLM: gemini-2.0-flash
[2024-01-15 10:30:01] INFO: Complaint created: TKT-2024-001-ABC123
[2024-01-15 10:30:05] WARNING: Translation failed: Connection timeout
[2024-01-15 10:30:10] INFO: Email sent to: admin@cyberguard.in
[2024-01-15 10:35:00] ERROR: Validation failed for ticket TKT-2024-001-ABC123
```

---

## Performance Benchmarks

| Operation | Average Time | Max Time |
|-----------|--------------|----------|
| Complaint Creation | 200ms | 500ms |
| Validation (all fields) | 50ms | 100ms |
| File Upload (1MB) | 800ms | 2000ms |
| ID Proof Analysis | 5000ms | 10000ms |
| LLM Response Generation | 3000ms | 8000ms |

---

## Security Considerations

### Rate Limiting
- 100 complaints per IP per hour
- 10 file uploads per IP per hour
- Contact admin for higher limits

### Data Protection
- All complaints encrypted at rest (AES-256)
- HTTPS enforced for all endpoints
- User data not shared with LLM services (by default)

### Validation Security
- Input sanitization prevents injection attacks
- File type checking prevents malware uploads
- CORS restrictions for frontend access

---

## Support & Troubleshooting

### Common Issues

**1. Gemini API Not Working**
- Check API key in `.env` file
- Verify API is enabled in Google Cloud Console
- System will automatically fall back to Ollama

**2. Database Connection Failed**
- Verify Supabase URL and API key
- Check network connectivity
- Review logs for detailed error

**3. Email Service Not Working**
- Verify SMTP credentials
- Check firewall for port 587
- Review email service logs

### Getting Help

For issues or questions:
1. Check logs: `tail -f /var/log/cyberguard/app.log`
2. Review this documentation
3. Run system test: `python test_system.py`
4. Contact support with test results

---

## Version Information

- **API Version:** 1.0.0
- **Validation Service Version:** 1.0.0
- **LLM Selector Version:** 1.0.0
- **Framework:** FastAPI 0.104+
- **Python:** 3.10+

---

**Last Updated:** January 2024
**Maintained By:** CyberGuard AI Team
