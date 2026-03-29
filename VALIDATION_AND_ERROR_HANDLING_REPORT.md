# Validation and Error Handling Audit Report

**Generated:** March 29, 2026
**Scope:** Backend (FastAPI) + Frontend (React TypeScript)

---

## Executive Summary

The application has **moderate validation coverage** with **several critical gaps** and **error handling inconsistencies**. Key issues include:
- Missing input length/boundary validations
- Insufficient error boundaries in the React frontend
- Silent failures in microservices (Gemini, translation services)
- Race conditions in file uploads
- Incomplete database constraints
- No explicit error recovery mechanisms

---

## 1. CURRENT VALIDATIONS IN PLACE

### 1.1 Backend Validations (✅ Implemented)

#### Phone Number Validation
- **Location:** [backend/services/validation_service.py](backend/services/validation_service.py)
- **Pattern:** `^[6-9]\d{9}$` (10-digit Indian mobile)
- **Handles:** Country code stripping (91), non-digit removal
- **Coverage:** Used in complaint creation, chat flow, ID proof validation

```python
# Regex: ^[6-9]\d{9}$
# Accepts 10-digit numbers starting with 6-9
```

#### Email Validation
- **Location:** [backend/services/validation_service.py](backend/services/validation_service.py)
- **Method:** Uses `email_validator` library with fallback regex
- **Special Cases:** Accepts "N/A", "na", "none", "skip" as valid null values
- **Pattern:** `^[^@\s]+@[^@\s]+\.[^@\s]+$`

#### UTR/Transaction ID Validation
- **Pattern:** `^[A-Za-z0-9\-]{8,30}$`
- **Length:** 8-30 characters
- **Accepts null markers:** "N/A", "na", "none", "skip"

#### Complaint Type Validation
- **Location:** [backend/utils/constants.py](backend/utils/constants.py)
- **Method:** Enum check against `COMPLAINT_TYPES` list
- **Coverage:** 25+ predefined complaint types

#### Required Fields Validation
- **Location:** [backend/routes/complaint.py](backend/routes/complaint.py#L28)
- **Required Fields:** `full_name`, `phone_number`, `complaint_type`, `description`
- **Error Code:** HTTP 422 (Unprocessable Entity)
- **Response Format:** Array of error strings

### 1.2 Chat Flow Validations (✅ Implemented)

**Location:** [backend/routes/chat.py](backend/routes/chat.py)

#### Field-Specific Validators
- **full_name:** Min 3 chars, no digits, no complaint keywords (file/register/submit)
- **phone_number:** 10-digit validation
- **email:** RFC compliance or simple email regex
- **address:** Min 6 chars, at least 2 words
- **description:** Min 10 chars, at least 3 words
- **platform:** Min 2 chars
- **transaction_id:** UTR format validation
- **date_time:** ISO format or contains date indicators

#### Noise Detection
- Repeated characters (5+ of same char flagged as noise)
- Symbol-only inputs rejected
- Empty after normalization flagged

#### Optional Field Markers
Gracefully accepts: "N/A", "na", "none", "skip", "unknown" for optional fields

### 1.3 Frontend Validations (⚠️ Partial)

**Location:** [frontend/src/pages/ComplaintPage.tsx](frontend/src/pages/ComplaintPage.tsx)

#### Phone Validation
- Pattern: `^[6-9]\d{9}$` (pre-submission only)
- Normalizes: Removes non-digits, takes last 10

#### Email Validation
- Pattern: `^[^@\s]+@[^@\s]+\.[^@\s]+$` (pre-submission only)
- Accepts "N/A" as valid

#### Form Completeness Check
- Required fields: name, phone, email, address, type, date, description
- Financial complaint types trigger additional fields (amount, transaction ID, suspect details)

#### ID Proof Validation
- Mismatch detection against extracted data from uploaded ID
- Normalized comparison: name, phone, email, address
- User prompt for mismatch confirmation

---

## 2. GAPS & MISSING VALIDATIONS

### 🔴 Critical Gaps

#### 2.1 String Length Boundaries Missing
| Field | Current | Recommended |
|-------|---------|------------|
| `full_name` | Min 3 chars only | Add max 255 chars |
| `phone` | 10 digits exact | ✅ Correct |
| `email` | No max length | Add max 255 chars |
| `address` | Min 6 chars | Add max 500 chars |
| `description` | Min 10 chars | Add max 5000 chars |
| `complaint_type` | Enum checked | ✅ Correct |
| `amount_lost` | No validation | Add numeric + max allowed |
| `transaction_id` | 8-30 chars | ✅ Correct |
| `platform` | Min 2 chars | Add max 100 chars |
| `suspect_details` | No validation | Add max 1000 chars |
| `suspect_vpa` | No validation | Add UPI format validation |
| `suspect_phone` | No validation | Add phone format validation |
| `suspect_bank_account` | No validation | Add IFSC + account format |

**Impact:** DB overflow attacks, data truncation issues

#### 2.2 Numeric Field Validation Missing
- `amount_lost`: No validation that it's numeric or within reasonable bounds
  - Could accept: "infinite", "999999999999", special characters
  - No range validation (max loss amount)
- `transaction_id`: Treated as alphanumeric string, no format-specific validation per complaint type

**Impact:** Garbage data in database, invalid UTRs, integer overflow

#### 2.3 Input Sanitization Gaps
- XSS Protection: No explicit HTML escaping in React before display
- No input sanitization for special characters that could break queries
- No rate limiting on API endpoints

**Current State:** Frontend relies on React's default escaping, backend doesn't validate

#### 2.4 File Upload Validation Gaps
**Location:** [backend/routes/complaint.py](backend/routes/complaint.py#L116-L145)

Missing validations:
- ❌ No file size limits
- ❌ No file type validation (MIME type checking insufficient)
- ❌ No file extension whitelist
- ❌ No duplicate file detection
- ❌ No file integrity checks (magic bytes)
- ❌ No virus/malware scanning integration

**Code:**
```python
mime_type = file.content_type or mimetypes.guess_type(file.filename or "")[0]
# Content-Type header can be spoofed; no validation of actual file content
```

#### 2.5 Gender/Identification Data Validation Missing
- User model incomplete: No `gender`, `dob`, `id_proof_type` fields
- ID proof extraction: No validation of extracted data credibility
- No confidence score threshold enforcement

#### 2.6 Language Code Validation
- Backend accepts any string as language parameter
- No validation against `SUPPORTED_LANGUAGES` list

**Current:** [backend/routes/chat.py#L512](backend/routes/chat.py#L512) does validate, but only for chat endpoint

#### 2.7 SQL Injection Prevention (⚠️ Depends on ORM)
- Using SQLAlchemy ORM (good), but:
- No explicit parameterized query examples in custom filtering
- Assumes ORM protects all cases

#### 2.8 Authentication/Authorization Gaps
- No CSRF protection visible
- JWT token expiry: 60 minutes (hardcoded) - not configurable
- No token refresh mechanism
- Admin role hardcoded (no fine-grained permissions)

---

### 🟡 Medium Priority Gaps

#### 2.9 Date/Time Field Validation
- Accepts any ISO format or "natural language" with digits
- No bounds checking (doesn't reject dates far in past/future)
- Potential for: "2099-12-31", "1900-01-01", "9999-12-31"

#### 2.10 Complaint Type Keyword Matching
**Location:** [backend/routes/chat.py](backend/routes/chat.py)

Missing normalization logic:
- Typos not handled (e.g., "UPI Fraud" vs "UPI  Fraud" vs "upi fraud")
- Multilingual keyword matching only partially implemented
- No fuzzy matching for similar complaint types

#### 2.11 Form State Race Conditions
**Location:** [frontend/src/pages/ComplaintPage.tsx](frontend/src/pages/ComplaintPage.tsx)

Potential issues:
```tsx
// Multiple setState calls without proper sequencing
setForm(...); // Updates form
handleSubmit(); // Could trigger before form updates are batched
```

#### 2.12 Conversation History Limits
- No max conversation length limit
- Could lead to unbounded storage/memory issues
- No retention policy defined

#### 2.13 Evidence File Extraction Fallback
**Location:** [backend/routes/complaint.py#L126-L128](backend/routes/complaint.py#L126-L128)

```python
try:
    extracted_text = gemini_service.analyze_evidence(...)
except Exception as exc:
    logger.warning(...)
    # Continues with empty extracted_text - no retry or user notification
```

---

## 3. ERROR HANDLING ISSUES

### 3.1 Inconsistent Error Responses

#### Backend Error Handling Patterns ⚠️

| Endpoint | Error Code | Response Format | Issue |
|----------|-----------|-----------------|-------|
| POST /complaint/create | 422 | Array of strings | ✅ Consistent |
| GET /admin/complaints | 500 (implied) | No error schema | ❌ No specification |
| PUT /admin/update-status | 404 | String detail | ⚠️ Inconsistent format |
| POST /complaint/validate-id | 500 (implied) | Generic log warning | ❌ No user feedback |
| POST /ai/speech-to-text | 400 | String detail | ✅ Consistent |

#### Frontend Error Handling Patterns ⚠️

```tsx
// Pattern 1: Error caught and displayed
catch (err) {
  const raw = err instanceof Error ? err.message : t("failedSubmit");
  setSubmitError(message);
  speak(guidance, language.speechCode); // Speaks error message
}

// Issues:
// - No retry logic
// - No error categorization
// - Silent failures in validateIdProof
// - No error tracking/logging
```

### 3.2 Service Integration Errors (Silent Failures)

#### Gemini Service Failures
**Location:** [backend/routes/complaint.py#L73-L75](backend/routes/complaint.py#L73-L75)

```python
analysis = {}
try:
    analysis = gemini_service.analyze_id_proof(...)
except Exception as exc:
    logger.warning("ID validation failed...")
    # Returns empty analysis - no user indication of failure
```

**Impact:** ID proof validation silently fails; form proceeds with incomplete data

#### Translation Service Failures
**Location:** [backend/routes/complaint.py#L39-L50](backend/routes/complaint.py#L39-L50)

```python
try:
    localized_summary = translation_service.translate(...)
except Exception as exc:
    logger.warning("Translation failed...")
    # Falls back to original language - could be unexpected
```

**Impact:** User receives summary in wrong language without notification

#### LLM (Ollama/Gemini) Failures in Chat
**Location:** [backend/routes/chat.py#L546-L580](backend/routes/chat.py#L546-L580)

```python
try:
    llm_payload = await asyncio.wait_for(ollama_service.generate_complaint_chat_reply(...), timeout=30)
except Exception as e:
    # Falls back to Gemini
    # If both fail: no error response to user
```

### 3.3 Database Constraint Gaps

**Current Model:** [backend/models/complaint.py](backend/models/complaint.py)

Missing constraints:
- ❌ No `NOT NULL` constraint on required fields (except ticket_id, user_id)
- ❌ `amount_lost` stored as string (should be numeric)
- ❌ `description` length unlimited (could be massive)
- ❌ `date_time` stored as string (should be datetime)
- ❌ `complaint_type` not foreign key to COMPLAINT_TYPES table
- ❌ No `created_at` index for pagination
- ❌ No cascade delete if user is deleted

**User Model:** [backend/models/user.py](backend/models/user.py)

Issues:
- ❌ `email` marked unique but can store fallback emails (noemail_XXXXX@cyberguard.local)
- ❌ Repeated emails on re-submission create duplicates
- ❌ No email verification
- ❌ `phone` not marked unique (multiple complaints by same person)

### 3.4 Validation vs. Error Handling Disconnect

**Chat Route Issue:**
```python
# validate_create_complaint is called and returns errors
errors = validation_service.validate_create_complaint(payload.model_dump())
if errors:
    raise HTTPException(status_code=422, detail=errors)

# But in chat.py, _is_valid_field_value is called but never raises
# It just returns bool - caller must decide what to do
if not _is_valid_field_value(field, value):
    # What happens here? Just move to next field?
```

**Result:** Inconsistent error handling between endpoints

### 3.5 Frontend Error Recovery Absence

**Location:** [frontend/src/pages/ComplaintPage.tsx](frontend/src/pages/ComplaintPage.tsx)

```tsx
// On error, shows message but no recovery options:
setSubmitError(message); // Shows error
// No: retry button, fallback UX, data persistence, partial submission

// Upload errors silently fail
for (const file of data.evidenceFiles || []) {
  await uploadSingleFile(file, "evidence"); 
  // If one fails, subsequent files don't upload
}
```

---

## 4. EDGE CASES NOT COVERED

### 4.1 phone Number Edge Cases
- ✅ Handles: Country code (91), leading zeros removed
- ❌ Missing: Phone numbers with spaces (e.g., "9876 543210")
- ❌ Missing: Phone numbers with dashes (e.g., "98765-43210")
- ❌ Missing: Validation of actual phone number ownership
- ❌ Missing: Whitelist/blacklist of known fraud numbers

### 4.2 Email Edge Cases
- ✅ Handles: Special characters in email_validator
- ❌ Missing: Plus-addressing (user+tag@example.com) - could create duplicate accounts
- ❌ Missing: Disposable email detection
- ❌ Missing: Email verification/confirmation flow

### 4.3 Amount Lost Edge Cases
- ❌ Zero amount (0)? Valid or should require explicit N/A?
- ❌ Negative amounts (e.g., "-100")?
- ❌ Very large amounts (e.g., "99999999999")?
- ❌ Currency not specified (INR vs other)?
- ❌ Decimal amounts (e.g., "1000.50")?

### 4.4 Address Matching in ID Proof
**Location:** [backend/routes/complaint.py#L99](backend/routes/complaint.py#L99)

```python
# Current logic:
if address.strip() and extracted_address and \
   norm_text(address) not in norm_text(extracted_address) and \
   norm_text(extracted_address) not in norm_text(address):
    mismatches.append("address")
```

**Issues:**
- ❌ Substring matching could be too lenient (e.g., "45 Street" matches "145 Street")
- ❌ No handling of address abbreviations (e.g., "Rd" vs "Road", "St" vs "Street")
- ❌ No ZIP code/postal code validation

### 4.5 Concurrent Complaint Submissions
- ❌ No prevention of duplicate submissions within time window
- ❌ User could double-click submit and create 2 complaints
- ❌ No idempotency keys on submission endpoint

### 4.6 Language Switching Mid-Form
**Location:** [frontend/src/pages/ComplaintPage.tsx](frontend/src/pages/ComplaintPage.tsx#L690-L700)

```tsx
if (requestedLanguage && isLanguageChangeRequest(text)) {
    setLanguage(requestedLanguage);
    // Form data stays in old language
    // Do labels update? Do strings in description stay English?
}
```

**Issues:**
- ❌ Form labels switch but user-entered data in old language
- ❌ Description entered in English might be incompatible with Telugu workflow

### 4.7 Very Long Text Inputs
- ❌ No maximum length enforced for `description` field
- ❌ Could accept 1MB of text, causing:
  - DB storage issues
  - UI rendering lag
  - Translation API timeout

### 4.8 Unicode and Special Characters
- ❌ RTL language (Urdu, Arabic) not explicitly handled
- ❌ Emoji in names accepted (could be disruptive)
- ❌ Zero-width characters not filtered
- ❌ Homograph attacks not prevented (e.g., Cyrillic 'A' vs Latin 'A')

### 4.9 Session Management
**Location:** [frontend/src/services/api.ts#L165-L172](frontend/src/services/api.ts#L165-L172)

```typescript
function getSessionId() {
  const key = "cyberguard_session_id";
  const existing = localStorage.getItem(key);
  if (existing) return existing; // Reuses same session forever
  const generated = `sess-${crypto.randomUUID()}`;
  localStorage.setItem(key, generated);
  return generated;
}
```

**Issues:**
- ❌ No session expiration
- ❌ No cleanup of old sessions on backend
- ❌ Conversation history accumulates indefinitely

### 4.10 Complaint Type Fuzzy Matching
- ❌ No typo correction (e.g., "UPi Fraud" → "UPI / Payment Fraud")
- ❌ No abbreviation handling (e.g., "upi" → "UPI / Payment Fraud")
- ❌ Multilingual keyword matching incomplete

---

## 5. POTENTIAL SECURITY ISSUES

### 5.1 Authentication & Authorization

| Issue | Severity | Location | Details |
|-------|----------|----------|---------|
| No CSRF Protection | High | FastAPI app | No CSRF token validation |
| JWT Not Refreshable | Medium | [security.py](backend/utils/security.py) | Fixed 60min expiry, no refresh token |
| No Rate Limiting | High | All endpoints | Could enable brute force |
| Admin Role Hardcoded | Medium | [admin.py](backend/routes/admin.py) | Only binary admin/non-admin |
| No Audit Logging | Medium | All endpoints | No logging of who changed what |

### 5.2 Input Validation Vulnerabilities

| Vulnerability | Risk | Location |
|---|---|---|
| File Upload No Type Check | High | [complaint.py#L116](backend/routes/complaint.py#L116) |
| No File Size Limits | Medium | [complaint.py](backend/routes/complaint.py) |
| Uploaded File Path Exposure | Medium | [api.ts](frontend/src/services/api.ts) - returns file_path |
| Language Code Injection | Low | [chat.py#L512](backend/routes/chat.py#L512) |

### 5.3 Data Exposure Risks

| Risk | Severity | Impact |
|------|----------|--------|
| Sensitive paths in error messages | Low | [api.ts#L169-L180](frontend/src/services/api.ts#L169-L180) - file paths exposed |
| Unencrypted Passwords → Hash | Medium | [security.py](backend/utils/security.py) - Uses bcrypt ✅ |
| Evidence files publicly accessible | High | `/uploads` mount - anyone can access |
| Admin dashboard no rate limiting | High | Could enumerate complaints |

---

## 6. DATABASE INTEGRITY ISSUES

### Current Schema Gaps

```sql
-- Current Issues:

-- ❌ No NOT NULL constraints on required fields
ALTER TABLE complaints ADD CONSTRAINT check_required_fields 
  CHECK (full_name IS NOT NULL AND phone_number IS NOT NULL);

-- ❌ No enum constraint on status
ALTER TABLE complaints ADD CONSTRAINT check_status 
  CHECK (status IN ('pending', 'reviewing', 'resolved'));

-- ❌ amount_lost as TEXT, should be NUMERIC
ALTER TABLE complaints MODIFY COLUMN amount_lost DECIMAL(10, 2);

-- ❌ date_time as TEXT, should be DATETIME
ALTER TABLE complaints MODIFY COLUMN date_time TIMESTAMP;

-- ❌ description unlimited, should be bounded
ALTER TABLE complaints MODIFY COLUMN description VARCHAR(5000) NOT NULL;

-- ❌ No foreign key for complaint_type
ALTER TABLE complaints ADD CONSTRAINT fk_complaint_type 
  FOREIGN KEY (complaint_type) REFERENCES complaint_types(name);

-- ❌ User email uniqueness conflict with fallback emails
-- Current data has: noemail_5551234567@cyberguard.local (not truly unique per user)

-- ✅ User has unique email constraint (good for real emails)
-- But fallback email workaround breaks this intent
```

---

## 7. TODO/FIXME ITEMS FOUND

| File | Line | Comment | Status |
|------|------|---------|--------|
| [src/services/api.ts](src/services/api.ts) | 72 | TODO: Replace with actual API call | ❌ Old Code |
| [src/services/api.ts](src/services/api.ts) | 85 | TODO: Replace with actual API call | ❌ Old Code |
| [src/services/api.ts](src/services/api.ts) | 139 | TODO: Replace with actual Gemini API call | ❌ Old Code |

**Note:** These appear to be in the old `/src` directory (not `/frontend/src`), suggesting legacy code that should be cleaned up.

---

## 8. EXCEPTION HANDLING PATTERNS

### 8.1 Current Patterns

#### Pattern A: Log & Suppress (Most Common)
```python
# Location: routes/complaint.py, routes/admin.py
try:
    result = external_service.do_something()
except Exception as exc:
    logger.warning("Operation failed: %s", exc)
    # Continue without result or with default
```

**Issues:** 
- User unaware of failure
- Data quality degradation
- No retry mechanism

#### Pattern B: Raise HTTPException (Explicit)
```python
# Location: routes/complaint.py, routes/ai.py
if not file.filename:
    raise HTTPException(status_code=400, detail="Invalid file")
```

**Issues:**
- Inconsistent error response format
- No request context/tracing

#### Pattern C: Frontend Try-Catch with Message
```typescript
// Location: ComplaintPage.tsx
try {
    const res = await submitComplaint(...);
} catch (err) {
    setSubmitError(message);
    speak(guidance, language.speechCode);
}
```

**Issues:**
- Generic error message to user
- No error categorization
- No user guidance for recovery

### 8.2 Missing Error Handling

```python
# ❌ No error handling
@router.post("/detect-language")
def detect_language(payload: DetectLanguageRequest):
    language = translation_service.detect_language(payload.text)
    # What if text is empty? What if service fails?
    return DetectLanguageResponse(language=language)

# ❌ No validation
@router.post("/translate")
def translate(payload: TranslateRequest):
    translated = translation_service.translate(...)
    # No null check, no error handling
    return TranslateResponse(translated_text=translated)
```

---

## 9. RECOMMENDATIONS FOR IMPROVEMENTS

### 🔴 Critical (Do Immediately)

1. **Add File Upload Validations**
   - Max file size: 50MB
   - Whitelist MIME types: pdf, image/*, audio/*, video/*
   - Verify file magic bytes (magic number check)
   - Scan for malware integration (VirusTotal API)
   
   ```python
   class FileValidator:
       MAX_SIZE = 50 * 1024 * 1024
       ALLOWED_TYPES = {'application/pdf', 'image/jpeg', 'image/png', ...}
       
       def validate(self, file: UploadFile):
           if len(file.file.read()) > self.MAX_SIZE:
               raise HTTPException(413, "File too large")
           # Check magic bytes
   ```

2. **Add Database Constraints**
   ```sql
   ALTER TABLE complaints ADD CONSTRAINT check_amounts 
     CHECK (amount_lost IS NULL OR try_cast(amount_lost AS DECIMAL) >= 0);
   ALTER TABLE complaints ADD CONSTRAINT check_description_length 
     CHECK (length(description) BETWEEN 10 AND 5000);
   ```

3. **Implement Request Rate Limiting**
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   
   @app.post("/complaint/create")
   @limiter.limit("10/hour")
   def create_complaint(...):
       ...
   ```

4. **Add Duplicate Submission Prevention**
   ```python
   @router.post("/complaint/create")
   def create_complaint(payload: ComplaintCreateRequest, db: Session):
       # Check if identical complaint submitted in last 5 minutes
       recent = db.query(Complaint).filter(
           Complaint.user_id == user.id,
           Complaint.created_at > datetime.utcnow() - timedelta(minutes=5),
           Complaint.description == payload.description
       ).first()
       if recent:
           raise HTTPException(409, "Duplicate submission detected")
   ```

5. **Implement Proper Error Response Schema**
   ```python
   class APIError(BaseModel):
       code: str  # "VALIDATION_ERROR", "FILE_TOO_LARGE", etc.
       message: str
       details: Optional[Dict] = None
       timestamp: datetime
       request_id: str  # For debugging
   ```

### 🟡 High Priority (Next Sprint)

6. **Add String Length Boundaries to All Fields**
   ```python
   class ComplaintCreateRequest(BaseModel):
       full_name: Annotated[str, Field(min_length=3, max_length=255)]
       description: Annotated[str, Field(min_length=10, max_length=5000)]
       # ...
   ```

7. **Add Numeric Field Validation**
   ```python
   class ComplaintCreateRequest(BaseModel):
       amount_lost: Optional[Annotated[float, Field(ge=0, le=10000000)]]
   ```

8. **Implement Error Retry Logic**
   ```python
   async def call_with_retry(fn, max_retries=3, backoff=1):
       for attempt in range(max_retries):
           try:
               return await fn()
           except RetryableError:
               if attempt < max_retries - 1:
                   await asyncio.sleep(backoff * (2 ** attempt))
               else:
                   raise
   ```

9. **Add Input Sanitization Middleware**
   ```python
   # Clean XSS vectors, normalize unicode, etc.
   def sanitize_input(value: str) -> str:
       # HTML escape, remove control chars, normalize
       return html.escape(value).strip()
   ```

10. **Add Complaint Type Fuzzy Matching**
    ```python
    from difflib import get_close_matches
    
    def normalize_complaint_type(user_input: str):
        matches = get_close_matches(user_input, COMPLAINT_TYPES, n=1, cutoff=0.8)
        return matches[0] if matches else None
    ```

### 🟢 Medium Priority (Backlog)

11. **Implement Frontend Error Boundaries**
    ```tsx
    class ErrorBoundary extends React.Component {
      componentDidCatch(error, errorInfo) {
        // Log to Sentry or similar
        // Show user-friendly error UI
      }
    }
    ```

12. **Add Session Expiration**
    ```typescript
    // In localStorage
    interface Session {
      id: string;
      createdAt: number;
      expiresAt: number;  // 30 min from now
    }
    ```

13. **Add Evidence Upload Progress Tracking**
    ```tsx
    // Show % uploaded for each file
    // Allow pause/resume
    // Rollback on failure
    ```

14. **Implement Audit Logging**
    ```python
    class AuditLog(Base):
        __tablename__ = "audit_logs"
        id: int
        user_id: Optional[int]
        action: str  # "complaint_created", "complaint_updated", etc.
        resource_id: str  # ticket_id
        timestamp: datetime
        ip_address: str
    ```

15. **Add Email Verification Flow**
    ```python
    # User receives email with confirmation link
    # Link contains JWT token
    # Only after confirmation is email valid
    ```

16. **Implement Complaint Status Transition Validation**
    ```python
    VALID_STATUS_TRANSITIONS = {
        "pending": {"reviewing", "resolved"},
        "reviewing": {"pending", "resolved"},
        "resolved": set()  # Terminal state
    }
    ```

17. **Add Payment Amount Validation by Complaint Type**
    ```python
    AMOUNT_LIMITS = {
        "UPI / Payment Fraud": {"min": 0, "max": 100000},
        "Internet Banking Fraud": {"min": 0, "max": 1000000},
    }
    ```

18. **Implement Language-Specific Keyboard Support**
    ```tsx
    // Detect RTL languages and adjust input layout
    const isRTL = (lang: string) => ["ur", "ar"].includes(lang);
    // Apply dir="rtl" to inputs
    ```

---

## 10. IMPLEMENTATION ROADMAP

### Phase 1: Critical Security (Week 1)
- [ ] File upload validations
- [ ] Request rate limiting
- [ ] Duplicate submission prevention
- [ ] Database constraints

### Phase 2: Data Integrity (Week 2)
- [ ] String length boundaries
- [ ] Numeric field validation
- [ ] Input sanitization middleware
- [ ] Error response schema

### Phase 3: Resilience (Week 3)
- [ ] Retry logic for external services
- [ ] Error boundaries in frontend
- [ ] Session expiration
- [ ] Audit logging

### Phase 4: UX & Polish (Week 4)
- [ ] Email verification
- [ ] Fuzzy complaint type matching
- [ ] Upload progress tracking
- [ ] Better error messages

---

## 11. VALIDATION CHECKLIST

Use this checklist when adding new endpoints:

- [ ] Input schema with min/max lengths
- [ ] Required vs optional fields clearly marked
- [ ] Numeric ranges specified (ge, le)
- [ ] Enum values validated
- [ ] Error handling with specific error codes
- [ ] Typed error response
- [ ] Rate limiting applied
- [ ] File uploads (if any) validated
- [ ] Database constraints matching validation
- [ ] Frontend form validation matches backend
- [ ] Error recovery path documented
- [ ] Retry logic for external services
- [ ] Audit log entry (if applicable)

---

## 12. KEY METRICS TO MONITOR

```python
# Add to backend telemetry
metrics = {
    "validation_failures_per_endpoint": {
        "/complaint/create": Counter,
        "/ai/speech-to-text": Counter,
    },
    "error_codes_by_type": {
        "422_validation": Counter,
        "500_internal": Counter,
        "413_file_too_large": Counter,
    },
    "external_service_failures": {
        "gemini_id_analysis": Counter,
        "translation_service": Counter,
        "ollama_chat": Counter,
    },
    "user_error_recovery_attempts": Counter,
}
```

---

## APPENDIX: VALIDATION MATRIX

| Field | Type | Min | Max | Pattern | Required | DB Constraint | Frontend | Backend | Notes |
|-------|------|-----|-----|---------|----------|---|---|---|---|
| full_name | str | 3 | ❌ | None | Yes | ❌ | ✅ | ✅ | Needs max |
| phone | str | 10 | 10 | `^[6-9]\d{9}$` | Yes | ❌ | ✅ | ✅ | ✅ Good |
| email | str | 1 | ❌ | RFC/simple | No | ❌ | ✅ | ✅ | Needs max |
| address | str | 6 | ❌ | None | No | ❌ | ⚠️ | ✅ | Needs max |
| complaint_type | enum | - | - | COMPLAINT_TYPES | Yes | ❌ | ✅ | ✅ | Add FK |
| description | str | 10 | ❌ | None | Yes | ❌ | ✅ | ✅ | Needs max=5000 |
| amount_lost | numeric | 0 | ❌ | - | No | ❌ | ⚠️ | ❌ | Stored as TEXT! |
| transaction_id | str | 8 | 30 | Alphanumeric | No | ❌ | ✅ | ✅ | ✅ Good |
| platform | str | 2 | ❌ | None | No | ❌ | ✅ | ✅ | Needs max |
| date_time | str | - | - | ISO/natural | No | ❌ | ⚠️ | ✅ | Stored as TEXT! |
| suspect_vpa | str | - | ❌ | UPI format | No | ❌ | ❌ | ❌ | ❌ Major gap |
| suspect_phone | str | - | ❌ | Phone pattern | No | ❌ | ❌ | ❌ | ❌ Major gap |
| suspect_bank_account | str | - | ❌ | IFSC+Account | No | ❌ | ❌ | ❌ | ❌ Major gap |

---

## CONCLUSION

Your application has a **solid foundation** for validation but needs:

1. **Immediate attention:** File uploads, rate limiting, database constraints
2. **High priority:** String boundaries, numeric validation, error responses
3. **Direction:** Add comprehensive retry logic and error recovery

Start with Phase 1 (Critical Security) to harden the system, then progressively add resilience and polish.

