# CyberGuard AI - Developer Quick Reference

## 🚀 Quick Start

### Local Development (5 minutes)

```bash
# Terminal 1: Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python test_system.py  # Verify all tests pass
python -m uvicorn main:app --reload

# Terminal 2: Frontend
cd frontend
npm install
npm run dev

# Access: Frontend at http://localhost:5173, API at http://localhost:8000
```

---

## 📦 Key Services & Usage

### 1. Enhanced Validation Service

**File:** `backend/services/enhanced_validation.py`

```python
from services.enhanced_validation import validation_service

# Validate complete data
is_valid, errors = validation_service.validate_complaint_data({
    'full_name': 'Test User',
    'phone_number': '9876543210',
    'email': 'test@example.com',
    'complaint_type': 'UPI Fraud',
    'description': 'Details here',
    'amount_lost': '5000'
})
# Returns: (True, []) or (False, ['error1', 'error2'])

# Validate individual fields
valid, error = validation_service.validate_phone_number('9876543210')  # True
valid, error = validation_service.validate_email('test@example.com')   # True
valid, error = validation_service.validate_amount('5000')              # True

# Check for SQL injection
is_safe = validation_service.is_safe_from_injection('normal text')           # True
is_safe = validation_service.is_safe_from_injection("'; DROP TABLE --")  # False

# File upload validation
is_valid, error = validation_service.validate_file_upload(
    'document.pdf', 1024*1024, 'application/pdf'
)  # (True, None)

# Sanitize input
cleaned = validation_service.sanitize_input("<script>alert('xss')</script>")
# Returns: "scriptalert('xss')/script"
```

---

### 2. LLM Selector Service

**File:** `backend/services/llm_selector.py`

```python
from services.llm_selector import get_active_llm_name, get_llm_client, generate_ai_response

# Get active LLM
llm = get_active_llm_name()  # "gemini-2.0-flash" or "ollama-llama2"

# Get client for direct API calls
client = get_llm_client()
# Standard Google GenerativeAI or Ollama client interface

# Generate AI response (preferred method)
response = generate_ai_response(
    complaint_type="UPI Fraud",
    description="Unauthorized UPI transaction",
    language="en"
)
# Returns: "AI-generated analysis and recommendations..."

# Environment check
gemini_available = os.getenv('GEMINI_API_KEY') is not None
ollama_available = check_ollama_connection()
```

---

### 3. Database Models & Operations

**File:** `backend/models/`

```python
from models.complaint import Complaint
from models.user import User
from models.evidence import Evidence
from database import SessionLocal

db = SessionLocal()

# Create complaint
complaint = Complaint(
    ticket_id="TKT-2024-001",
    user_id=1,
    complaint_type="UPI Fraud",
    description="Details",
    status="submitted",
    language="en"
)
db.add(complaint)
db.commit()

# Query complaints
recent_complaints = db.query(Complaint)\
    .order_by(Complaint.created_at.desc())\
    .limit(10)\
    .all()

# Get specific ticket
complaint = db.query(Complaint)\
    .filter(Complaint.ticket_id == "TKT-2024-001")\
    .first()

# Update status
complaint.status = "under_review"
db.commit()

db.close()
```

---

### 4. API Routes Integration

**File:** `backend/routes/complaint.py`

```python
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/complaint", tags=["Complaint"])

@router.post("/create")
def create_complaint(payload: ComplaintCreateRequest, db: Session = Depends(get_db)):
    # Enhanced validation
    is_valid, errors = enhanced_validation.validate_complaint_data(payload.model_dump())
    if not is_valid:
        raise HTTPException(status_code=422, detail=errors)
    
    # Log which LLM is used
    llm = get_active_llm_name()
    logger.info(f"Processing with LLM: {llm}")
    
    # Process complaint...
    return ComplaintCreateResponse(...)
```

---

### 5. Email Service

**File:** `backend/services/email_service.py`

```python
from services.email_service import email_service

# Send confirmation email
email_service.send_user_ticket_email(
    email="user@example.com",
    ticket_id="TKT-2024-001",
    summary="Complaint submitted successfully"
)

# Send admin alert
email_service.send_admin_complaint_email(
    admin_email="admin@example.com",
    ticket_id="TKT-2024-001",
    summary="New complaint received"
)
```

---

### 6. Translation Service

**File:** `backend/services/translation_service.py`

```python
from services.translation_service import translation_service

# Translate to specific language
translated = translation_service.translate(
    text="Your complaint has been registered",
    language="hi"
)
# Returns: "आपकी शिकायत दर्ज की गई है"

# Translate to English
english = translation_service.to_english("मेरा UPI लेनदेन धोखाधड़ी है")
# Returns: "My UPI transaction is fraud"
```

---

## 🧪 Testing Quick Reference

### Run All Tests
```bash
cd backend
python test_system.py
```

**Output Check:**
- ✅ Validation Layer tests
- ✅ LLM Selection tests
- ✅ Database Operations tests
- ✅ File Handling tests
- ✅ Character Handling tests

### Run Individual Service Tests

```bash
# Test validation
python -c "
from services.enhanced_validation import validation_service
is_valid, errors = validation_service.validate_complaint_data({
    'full_name': 'Test',
    'phone_number': '9876543210',
    'email': 'test@example.com',
    'complaint_type': 'UPI Fraud',
    'description': 'Test',
    'amount_lost': '1000'
})
print('Valid:', is_valid)
"

# Test LLM selector
python -c "
from services.llm_selector import get_active_llm_name
print('Active LLM:', get_active_llm_name())
"

# Test database
python -c "
from database import SessionLocal
db = SessionLocal()
print('✅ DB Connection OK')
db.close()
"
```

---

## 📝 Common Validation Patterns

### Phone Number
```python
# Valid Indian phone numbers
'9876543210'    # ✅ Valid
'+919876543210' # ✅ Valid (auto-converts)
'9876543210'    # ✅ Valid
'1234567890'    # ❌ Invalid (starts with 1)
'12345'         # ❌ Too short
```

### Email
```python
# Valid emails
'user@example.com'         # ✅ Valid
'first.last@domain.co.uk'  # ✅ Valid
'user@example'             # ❌ Invalid (no TLD)
'invalid email@example'    # ❌ Invalid (spaces)
```

### Amount
```python
# Valid amounts
'5000'      # ✅ Valid
'5000.50'   # ✅ Valid (decimal)
'-5000'     # ❌ Invalid (negative)
'abc'       # ❌ Invalid (non-numeric)
```

### Red Flags (Security)
```python
# Injection attempts detected
"'; DROP TABLE --"         # ❌ SQL injection
"<script>alert('xss')</script>"  # ❌ XSS attempt
"1' OR '1'='1"             # ❌ Boolean attack
"*/ /* comment"            # ❌ Comment-based attack
```

---

## 🔧 Configuration & Environment

### .env Template
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/cyberguard
SUPABASE_URL=https://project.supabase.co
SUPABASE_KEY=your-key

# AI
GEMINI_API_KEY=your-gemini-key
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_MODEL=llama2

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=email@gmail.com
SMTP_PASSWORD=app-specific-password

# App
ENVIRONMENT=development
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key
DEBUG=false
```

### Load Environment
```python
from config import settings

api_key = settings.gemini_api_key
db_url = settings.database_url
smtp_host = settings.smtp_host
```

---

## 📊 API Endpoint Quick Reference

### Complaint Management

```bash
# Create complaint
POST /complaint/create
Content-Type: application/json
{
  "full_name": "Test User",
  "phone_number": "9876543210",
  "email": "test@example.com",
  "complaint_type": "UPI Fraud",
  "description": "Details",
  "language": "en",
  "amount_lost": "5000"
}
# Response: {"ticket_id": "TKT-2024-001", "status": "submitted", ...}

# Upload evidence
POST /complaint/upload
Content-Type: multipart/form-data
ticket_id: TKT-2024-001
file: @document.pdf
# Response: {"ticket_id": "TKT-2024-001", "evidence_id": 1, ...}

# Track complaint
GET /complaint/TKT-2024-001
# Response: {"ticket_id": "TKT-2024-001", "status": "under_review", ...}

# Validate ID
POST /complaint/validate-id
Content-Type: multipart/form-data
full_name: Test User
phone_number: 9876543210
file: @id-proof.jpg
# Response: {"analysis": {...}, "mismatch_fields": [], ...}
```

---

## 🐛 Debugging Tips

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### Check LLM Status
```bash
curl http://localhost:8000/health
# or in code:
from services.llm_selector import get_active_llm_name
print(get_active_llm_name())
```

### Database Connection
```python
from database import SessionLocal, engine
from sqlalchemy import text

# Test connection
with SessionLocal() as db:
    result = db.execute(text("SELECT 1"))
    print("✅ DB Connected")
```

### View Logs
```bash
# Application logs
tail -f logs/app.log

# Error logs
tail -f logs/error.log

# Systemd logs (production)
sudo journalctl -u cyberguard-backend -f
```

---

## 📂 Project Structure Quick Reference

```
backend/
├── main.py                 # FastAPI app entry point
├── config.py              # Configuration & settings
├── database.py            # DB connection & session
├── requirements.txt       # Python dependencies
├── test_system.py         # System tests
├── models/                # Database models
│   ├── complaint.py
│   ├── user.py
│   └── evidence.py
├── routes/                # API endpoints
│   ├── complaint.py       # Updated with enhanced validation
│   └── admin.py
├── services/              # Business logic
│   ├── enhanced_validation.py    # NEW: Advanced validation
│   ├── llm_selector.py          # NEW: LLM selection
│   ├── gemini_service.py
│   ├── translation_service.py
│   └── email_service.py
├── schemas/               # Request/response models
└── utils/                 # Utilities
    ├── security.py
    └── file_utils.py

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   ├── services/         # API client
│   ├── contexts/
│   ├── hooks/
│   └── App.tsx
└── package.json

docs/
├── API_INTEGRATION_GUIDE.md      # NEW: API reference
├── DEPLOYMENT_GUIDE.md            # NEW: Deployment steps
└── FEATURES.md                    # NEW: Feature documentation
```

---

## 🔐 Security Checklist

- [ ] All inputs validated via `enhanced_validation_service`
- [ ] SQL injection checks passed
- [ ] XSS protections enabled
- [ ] File uploads limited to 50 MB max
- [ ] Allowed file types: PDF, PNG, JPG, JPEG, DOCX only
- [ ] HTTPS enabled in production
- [ ] API keys not in code (use .env)
- [ ] Database URL from environment
- [ ] User data encrypted at rest
- [ ] Rate limiting implemented

---

## 🚢 Deployment Checklist

- [ ] All environment variables configured
- [ ] Database migrations applied
- [ ] test_system.py passes all tests
- [ ] Gemini API key configured (or Ollama ready)
- [ ] Email service tested
- [ ] HTTPS certificate generated
- [ ] Nginx reverse proxy configured
- [ ] Systemd service created
- [ ] Log rotation configured
- [ ] Backup strategy implemented
- [ ] Monitoring tools installed

---

## 💡 Pro Tips

1. **Use validation service first**
   ```python
   # ✅ Good
   is_valid, errors = enhanced_validation.validate_complaint_data(data)
   
   # ❌ Avoid
   # Don't trust user input directly
   ```

2. **Check LLM before using**
   ```python
   # ✅ Good
   llm = get_active_llm_name()
   logger.info(f"Using: {llm}")
   
   # ❌ Avoid
   # Don't assume Gemini is always available
   ```

3. **Always use context managers for DB**
   ```python
   # ✅ Good
   with SessionLocal() as db:
       result = db.query(...)
   
   # ❌ Avoid
   db = SessionLocal()
   # ... might forget to close
   ```

4. **Log important transitions**
   ```python
   # ✅ Good
   logger.info(f"Ticket {ticket_id} status: submitted → under_review")
   
   # ❌ Avoid
   # Silent updates make debugging hard
   ```

---

## 📞 Quick Help

**Need to...**

| Task | Command/File |
|------|--------------|
| Run tests | `python test_system.py` |
| Start backend | `python -m uvicorn main:app --reload` |
| Start frontend | `npm run dev` |
| Deploy | See `DEPLOYMENT_GUIDE.md` |
| Check API | See `API_INTEGRATION_GUIDE.md` |
| Configure env | Edit `.env` file |
| View logs | `tail -f logs/app.log` |
| Debug DB | `python -c "from database import SessionLocal; SessionLocal()"` |
| Check LLM | `python -c "from services.llm_selector import get_active_llm_name; print(get_active_llm_name())"` |

---

**Last Updated:** January 2024 | **Version:** 1.0.0
