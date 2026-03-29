# CyberGuard AI - Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the CyberGuard AI system with enhanced validation, LLM selection, and multi-language support.

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Environment Setup](#environment-setup)
3. [Local Development Deployment](#local-development-deployment)
4. [Production Deployment](#production-deployment)
5. [Docker Deployment](#docker-deployment)
6. [Vercel Deployment (Frontend)](#vercel-deployment-frontend)
7. [Post-Deployment Verification](#post-deployment-verification)
8. [Troubleshooting](#troubleshooting)

---

## Pre-Deployment Checklist

### Required Services & APIs

- [ ] **Supabase Project**
  - Database created and running
  - API keys generated
  - Migrations applied

- [ ] **Gemini API** (Optional, for AI features)
  - API key obtained from Google Cloud
  - API enabled in Google Cloud Console
  - Quota configured (recommended: 1000 requests/day minimum)

- [ ] **Ollama** (Optional, for local LLM)
  - Installed and running locally or on separate server
  - Llama2 model downloaded
  - API accessible at configured URL

- [ ] **Email Service**
  - SMTP credentials for sending notifications
  - Sender email address configured
  - Test email sent successfully

### System Requirements

**Backend:**
- Python 3.10+
- 4GB RAM (8GB recommended)
- 10GB disk space (20GB for production)
- Linux, macOS, or Windows (WSL2)

**Frontend:**
- Node.js 18+
- npm or bun
- 2GB disk space

---

## Environment Setup

### 1. Backend Environment Variables

Create `.env` file in `/backend/:

```bash
# Database
DATABASE_URL=postgresql://user:password@host:5432/cyberguard
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-api-key

# AI Services
GEMINI_API_KEY=your-gemini-api-key
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_MODEL=llama2

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-specific-password
SENDER_EMAIL=noreply@cyberguard.ai
ADMIN_EMAIL=admin@cyberguard.ai

# Application Settings
ENVIRONMENT=development  # or 'production'
SECRET_KEY=your-secret-key-change-in-production
DEBUG=false
LOG_LEVEL=INFO

# Frontend URL (for CORS)
FRONTEND_URL=http://localhost:5173

# File Upload
MAX_FILE_SIZE=52428800  # 50 MB in bytes
UPLOAD_DIR=/var/uploads/cyberguard
```

### 2. Frontend Environment Variables

Create `.env.local` file in `/frontend/:

```bash
VITE_API_URL=http://localhost:8000
VITE_API_TIMEOUT=30000
```

For production:
```bash
VITE_API_URL=https://api.cyberguard.ai
VITE_API_TIMEOUT=30000
```

### 3. Google Cloud Setup (for Gemini API)

```bash
# 1. Create Google Cloud Project
gcloud projects create cyberguard-ai

# 2. Enable Generative AI API
gcloud services enable generativelanguage.googleapis.com

# 3. Create service account
gcloud iam service-accounts create cyberguard-ai

# 4. Grant permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member=serviceAccount:cyberguard-ai@PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/aiplatform.user

# 5. Create API key
gcloud alpha services api-keys create --display-name="CyberGuard AI"

# 6. Add to .env
GEMINI_API_KEY=your-generated-api-key
```

---

## Local Development Deployment

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python init_supabase.py

# Run tests
python test_system.py

# Start development server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install  # or: bun install

# Start development server
npm run dev  # or: bun run dev

# Browser will open at http://localhost:5173
```

### 3. Ollama Setup (Optional, for Local LLM)

```bash
# Install Ollama (on macOS)
brew install ollama

# Run Ollama service
ollama serve

# In another terminal, download Llama2 model
ollama pull llama2

# Ollama API will be available at http://localhost:11434
```

---

## Production Deployment

### 1. Server Preparation (Ubuntu/Debian)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3.10 python3-pip python3.10-venv

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Create application user
sudo useradd -m -s /bin/bash cyberguard

# Create directories
sudo mkdir -p /opt/cyberguard/{backend,frontend}
sudo mkdir -p /var/uploads/cyberguard
sudo mkdir -p /var/log/cyberguard

# Set permissions
sudo chown -R cyberguard:cyberguard /opt/cyberguard
sudo chown -R cyberguard:cyberguard /var/uploads/cyberguard
sudo chown -R cyberguard:cyberguard /var/log/cyberguard
```

### 2. Backend Production Setup

```bash
# Switch to cyberguard user
sudo su - cyberguard

# Navigate to backend
cd /opt/cyberguard/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Copy application code
# (from your version control system)

# Install dependencies
pip install -r requirements.txt
pip install gunicorn systemctl

# Run database migrations
python init_supabase.py

# Run system test
python test_system.py
```

### 3. Create Backend Systemd Service

Create `/etc/systemd/system/cyberguard-backend.service`:

```ini
[Unit]
Description=CyberGuard AI Backend Service
After=network.target
Requires=cyberguard-backend.socket

[Service]
Type=notify
User=cyberguard
Group=cyberguard
WorkingDirectory=/opt/cyberguard/backend
Environment="PATH=/opt/cyberguard/backend/venv/bin"
EnvironmentFile=/opt/cyberguard/backend/.env
ExecStart=/opt/cyberguard/backend/venv/bin/gunicorn \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind unix:/run/gunicorn.sock \
    --access-logfile /var/log/cyberguard/access.log \
    --error-logfile /var/log/cyberguard/error.log \
    --log-level info \
    main:app

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 4. Frontend Production Build

```bash
cd /opt/cyberguard/frontend

# Install dependencies
npm install

# Create production build
npm run build

# Build output in 'dist' directory
```

### 5. Setup Nginx Reverse Proxy

Create `/etc/nginx/sites-available/cyberguard`:

```nginx
upstream cyberguard_backend {
    server unix:/run/gunicorn.sock fail_timeout=0;
}

server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;
    
    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Frontend static files
    location / {
        root /opt/cyberguard/frontend/dist;
        try_files $uri $uri/ /index.html;
    }
    
    # Backend API proxy
    location /api/ {
        proxy_pass http://cyberguard_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://cyberguard_backend;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/cyberguard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6. Setup SSL Certificate

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain certificate
sudo certbot certonly --nginx -d your-domain.com -d www.your-domain.com

# Auto-renewal (already configured)
sudo systemctl enable certbot.timer
```

### 7. Enable and Start Services

```bash
# Backend service
sudo systemctl enable cyberguard-backend
sudo systemctl start cyberguard-backend
sudo systemctl status cyberguard-backend

# Check logs
sudo journalctl -u cyberguard-backend -f

# Nginx
sudo systemctl restart nginx
```

---

## Docker Deployment

### 1. Backend Dockerfile

```dockerfile
# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 cyberguard && \
    chown -R cyberguard:cyberguard /app
USER cyberguard

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Frontend Dockerfile

```dockerfile
# Build stage
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --prefer-offline --no-audit

COPY . .
RUN npm run build

# Production stage
FROM node:18-alpine

WORKDIR /app

# Install serve to run the production build
RUN npm install -g serve

# Copy built application from builder
COPY --from=builder /app/dist ./dist

# Create non-root user
RUN addgroup -g 1000 frontend && \
    adduser -D -u 1000 -G frontend frontend && \
    chown -R frontend:frontend /app
USER frontend

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:3000/

# Expose port
EXPOSE 3000

# Run application
CMD ["serve", "-s", "dist", "-l", "3000"]
```

### 3. Docker Compose

Create `docker-compose.yml` in project root:

```yaml
version: '3.9'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OLLAMA_API_BASE=${OLLAMA_API_BASE}
      - SMTP_HOST=${SMTP_HOST}
      - SMTP_PORT=${SMTP_PORT}
      - SMTP_USER=${SMTP_USER}
      - SMTP_PASSWORD=${SMTP_PASSWORD}
    volumes:
      - uploads:/var/uploads/cyberguard
      - logs:/var/log/cyberguard
    depends_on:
      - ollama
    networks:
      - cyberguard

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://backend:8000
    depends_on:
      - backend
    networks:
      - cyberguard

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
    networks:
      - cyberguard
    # Pull model on startup
    command: sh -c "ollama pull llama2 && ollama serve"

volumes:
  uploads:
  logs:
  ollama:

networks:
  cyberguard:
    driver: bridge
```

### 4. Deploy with Docker Compose

```bash
# Create environment file
cp .env.example .env
# Edit .env with your credentials

# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Vercel Deployment (Frontend)

### 1. Connect to Vercel

```bash
cd frontend
npm install -g vercel
vercel link
```

### 2. Set Environment Variables

```bash
vercel env add VITE_API_URL
# Enter production API URL
```

### 3. Deploy

```bash
vercel --prod
```

Or configure automatic deployment:
- Push to GitHub
- Vercel automatically deploys on push to main branch

---

## Post-Deployment Verification

### 1. Backend Health Check

```bash
# Check backend is running
curl http://your-domain.com/health

# Expected response:
# {"status": "ok"}
```

### 2. Database Connection

```bash
python -c "
from database import SessionLocal
db = SessionLocal()
print('✅ Database connection successful')
db.close()
"
```

### 3. LLM Selection

```bash
curl http://your-domain.com/api/debug/llm-status

# Expected response:
# {"active_llm": "gemini-2.0-flash", "gemini_available": true}
```

### 4. Run System Tests

```bash
python test_system.py
```

### 5. Test Complaint Submission

```bash
curl -X POST http://your-domain.com/api/complaint/create \
  -H "Content-Type: application/json" \
  -d '{
    "full_name": "Test User",
    "phone_number": "9876543210",
    "email": "test@example.com",
    "complaint_type": "UPI Fraud",
    "description": "Test complaint",
    "language": "en"
  }'
```

---

## Monitoring & Maintenance

### 1. Log Rotation

Create `/etc/logrotate.d/cyberguard`:

```
/var/log/cyberguard/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 cyberguard cyberguard
    sharedscripts
    postrotate
        systemctl reload cyberguard-backend > /dev/null 2>&1 || true
    endscript
}
```

### 2. Backup Strategy

```bash
# Daily database backups
0 2 * * * /usr/local/bin/backup-cyberguard.sh

# Script: /usr/local/bin/backup-cyberguard.sh
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/cyberguard"
mkdir -p $BACKUP_DIR
pg_dump -h $DB_HOST -U $DB_USER $DB_NAME | gzip > $BACKUP_DIR/cyberguard_$DATE.sql.gz
find $BACKUP_DIR -type f -mtime +30 -delete
```

### 3. Monitoring Setup

```bash
# Install Prometheus node exporter
sudo wget https://github.com/prometheus/node_exporter/releases/latest/download/node_exporter-latest.linux-amd64.tar.gz
sudo tar xvfz node_exporter-latest.linux-amd64.tar.gz -C /opt/
```

---

## Troubleshooting

### Issue: Backend won't start

```bash
# Check logs
sudo journalctl -u cyberguard-backend -n 50

# Verify environment variables
env | grep GEMINI

# Test database connection
python -c "from database import SessionLocal; SessionLocal()"
```

### Issue: High memory usage

```bash
# Check memory
ps aux | grep gunicorn

# Reduce workers in systemd service
# Change: --workers 4 to --workers 2
```

### Issue: CORS errors from frontend

```bash
# Check CORS configuration in main.py
# Ensure FRONTEND_URL is correct in .env
FRONTEND_URL=http://localhost:5173
```

### Issue: Email not sending

```bash
# Test SMTP credentials
python -c "
import smtplib
from email.mime.text import MIMEText

sender = 'your-email@gmail.com'
password = 'your-app-specific-password'

try:
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender, password)
    print('✅ SMTP connection successful')
    server.quit()
except Exception as e:
    print(f'❌ SMTP error: {e}')
"
```

---

## Performance Optimization

### 1. Database Optimization

```sql
-- Add indexes for frequently queried fields
CREATE INDEX idx_ticket_id ON complaints(ticket_id);
CREATE INDEX idx_user_phone ON users(phone_number);
CREATE INDEX idx_status ON complaints(status);
```

### 2. Caching

```python
# Add Redis caching in main.py
from fastapi_cache2 import FastAPICache2
from fastapi_cache2.backends.redis import RedisBackend
from redis import asyncio as aioredis

# Configure in startup event
```

### 3. CDN Setup

```nginx
# Serve static files from CDN
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

---

## Security Hardening

### 1. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/complaint/create")
@limiter.limit("100/hour")
async def create_complaint(...):
    ...
```

### 2. HTTPS Enforcement

```nginx
# Already configured in Nginx setup
add_header Strict-Transport-Security "max-age=31536000" always;
```

### 3. Input Sanitization

- Already implemented in enhanced validation service
- All inputs are sanitized before processing
- SQL injection and XSS attacks prevented

---

**Last Updated:** January 2024
**Version:** 1.0.0
