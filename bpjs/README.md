# 🏥 BPJS JKN AI Assistant - Complete System

Sistem AI terpadu untuk BPJS JKN dengan 2 aplikasi utama:
1. **Conversational AI Assistant** - Chat assistant untuk informasi JKN
2. **OCR Claim Service** - Ekstraksi otomatis data struk medis dengan Gemini Vision AI

---

## 📋 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [AWS EC2 Deployment Architecture](#-aws-ec2-deployment-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Local Setup](#-local-setup)
- [AWS EC2 Deployment Guide](#-aws-ec2-deployment-guide)
- [Database Setup](#-database-setup)
- [Usage Guide](#-usage-guide)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)

---

## 🏗️ Architecture Overview

### Application Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BPJS JKN AI System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────┐      ┌────────────────────────┐    │
│  │   main.py (Port 8000)  │      │  main1.py (Port 8001)  │    │
│  │  Chat Assistant        │      │  OCR Claim Service     │    │
│  │                        │      │                        │    │
│  │  - FastAPI             │      │  - FastAPI             │    │
│  │  - LangGraph           │      │  - Gemini Vision OCR   │    │
│  │  - Gemini AI           │      │  - Smart Validation    │    │
│  │  - Session Management  │      │  - Approval Prediction │    │
│  └───────────┬────────────┘      └───────────┬────────────┘    │
│              │                                 │                 │
│              └────────────┬────────────────────┘                │
│                           │                                      │
│              ┌────────────▼──────────────┐                      │
│              │   Google Gemini API       │                      │
│              │   - Gemini 2.0 Flash Exp  │                      │
│              │   - Vision OCR            │                      │
│              └───────────────────────────┘                      │
│                           │                                      │
│              ┌────────────▼──────────────┐                      │
│              │   PostgreSQL Database     │                      │
│              │   (localhost:5432)        │                      │
│              │                           │                      │
│              │  ┌─────────────────────┐ │                      │
│              │  │ Tables:             │ │                      │
│              │  │ • peserta           │ │                      │
│              │  │ • klaim             │ │                      │
│              │  │ • faskes            │ │                      │
│              │  │ • ocr_log           │ │                      │
│              │  └─────────────────────┘ │                      │
│              └───────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## ☁️ AWS EC2 Deployment Architecture


![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/bpjs/static/arch.JPG)


### Architecture Components Explanation

| Component | Description |
|-----------|-------------|
| **EC2 Instance** | Ubuntu 22.04 server running both applications |
| **Public IP** | 13.212.146.114 - accessible from internet |
| **tmux Sessions** | 2 persistent sessions for both apps |
| **Security Group** | Firewall rules controlling inbound/outbound traffic |
| **PostgreSQL** | Local database on same EC2 instance |
| **Internet Gateway** | Allows public internet access |
| **External APIs** | Google Gemini for AI processing |

### Data Flow

```
User Browser
    │
    ▼
http://13.212.146.114:8000 (Chat)
http://13.212.146.114:8001 (OCR)
    │
    ▼
Internet Gateway
    │
    ▼
Security Group (Port 8000/8001 allowed)
    │
    ▼
EC2 Instance
    │
    ├──▶ main.py (tmux: bpjs-chat)
    │       │
    │       ├──▶ LangGraph Processing
    │       ├──▶ Google Gemini API
    │       └──▶ PostgreSQL (localhost:5432)
    │
    └──▶ main1.py (tmux: bpjs-ocr)
            │
            ├──▶ Gemini Vision OCR
            ├──▶ Image Processing
            └──▶ PostgreSQL (localhost:5432)
```

---

## ✨ Features

### 🤖 Application 1: Chat Assistant (`main.py`) - Port 8000

- ✅ **Conversational Interface** - Clean, responsive chat UI
- ✅ **Multi-Intent Support** - Registration, Claims, Complaints, General Info
- ✅ **LangGraph Workflow** - State-based conversation management
- ✅ **Quick Reply Buttons** - Fast user interactions
- ✅ **Session Persistence** - Maintains conversation context
- ✅ **Gemini AI Integration** - Powered by Google Gemini 2.0 Flash

**Access:** `http://13.212.146.114:8000`

### 📸 Application 2: OCR Claim Service (`main1.py`) - Port 8001

- ✅ **Gemini Vision OCR** - Automatic receipt data extraction
- ✅ **Drag & Drop Upload** - User-friendly file upload
- ✅ **Smart Validation** - Business rule-based claim validation
- ✅ **Approval Prediction** - AI-powered approval probability
- ✅ **PostgreSQL Storage** - All data persisted in database
- ✅ **Beautiful Results Table** - Clean data presentation
- ✅ **Real-time Processing** - 3-5 second OCR processing

**Access:** `http://13.212.146.114:8001`

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Backend** | FastAPI | 0.109.0 |
| **Language** | Python | 3.10+ |
| **AI Framework** | LangChain | 0.1.5 |
| **Workflow** | LangGraph | 0.0.20 |
| **LLM** | Google Gemini | 2.0 Flash Exp |
| **Vision AI** | Gemini Vision API | Latest |
| **Database** | PostgreSQL | 14+ |
| **Server** | Uvicorn | 0.27.0 |
| **Frontend** | HTML5, CSS3, JS | Native |
| **Cloud** | AWS EC2 | Ubuntu 22.04 |
| **Process Manager** | tmux | 3.x |

---

## 🚀 Local Setup

### Prerequisites

- Python 3.10 or higher
- PostgreSQL 14 or higher
- Google API Key (Gemini)
- LangChain API Key (optional)

### Installation Steps

1. **Clone/Navigate to project**
   ```bash
   cd d:\Langsmith-main\bpjs
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file**
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   LANGCHAIN_API_KEY=your_langchain_api_key_here
   ```

5. **Setup PostgreSQL Database**
   ```sql
   CREATE DATABASE bpjs;
   ALTER USER postgres WITH PASSWORD 'permataputihg101';
   ```

6. **Run Applications**
   ```bash
   # Terminal 1 - Chat Assistant
   python main.py

   # Terminal 2 - OCR Service
   python main1.py
   ```

7. **Access**
   - Chat: http://localhost:8000
   - OCR: http://localhost:8001

---

## ☁️ AWS EC2 Deployment Guide

### Complete Step-by-Step Deployment

#### Step 1: Connect to EC2

```bash
ssh -i fastapi.pem ubuntu@13.212.146.114
```

#### Step 2: System Update & Python Installation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and tools
sudo apt install python3 python3-pip python3-venv -y
```

#### Step 3: Install PostgreSQL

```bash
# Update package list
sudo apt update

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Start PostgreSQL
sudo systemctl start postgresql

# Enable on boot
sudo systemctl enable postgresql

# Verify running
sudo systemctl status postgresql
```

#### Step 4: Configure PostgreSQL Database

```bash
# Switch to postgres user
sudo -u postgres psql
```

Run these SQL commands:
```sql
-- Create database
CREATE DATABASE bpjs;

-- Set password
ALTER USER postgres WITH PASSWORD 'permataputihg101';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE bpjs TO postgres;

-- Exit
\q
```

#### Step 5: Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Create project directory
mkdir bpjs
cd bpjs
```

#### Step 6: Create Project Files

**requirements.txt:**
```bash
nano requirements.txt
```

Paste this content:
```
fastapi==0.109.0
uvicorn==0.27.0
python-dotenv==1.0.0
psycopg2-binary==2.9.9
langchain==0.1.5
langchain-google-genai==0.0.6
langchain-community==0.0.16
langchain-core==0.1.19
langgraph==0.0.20
pydantic==2.5.3
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**.env file:**
```bash
nano .env
```

Paste:
```
GOOGLE_API_KEY=your_google_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

**Upload application files:**
```bash
# Create main.py
nano main.py
# Copy-paste content from local main.py, save with Ctrl+X, Y, Enter

# Create main1.py
nano main1.py
# Copy-paste content from local main1.py, save with Ctrl+X, Y, Enter
```

#### Step 7: Run with tmux

**Install tmux:**
```bash
sudo apt install tmux -y
```

**Start Chat Assistant:**
```bash
# Create session
tmux new -s bpjs-chat

# Inside tmux:
source venv/bin/activate
cd bpjs
python main.py

# Detach: Press Ctrl+B, then D
```

**Start OCR Service:**
```bash
# Create session
tmux new -s bpjs-ocr

# Inside tmux:
source venv/bin/activate
cd bpjs
python main1.py

# Detach: Press Ctrl+B, then D
```

**Manage tmux:**
```bash
# List sessions
tmux ls

# Attach to session
tmux attach -t bpjs-chat
tmux attach -t bpjs-ocr

# Kill session
tmux kill-session -t bpjs-chat
```

#### Step 8: Configure AWS Security Group

In AWS Console → EC2 → Security Groups:

| Type | Protocol | Port Range | Source | Description |
|------|----------|------------|--------|-------------|
| SSH | TCP | 22 | Your IP | SSH access |
| Custom TCP | TCP | 8000 | 0.0.0.0/0 | Chat Assistant |
| Custom TCP | TCP | 8001 | 0.0.0.0/0 | OCR Service |

#### Step 9: Access Your Applications

- **Chat Assistant:** http://13.212.146.114:8000
- **OCR Service:** http://13.212.146.114:8001

---

## 🗄️ Database Setup

### Database Configuration

**File:** `main1.py` (Lines 24-30)

```python
DB_CONFIG = {
    "dbname": "bpjs",
    "user": "postgres",
    "password": "permataputihg101",
    "host": "localhost",
    "port": "5432"
}
```

### Tables Auto-Created

#### 1. `peserta` (Participants)
```sql
CREATE TABLE peserta (
    id SERIAL PRIMARY KEY,
    no_kartu VARCHAR(13) UNIQUE,
    nik VARCHAR(16),
    nama VARCHAR(200),
    tanggal_lahir DATE,
    jenis_kelamin VARCHAR(1),
    kelas_rawat INTEGER DEFAULT 3,
    status_peserta VARCHAR(20) DEFAULT 'AKTIF',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 2. `klaim` (Claims)
```sql
CREATE TABLE klaim (
    id SERIAL PRIMARY KEY,
    no_klaim VARCHAR(20) UNIQUE,
    no_kartu VARCHAR(13) REFERENCES peserta(no_kartu),
    tanggal_pelayanan DATE,
    nama_faskes VARCHAR(200),
    jenis_pelayanan VARCHAR(50),
    diagnosa TEXT,
    tindakan TEXT[],
    obat_diberikan TEXT[],
    total_biaya DECIMAL(15,2),
    biaya_disetujui DECIMAL(15,2),
    status_klaim VARCHAR(20) DEFAULT 'PENDING',
    approval_probability DECIMAL(3,2),
    warnings TEXT[],
    missing_documents TEXT[],
    dokumen_receipt TEXT,
    dokumen_resume_medis TEXT,
    dokumen_rujukan TEXT,
    keterangan TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 3. `faskes` (Healthcare Facilities)
```sql
CREATE TABLE faskes (
    id SERIAL PRIMARY KEY,
    kode_faskes VARCHAR(10) UNIQUE,
    nama_faskes VARCHAR(200),
    jenis_faskes VARCHAR(50),
    kelas VARCHAR(10),
    kerjasama_bpjs BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 4. `ocr_log` (OCR Processing Logs)
```sql
CREATE TABLE ocr_log (
    id SERIAL PRIMARY KEY,
    image_hash VARCHAR(64),
    extraction_type VARCHAR(50),
    extracted_data JSONB,
    processing_time_ms INTEGER,
    success BOOLEAN,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 📖 Usage Guide

### Chat Assistant Usage

1. **Open:** http://13.212.146.114:8000

2. **Interact:**
   - Click quick reply buttons
   - Type your questions
   - Get instant responses

3. **Examples:**
   ```
   "Bagaimana cara daftar JKN?"
   "Saya mau mengajukan klaim"
   "Cara mengajukan pengaduan"
   ```

### OCR Service Usage

1. **Open:** http://13.212.146.114:8001

2. **Process:**
   - Enter No. Kartu JKN: `0001234567890`
   - Upload medical receipt (JPG/PNG)
   - Click "Proses OCR & Buat Klaim"
   - View results in 3-5 seconds

3. **Results Show:**
   - ✅ Claim number generated
   - 📊 Approval probability (0-100%)
   - 💰 Estimated approved amount
   - 📋 All extracted data in table
   - ⚠️ Validation warnings

---

## 🔌 API Documentation

### POST `/api/chat`

**Request:**
```json
{
  "message": "Cara daftar JKN",
  "session_id": "optional",
  "action": "optional",
  "value": "optional"
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "response": {
    "role": "assistant",
    "content": "Response text",
    "quick_replies": [...]
  }
}
```

### POST `/api/process_receipt`

**Request:** (multipart/form-data)
```
receipt_image: File
no_kartu: "0001234567890"
```

**Response:**
```json
{
  "success": true,
  "no_klaim": "KLM20250108ABC123",
  "extracted_data": {...},
  "validation": {
    "approval_probability": 0.9,
    "estimated_approval_amount": 722500
  }
}
```

### GET `/api/statistics`

```json
{
  "total_claims": 15,
  "pending_claims": 8,
  "average_approval_probability": 0.853,
  "total_claimed_amount": 45750000
}
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Cannot Connect to Database

**Error:** `could not connect to server`

**Solution:**
```bash
sudo systemctl status postgresql
sudo systemctl start postgresql
```

#### 2. Column Does Not Exist

**Error:** `column "diagnosa" does not exist`

**Solution:**
```sql
-- Connect and drop tables
sudo -u postgres psql -d bpjs
DROP TABLE IF EXISTS klaim CASCADE;
\q

-- Restart application (will recreate)
python main1.py
```

#### 3. Port Already in Use

```bash
# Find and kill process
lsof -i :8000
kill -9 <PID>
```

#### 4. tmux Session Lost

```bash
# List sessions
tmux ls

# Recreate if needed
tmux new -s bpjs-chat
source venv/bin/activate
cd bpjs
python main.py
```

#### 5. Cannot Access from Browser

**Check:**
1. Security Group has port 8000/8001 open
2. Application is running: `tmux ls`
3. Firewall allows traffic: `sudo ufw status`

---

## 📊 Monitoring & Logs

### View Application Logs

```bash
# Attach to tmux session
tmux attach -t bpjs-chat
tmux attach -t bpjs-ocr

# View logs in real-time
```

### Database Queries

```bash
# Connect to database
sudo -u postgres psql -d bpjs

# View claims
SELECT no_klaim, nama_faskes, total_biaya, status_klaim
FROM klaim
ORDER BY created_at DESC
LIMIT 10;

# View statistics
SELECT COUNT(*) as total,
       AVG(approval_probability) as avg_approval
FROM klaim;
```

---

## 🔒 Security Best Practices

### Production Recommendations

1. **Change Database Password**
   ```sql
   ALTER USER postgres WITH PASSWORD 'your_secure_password';
   ```

2. **Use Environment Variables**
   ```python
   import os
   DB_CONFIG = {
       "password": os.getenv("DB_PASSWORD")
   }
   ```

3. **Restrict Security Group**
   - Only allow SSH from your IP
   - Consider using VPN for database access

4. **Enable HTTPS**
   - Install Nginx
   - Use Let's Encrypt SSL

5. **Regular Backups**
   ```bash
   pg_dump -U postgres bpjs > backup_$(date +%Y%m%d).sql
   ```

---

## 📝 Project Structure

```
~/bpjs/
├── main.py              # Chat Assistant (Port 8000)
├── main1.py             # OCR Claim Service (Port 8001)
├── requirements.txt     # Python dependencies
├── .env                 # API keys (DO NOT COMMIT)
└── README.md           # This file

Database: bpjs
├── peserta (table)
├── klaim (table)
├── faskes (table)
└── ocr_log (table)
```

---

## 🎯 Quick Reference

### Essential Commands

```bash
# SSH Connect
ssh -i fastapi.pem ubuntu@13.212.146.114

# tmux Management
tmux ls                          # List sessions
tmux attach -t bpjs-chat        # Attach to session
Ctrl+B, D                       # Detach from session
tmux kill-session -t bpjs-chat  # Kill session

# Application Control
source venv/bin/activate        # Activate venv
python main.py                  # Start chat app
python main1.py                 # Start OCR app

# Database
sudo systemctl status postgresql # Check status
sudo -u postgres psql -d bpjs   # Connect to DB
\dt                             # List tables
\q                              # Exit psql

# Process Management
ps aux | grep python            # Find processes
kill -9 <PID>                   # Kill process
```

### Access URLs

| Service | URL |
|---------|-----|
| Chat Assistant | http://13.212.146.114:8000 |
| OCR Service | http://13.212.146.114:8001 |
| SSH Access | `ssh -i fastapi.pem ubuntu@13.212.146.114` |

---

## 🆘 Support

**Common Solutions:**

1. **App not accessible** → Check Security Group & tmux sessions
2. **Database errors** → Restart PostgreSQL
3. **OCR fails** → Check GOOGLE_API_KEY in .env
4. **Session lost** → Recreate tmux session

**Deployment Checklist:**

- [ ] EC2 instance running
- [ ] PostgreSQL installed and running
- [ ] Python dependencies installed
- [ ] .env file configured
- [ ] Security Group ports open (22, 8000, 8001)
- [ ] tmux sessions running both apps
- [ ] Applications accessible via browser

---

**Version:** 1.0.0
**Last Updated:** January 2025
**Deployed On:** AWS EC2 (13.212.146.114)
**Database:** PostgreSQL 14
**Python:** 3.10+
