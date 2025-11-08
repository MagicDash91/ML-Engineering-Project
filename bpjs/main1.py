"""
BPJS JKN - OCR Claim Assistant Service
Extracts medical receipt data using Gemini Vision and stores in PostgreSQL
"""

import os
import json
import base64
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DB_CONFIG = {
    "dbname": "bpjs",
    "user": "postgres",
    "password": "permataputihg101",
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    """Create PostgreSQL database connection"""
    return psycopg2.connect(**DB_CONFIG)

def init_database():
    """Initialize PostgreSQL database schema"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table: Peserta (Participants)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS peserta (
            id SERIAL PRIMARY KEY,
            no_kartu VARCHAR(13) UNIQUE,
            nik VARCHAR(16),
            nama VARCHAR(200),
            tanggal_lahir DATE,
            jenis_kelamin VARCHAR(1),
            kelas_rawat INTEGER DEFAULT 3,
            status_peserta VARCHAR(20) DEFAULT 'AKTIF',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table: Klaim (Claims)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS klaim (
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
        )
    """)

    # Table: OCR Processing Log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ocr_log (
            id SERIAL PRIMARY KEY,
            image_hash VARCHAR(64),
            extraction_type VARCHAR(50),
            extracted_data JSONB,
            processing_time_ms INTEGER,
            success BOOLEAN,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table: Faskes (Healthcare Facilities)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faskes (
            id SERIAL PRIMARY KEY,
            kode_faskes VARCHAR(10) UNIQUE,
            nama_faskes VARCHAR(200),
            jenis_faskes VARCHAR(50),
            kelas VARCHAR(10),
            kerjasama_bpjs BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add missing columns if tables already exist (migration)
    try:
        # Fix faskes table
        cursor.execute("""
            ALTER TABLE faskes ADD COLUMN IF NOT EXISTS kerjasama_bpjs BOOLEAN DEFAULT TRUE
        """)
        cursor.execute("""
            ALTER TABLE faskes ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        """)

        # Fix klaim table - add all potentially missing columns
        cursor.execute("""
            ALTER TABLE klaim ADD COLUMN IF NOT EXISTS diagnosa TEXT
        """)
        cursor.execute("""
            ALTER TABLE klaim ADD COLUMN IF NOT EXISTS tindakan TEXT[]
        """)
        cursor.execute("""
            ALTER TABLE klaim ADD COLUMN IF NOT EXISTS obat_diberikan TEXT[]
        """)
        cursor.execute("""
            ALTER TABLE klaim ADD COLUMN IF NOT EXISTS approval_probability DECIMAL(3,2)
        """)
        cursor.execute("""
            ALTER TABLE klaim ADD COLUMN IF NOT EXISTS warnings TEXT[]
        """)
        cursor.execute("""
            ALTER TABLE klaim ADD COLUMN IF NOT EXISTS missing_documents TEXT[]
        """)
        cursor.execute("""
            ALTER TABLE klaim ADD COLUMN IF NOT EXISTS dokumen_receipt TEXT
        """)
        cursor.execute("""
            ALTER TABLE klaim ADD COLUMN IF NOT EXISTS dokumen_resume_medis TEXT
        """)
        cursor.execute("""
            ALTER TABLE klaim ADD COLUMN IF NOT EXISTS dokumen_rujukan TEXT
        """)
        cursor.execute("""
            ALTER TABLE klaim ADD COLUMN IF NOT EXISTS keterangan TEXT
        """)
        cursor.execute("""
            ALTER TABLE klaim ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        """)

        conn.commit()
        print("✅ Database migration completed")
    except Exception as e:
        print(f"⚠️  Migration note: {e}")
        conn.rollback()

    # Insert sample peserta for testing
    cursor.execute("""
        INSERT INTO peserta (no_kartu, nik, nama, tanggal_lahir, jenis_kelamin, kelas_rawat)
        VALUES
        ('0001234567890', '3201234567890001', 'John Doe', '1990-01-15', 'L', 2),
        ('0009876543210', '3201987654321002', 'Jane Smith', '1985-05-20', 'P', 1)
        ON CONFLICT (no_kartu) DO NOTHING
    """)

    # Insert sample faskes
    cursor.execute("""
        INSERT INTO faskes (kode_faskes, nama_faskes, jenis_faskes, kelas, kerjasama_bpjs)
        VALUES
        ('RS001', 'RSUD Kota Jakarta', 'RUMAH SAKIT', 'A', TRUE),
        ('RS002', 'RS Siloam Hospitals', 'RUMAH SAKIT', 'A', TRUE),
        ('PK001', 'Puskesmas Menteng', 'PUSKESMAS', 'C', TRUE),
        ('KL001', 'Klinik Sehat Sentosa', 'KLINIK', 'B', TRUE)
        ON CONFLICT (kode_faskes) DO NOTHING
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ PostgreSQL Database initialized successfully")

# Initialize database
init_database()

# ============================================================================
# GEMINI VISION OCR
# ============================================================================

llm_vision = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)

def extract_receipt_data(image_base64: str) -> Dict:
    """
    Extract comprehensive medical receipt data using Gemini Vision
    """
    start_time = datetime.now()

    prompt = """
Anda adalah sistem OCR untuk struk/kwitansi pelayanan kesehatan Indonesia.

Ekstrak SEMUA informasi berikut dari gambar struk dengan teliti:

**INFORMASI FASKES:**
- Nama Rumah Sakit/Klinik/Puskesmas
- Alamat faskes (jika ada)
- Nomor telepon faskes (jika ada)

**INFORMASI PASIEN:**
- Nama Pasien
- Nomor Rekam Medis (jika ada)
- Umur/Tanggal Lahir (jika ada)

**INFORMASI PELAYANAN:**
- Tanggal Pelayanan (format: YYYY-MM-DD)
- Jenis Pelayanan (Rawat Jalan/Rawat Inap/IGD/Operasi)
- Nomor Kwitansi/Invoice

**DETAIL MEDIS:**
- Diagnosa/Keluhan utama
- Tindakan medis yang dilakukan (list)
- Obat-obatan yang diberikan (list dengan nama obat)
- Pemeriksaan lab/radiologi (jika ada)

**BIAYA:**
- Biaya Konsultasi Dokter
- Biaya Tindakan
- Biaya Obat
- Biaya Pemeriksaan Penunjang
- Biaya Admin/lainnya
- Total Biaya (PENTING)

**INFORMASI TAMBAHAN:**
- Nama Dokter yang menangani
- Spesialisasi dokter (jika ada)
- Ruang/Poli (jika ada)

Return dalam format JSON (HANYA JSON, tanpa markdown atau teks lain):
{
    "nama_faskes": "...",
    "alamat_faskes": "..." or null,
    "telepon_faskes": "..." or null,
    "nama_pasien": "...",
    "no_rekam_medis": "..." or null,
    "tanggal_pelayanan": "YYYY-MM-DD",
    "jenis_pelayanan": "RAWAT JALAN/RAWAT INAP/IGD/OPERASI",
    "no_kwitansi": "...",
    "diagnosa": "...",
    "tindakan": ["tindakan 1", "tindakan 2"],
    "obat": ["nama obat 1", "nama obat 2"],
    "pemeriksaan_penunjang": ["lab 1", "radiologi 1"] or [],
    "biaya_konsultasi": 0,
    "biaya_tindakan": 0,
    "biaya_obat": 0,
    "biaya_penunjang": 0,
    "biaya_admin": 0,
    "total_biaya": 0,
    "nama_dokter": "..." or null,
    "spesialisasi_dokter": "..." or null,
    "ruang_poli": "..." or null
}

PENTING:
- Jika field tidak terbaca, isi dengan null atau [] untuk array
- Total biaya harus dalam angka (integer), tanpa titik/koma
- Tanggal harus format YYYY-MM-DD
"""

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_base64}"
            }
        ]
    )

    try:
        response = llm_vision.invoke([message])

        # Extract JSON from response
        content = response.content.strip()

        # Clean markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        extracted_data = json.loads(content)

        # Sanitize extracted data - ensure safe defaults
        sanitized_data = {
            'nama_faskes': extracted_data.get('nama_faskes') or '',
            'alamat_faskes': extracted_data.get('alamat_faskes'),
            'telepon_faskes': extracted_data.get('telepon_faskes'),
            'nama_pasien': extracted_data.get('nama_pasien') or '',
            'no_rekam_medis': extracted_data.get('no_rekam_medis'),
            'tanggal_pelayanan': extracted_data.get('tanggal_pelayanan'),
            'jenis_pelayanan': extracted_data.get('jenis_pelayanan') or 'RAWAT JALAN',
            'no_kwitansi': extracted_data.get('no_kwitansi'),
            'diagnosa': extracted_data.get('diagnosa') or '',
            'tindakan': extracted_data.get('tindakan') or [],
            'obat': extracted_data.get('obat') or [],
            'pemeriksaan_penunjang': extracted_data.get('pemeriksaan_penunjang') or [],
            'biaya_konsultasi': extracted_data.get('biaya_konsultasi') or 0,
            'biaya_tindakan': extracted_data.get('biaya_tindakan') or 0,
            'biaya_obat': extracted_data.get('biaya_obat') or 0,
            'biaya_penunjang': extracted_data.get('biaya_penunjang') or 0,
            'biaya_admin': extracted_data.get('biaya_admin') or 0,
            'total_biaya': extracted_data.get('total_biaya') or 0,
            'nama_dokter': extracted_data.get('nama_dokter'),
            'spesialisasi_dokter': extracted_data.get('spesialisasi_dokter'),
            'ruang_poli': extracted_data.get('ruang_poli')
        }

        # Calculate processing time
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        # Log OCR processing
        log_ocr_extraction(
            image_base64=image_base64,
            extraction_type="medical_receipt",
            extracted_data=sanitized_data,
            processing_time_ms=processing_time,
            success=True
        )

        return sanitized_data

    except Exception as e:
        print(f"❌ OCR Error: {e}")

        # Log failed OCR
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        log_ocr_extraction(
            image_base64=image_base64,
            extraction_type="medical_receipt",
            extracted_data={},
            processing_time_ms=processing_time,
            success=False,
            error_message=str(e)
        )

        return {}

def log_ocr_extraction(image_base64: str, extraction_type: str, extracted_data: dict,
                       processing_time_ms: int, success: bool, error_message: str = None):
    """Log OCR extraction to database"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create simple hash of image
    import hashlib
    image_hash = hashlib.sha256(image_base64[:1000].encode()).hexdigest()

    cursor.execute("""
        INSERT INTO ocr_log (image_hash, extraction_type, extracted_data, processing_time_ms, success, error_message)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (image_hash, extraction_type, json.dumps(extracted_data), processing_time_ms, success, error_message))

    conn.commit()
    cursor.close()
    conn.close()

# ============================================================================
# CLAIM VALIDATION & PROCESSING
# ============================================================================

def validate_claim(claim_data: Dict) -> Dict:
    """
    Validate claim against business rules
    Returns validation result with approval probability and warnings
    """
    # Handle None values safely
    total_biaya = claim_data.get('total_biaya') or 0
    if not isinstance(total_biaya, (int, float)):
        try:
            total_biaya = float(total_biaya)
        except:
            total_biaya = 0

    validation = {
        "is_valid": True,
        "warnings": [],
        "missing_documents": [],
        "approval_probability": 1.0,
        "estimated_approval_amount": total_biaya
    }

    tanggal_pelayanan = claim_data.get('tanggal_pelayanan')

    # Rule 1: Check if biaya exceeds threshold
    if total_biaya > 10000000:
        validation['warnings'].append("⚠️ Biaya melebihi Rp 10 juta, memerlukan verifikasi tambahan")
        validation['approval_probability'] -= 0.2

    if total_biaya > 50000000:
        validation['warnings'].append("⚠️ Biaya sangat tinggi (>50 juta), memerlukan review komite medis")
        validation['approval_probability'] -= 0.3

    # Rule 2: Check date - claim must be submitted within 30 days
    if tanggal_pelayanan:
        try:
            pelayanan_date = datetime.strptime(tanggal_pelayanan, '%Y-%m-%d')
            days_since = (datetime.now() - pelayanan_date).days

            if days_since > 30:
                validation['warnings'].append(f"⚠️ Pengajuan terlambat ({days_since} hari sejak pelayanan)")
                validation['approval_probability'] -= 0.15

            if days_since < 0:
                validation['warnings'].append("❌ Tanggal pelayanan di masa depan - tidak valid")
                validation['is_valid'] = False
                validation['approval_probability'] = 0

        except:
            validation['warnings'].append("⚠️ Format tanggal tidak valid")
            validation['approval_probability'] -= 0.1

    # Rule 3: Check required documents
    required_docs = ['struk_pembayaran']

    # Additional docs based on service type
    jenis_pelayanan = claim_data.get('jenis_pelayanan') or ''
    jenis_pelayanan = str(jenis_pelayanan).upper()

    if 'INAP' in jenis_pelayanan or 'OPERASI' in jenis_pelayanan:
        required_docs.extend(['resume_medis', 'surat_rujukan'])

    if total_biaya > 5000000:
        required_docs.append('resume_medis')

    # Check which documents are missing
    existing_docs = claim_data.get('dokumen', [])
    for doc in required_docs:
        if doc not in existing_docs:
            validation['missing_documents'].append(doc)
            validation['approval_probability'] -= 0.1

    if validation['missing_documents']:
        validation['is_valid'] = False
        validation['warnings'].append(f"📎 Dokumen yang harus dilengkapi: {', '.join(validation['missing_documents'])}")

    # Rule 4: Check faskes partnership
    nama_faskes = claim_data.get('nama_faskes') or ''
    nama_faskes = str(nama_faskes).strip()
    if nama_faskes:
        if not check_faskes_partnership(nama_faskes):
            validation['warnings'].append("⚠️ Faskes belum terdaftar dalam sistem BPJS")
            validation['approval_probability'] -= 0.2

    # Rule 5: Estimate approved amount (some costs may not be covered)
    # BPJS typically covers 80-100% depending on class and service
    if validation['approval_probability'] > 0.5:
        coverage_rate = 0.85  # 85% average coverage
        validation['estimated_approval_amount'] = int(total_biaya * coverage_rate)
    else:
        validation['estimated_approval_amount'] = 0

    # Ensure probability is between 0 and 1
    validation['approval_probability'] = max(0, min(1, validation['approval_probability']))

    return validation

def check_faskes_partnership(nama_faskes: str) -> bool:
    """Check if healthcare facility has partnership with BPJS"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT kerjasama_bpjs FROM faskes
        WHERE LOWER(nama_faskes) LIKE LOWER(%s) AND kerjasama_bpjs = TRUE
        LIMIT 1
    """, (f"%{nama_faskes}%",))

    result = cursor.fetchone()
    cursor.close()
    conn.close()

    return result is not None

def create_claim(no_kartu: str, receipt_data: Dict, dokumen_receipt_base64: str = None) -> Dict:
    """
    Create new claim in database
    Returns claim details with validation
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Generate claim number
    no_klaim = f"KLM{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:6].upper()}"

    # Prepare claim data with document check
    claim_data = {
        'no_kartu': no_kartu,
        'tanggal_pelayanan': receipt_data.get('tanggal_pelayanan'),
        'nama_faskes': receipt_data.get('nama_faskes'),
        'jenis_pelayanan': receipt_data.get('jenis_pelayanan', 'RAWAT JALAN'),
        'total_biaya': receipt_data.get('total_biaya', 0),
        'dokumen': ['struk_pembayaran'] if dokumen_receipt_base64 else []
    }

    # Validate claim
    validation = validate_claim(claim_data)

    # Insert claim to database
    cursor.execute("""
        INSERT INTO klaim
        (no_klaim, no_kartu, tanggal_pelayanan, nama_faskes, jenis_pelayanan,
         diagnosa, tindakan, obat_diberikan, total_biaya, biaya_disetujui,
         status_klaim, approval_probability, warnings, missing_documents, dokumen_receipt)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING *
    """, (
        no_klaim,
        no_kartu,
        receipt_data.get('tanggal_pelayanan'),
        receipt_data.get('nama_faskes'),
        receipt_data.get('jenis_pelayanan', 'RAWAT JALAN'),
        receipt_data.get('diagnosa'),
        receipt_data.get('tindakan', []),
        receipt_data.get('obat', []),
        receipt_data.get('total_biaya', 0),
        validation['estimated_approval_amount'],
        'PENDING' if validation['is_valid'] else 'INCOMPLETE',
        validation['approval_probability'],
        validation['warnings'],
        validation['missing_documents'],
        dokumen_receipt_base64[:100] if dokumen_receipt_base64 else None  # Store hash/reference
    ))

    claim_result = cursor.fetchone()

    conn.commit()
    cursor.close()
    conn.close()

    return {
        'claim': dict(claim_result),
        'validation': validation
    }

def get_claim_by_number(no_klaim: str) -> Optional[Dict]:
    """Get claim details by claim number"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("SELECT * FROM klaim WHERE no_klaim = %s", (no_klaim,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    return dict(result) if result else None

def update_claim_documents(no_klaim: str, document_type: str, document_base64: str) -> bool:
    """Update claim with additional documents"""
    conn = get_db_connection()
    cursor = conn.cursor()

    document_field_map = {
        'resume_medis': 'dokumen_resume_medis',
        'surat_rujukan': 'dokumen_rujukan'
    }

    field = document_field_map.get(document_type)
    if not field:
        return False

    cursor.execute(f"""
        UPDATE klaim
        SET {field} = %s, updated_at = CURRENT_TIMESTAMP
        WHERE no_klaim = %s
    """, (document_base64[:100], no_klaim))  # Store hash/reference

    conn.commit()
    cursor.close()
    conn.close()

    return True

def get_claim_statistics() -> Dict:
    """Get claim statistics from database"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Total claims
    cursor.execute("SELECT COUNT(*) as total FROM klaim")
    total_claims = cursor.fetchone()['total']

    # Pending claims
    cursor.execute("SELECT COUNT(*) as total FROM klaim WHERE status_klaim = 'PENDING'")
    pending_claims = cursor.fetchone()['total']

    # Average approval probability
    cursor.execute("SELECT AVG(approval_probability) as avg_prob FROM klaim WHERE approval_probability IS NOT NULL")
    avg_prob = cursor.fetchone()['avg_prob'] or 0

    # Total claimed amount
    cursor.execute("SELECT SUM(total_biaya) as total FROM klaim")
    total_claimed = cursor.fetchone()['total'] or 0

    cursor.close()
    conn.close()

    return {
        'total_claims': total_claims,
        'pending_claims': pending_claims,
        'average_approval_probability': float(avg_prob),
        'total_claimed_amount': float(total_claimed)
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_currency(amount: float) -> str:
    """Format amount as Indonesian Rupiah"""
    return f"Rp {amount:,.0f}".replace(",", ".")

def get_document_name(doc_code: str) -> str:
    """Get human-readable document name"""
    doc_names = {
        'struk_pembayaran': 'Struk Pembayaran/Kwitansi',
        'resume_medis': 'Resume Medis',
        'surat_rujukan': 'Surat Rujukan Dokter'
    }
    return doc_names.get(doc_code, doc_code)

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="BPJS JKN - OCR Claim Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML page"""
    return HTML_TEMPLATE

@app.post("/api/process_receipt")
async def process_receipt(
    receipt_image: UploadFile = File(...),
    no_kartu: str = Form(...)
):
    """
    Process medical receipt image:
    1. Extract data using OCR
    2. Validate claim
    3. Store in database
    4. Return results
    """
    try:
        # Read and encode image
        image_data = await receipt_image.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Extract data using Gemini Vision OCR
        receipt_data = extract_receipt_data(image_base64)

        if not receipt_data:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Gagal mengekstrak data dari gambar. Pastikan foto jelas dan terbaca."
                }
            )

        # Create claim in database
        result = create_claim(no_kartu, receipt_data, image_base64)

        return {
            "success": True,
            "no_klaim": result['claim']['no_klaim'],
            "extracted_data": receipt_data,
            "validation": result['validation'],
            "claim_status": result['claim']['status_klaim']
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Terjadi kesalahan: {str(e)}"
            }
        )

@app.get("/api/statistics")
async def get_statistics():
    """Get claim statistics"""
    return get_claim_statistics()

@app.get("/api/claims")
async def get_all_claims():
    """Get all claims from database"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT no_klaim, no_kartu, tanggal_pelayanan, nama_faskes,
               jenis_pelayanan, total_biaya, status_klaim,
               approval_probability, created_at
        FROM klaim
        ORDER BY created_at DESC
        LIMIT 50
    """)

    claims = [dict(row) for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    return claims

# ============================================================================
# HTML FRONTEND
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BPJS JKN - OCR Claim Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #00a651;
            --secondary-color: #005a2c;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #00a651 0%, #005a2c 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .main-container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .header {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            text-align: center;
        }

        .header h1 {
            color: var(--primary-color);
            margin: 0;
            font-size: 32px;
            font-weight: bold;
        }

        .header p {
            color: #666;
            margin: 10px 0 0 0;
        }

        .upload-section {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }

        .upload-area {
            border: 3px dashed var(--primary-color);
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }

        .upload-area:hover {
            background: #f0f9f4;
            border-color: var(--secondary-color);
        }

        .upload-area.dragover {
            background: #e6f5ed;
            border-color: var(--secondary-color);
            transform: scale(1.02);
        }

        .preview-image {
            max-width: 100%;
            max-height: 300px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .result-section {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            display: none;
        }

        .result-section.show {
            display: block;
        }

        .table-container {
            overflow-x: auto;
            margin-top: 20px;
        }

        .data-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
        }

        .data-table th {
            background: var(--primary-color);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }

        .data-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }

        .data-table tr:hover {
            background: #f8f9fa;
        }

        .badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }

        .badge-success {
            background: #28a745;
            color: white;
        }

        .badge-warning {
            background: #ffc107;
            color: #333;
        }

        .badge-danger {
            background: #dc3545;
            color: white;
        }

        .btn-primary {
            background: var(--primary-color);
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-weight: 600;
            transition: all 0.3s;
        }

        .btn-primary:hover {
            background: var(--secondary-color);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,166,81,0.3);
        }

        .loading-spinner {
            display: none;
            text-align: center;
            padding: 40px;
        }

        .loading-spinner.show {
            display: block;
        }

        .spinner-border {
            width: 3rem;
            height: 3rem;
            border-color: var(--primary-color);
            border-right-color: transparent;
        }

        .alert {
            border-radius: 10px;
            padding: 15px 20px;
            margin: 20px 0;
        }

        .validation-box {
            background: #f8f9fa;
            border-left: 4px solid var(--primary-color);
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }

        .validation-box h5 {
            color: var(--secondary-color);
            margin-bottom: 10px;
        }

        .probability-bar {
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }

        .probability-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #00a651);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 1s ease;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="header">
            <h1>🏥 BPJS JKN - OCR Claim Assistant</h1>
            <p>Upload struk pelayanan kesehatan untuk otomatis membuat klaim</p>
        </div>

        <div class="upload-section">
            <h3>📸 Upload Struk Pelayanan Kesehatan</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="mb-3">
                    <label for="no_kartu" class="form-label">No. Kartu JKN:</label>
                    <input type="text" class="form-control" id="no_kartu" name="no_kartu"
                           placeholder="Contoh: 0001234567890" required maxlength="13">
                    <small class="text-muted">Masukkan nomor kartu JKN BPJS (13 digit)</small>
                </div>

                <div class="upload-area" id="uploadArea">
                    <input type="file" id="receiptImage" name="receipt_image"
                           accept="image/*" style="display: none;" required>
                    <div id="uploadPrompt">
                        <h4>📁 Klik atau Drag & Drop</h4>
                        <p>Upload foto struk pelayanan kesehatan</p>
                        <p class="text-muted">Format: JPG, PNG (Max 5MB)</p>
                    </div>
                    <div id="imagePreview"></div>
                </div>

                <div class="text-center mt-3">
                    <button type="submit" class="btn btn-primary btn-lg">
                        🔍 Proses OCR & Buat Klaim
                    </button>
                </div>
            </form>

            <div class="loading-spinner" id="loadingSpinner">
                <div class="spinner-border" role="status"></div>
                <p class="mt-3">Memproses gambar dengan AI...</p>
            </div>
        </div>

        <div class="result-section" id="resultSection">
            <h3>📊 Hasil Ekstraksi Data</h3>

            <div id="claimInfo" class="alert alert-success">
                <strong>✅ Klaim berhasil dibuat!</strong><br>
                No. Klaim: <strong id="claimNumber"></strong>
            </div>

            <div class="validation-box">
                <h5>⚖️ Validasi Klaim</h5>
                <div>
                    <strong>Probabilitas Persetujuan:</strong>
                    <div class="probability-bar">
                        <div class="probability-fill" id="probabilityBar">0%</div>
                    </div>
                </div>
                <div class="mt-3">
                    <strong>Estimasi Biaya Disetujui:</strong>
                    <span id="approvedAmount" class="text-success"></span>
                </div>
                <div id="warningsBox" class="mt-3"></div>
            </div>

            <div class="table-container">
                <h5>📋 Data yang Diekstrak</h5>
                <table class="data-table">
                    <tbody id="extractedDataTable">
                    </tbody>
                </table>
            </div>

            <div class="text-center mt-4">
                <button class="btn btn-primary" onclick="resetForm()">
                    ➕ Upload Struk Lain
                </button>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('receiptImage');
        const uploadForm = document.getElementById('uploadForm');
        const loadingSpinner = document.getElementById('loadingSpinner');
        const resultSection = document.getElementById('resultSection');
        const imagePreview = document.getElementById('imagePreview');
        const uploadPrompt = document.getElementById('uploadPrompt');

        // Click to upload
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // File selected
        fileInput.addEventListener('change', handleFileSelect);

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect();
            }
        });

        function handleFileSelect() {
            const file = fileInput.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    uploadPrompt.style.display = 'none';
                    imagePreview.innerHTML = `<img src="${e.target.result}" class="preview-image" alt="Preview">`;
                };
                reader.readAsDataURL(file);
            }
        }

        // Form submission
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData(uploadForm);

            // Validate
            const noKartu = document.getElementById('no_kartu').value;
            if (noKartu.length !== 13) {
                alert('No. Kartu JKN harus 13 digit!');
                return;
            }

            if (!fileInput.files[0]) {
                alert('Pilih foto struk terlebih dahulu!');
                return;
            }

            // Show loading
            uploadForm.style.display = 'none';
            loadingSpinner.classList.add('show');
            resultSection.classList.remove('show');

            try {
                const response = await fetch('/api/process_receipt', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                loadingSpinner.classList.remove('show');

                if (data.success) {
                    displayResults(data);
                } else {
                    alert('Error: ' + data.error);
                    uploadForm.style.display = 'block';
                }

            } catch (error) {
                loadingSpinner.classList.remove('show');
                uploadForm.style.display = 'block';
                alert('Terjadi kesalahan: ' + error.message);
            }
        });

        function displayResults(data) {
            resultSection.classList.add('show');

            // Claim number
            document.getElementById('claimNumber').textContent = data.no_klaim;

            // Validation
            const probability = data.validation.approval_probability * 100;
            const probabilityBar = document.getElementById('probabilityBar');
            probabilityBar.style.width = probability + '%';
            probabilityBar.textContent = probability.toFixed(0) + '%';

            // Color based on probability
            if (probability >= 80) {
                probabilityBar.style.background = 'linear-gradient(90deg, #28a745, #00a651)';
            } else if (probability >= 50) {
                probabilityBar.style.background = 'linear-gradient(90deg, #ffc107, #ff9800)';
            } else {
                probabilityBar.style.background = 'linear-gradient(90deg, #dc3545, #c82333)';
            }

            // Approved amount
            document.getElementById('approvedAmount').textContent =
                'Rp ' + data.validation.estimated_approval_amount.toLocaleString('id-ID');

            // Warnings
            const warningsBox = document.getElementById('warningsBox');
            if (data.validation.warnings && data.validation.warnings.length > 0) {
                warningsBox.innerHTML = '<strong>⚠️ Peringatan:</strong><ul>' +
                    data.validation.warnings.map(w => `<li>${w}</li>`).join('') +
                    '</ul>';
            } else {
                warningsBox.innerHTML = '<div class="alert alert-success">✅ Tidak ada peringatan</div>';
            }

            // Extracted data table
            const tableBody = document.getElementById('extractedDataTable');
            const extractedData = data.extracted_data;

            const fieldLabels = {
                'nama_faskes': 'Nama Faskes',
                'nama_pasien': 'Nama Pasien',
                'tanggal_pelayanan': 'Tanggal Pelayanan',
                'jenis_pelayanan': 'Jenis Pelayanan',
                'diagnosa': 'Diagnosa',
                'tindakan': 'Tindakan',
                'obat': 'Obat',
                'total_biaya': 'Total Biaya',
                'nama_dokter': 'Nama Dokter',
                'no_kwitansi': 'No. Kwitansi'
            };

            let tableHTML = '';
            for (let [key, label] of Object.entries(fieldLabels)) {
                let value = extractedData[key];

                if (value === null || value === undefined) {
                    value = '-';
                } else if (Array.isArray(value)) {
                    value = value.join(', ');
                } else if (key === 'total_biaya') {
                    value = 'Rp ' + value.toLocaleString('id-ID');
                }

                tableHTML += `
                    <tr>
                        <th style="width: 30%">${label}</th>
                        <td>${value}</td>
                    </tr>
                `;
            }

            tableBody.innerHTML = tableHTML;
        }

        function resetForm() {
            uploadForm.reset();
            imagePreview.innerHTML = '';
            uploadPrompt.style.display = 'block';
            uploadForm.style.display = 'block';
            resultSection.classList.remove('show');
        }
    </script>
</body>
</html>
"""

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("🏥 BPJS JKN - OCR Claim Service")
    print("=" * 70)
    print("✅ Database initialized")
    print("✅ Gemini Vision OCR ready")
    print("=" * 70)
    print("📍 Open http://localhost:8001 in your browser")
    print("=" * 70)

    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
