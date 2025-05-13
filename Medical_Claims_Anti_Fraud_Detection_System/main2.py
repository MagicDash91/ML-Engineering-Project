from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import re
from paddleocr import PaddleOCR
from typing import List

ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
app = FastAPI()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Medical Receipt OCR</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
        }
        .upload-container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            padding: 2rem;
            margin-top: 2rem;
        }
        .result-container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            padding: 2rem;
            margin-top: 2rem;
        }
        .custom-file-upload {
            border: 2px dashed #dee2e6;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .custom-file-upload:hover {
            border-color: #0d6efd;
            background-color: #f8f9fa;
        }
        .table {
            background: white;
            border-radius: 10px;
            overflow: hidden;
        }
        .table thead {
            background-color: #0d6efd;
            color: white;
        }
        .btn-analyze {
            padding: 0.8rem 2rem;
            font-weight: 600;
        }
        .preview-image {
            max-width: 150px;
            max-height: 150px;
            object-fit: cover;
            border-radius: 8px;
            margin: 5px;
        }
        .image-preview-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 1rem;
        }
    </style>
</head>
<body>
<div class="container py-5">
    <div class="row justify-content-center">
        <div class="col-lg-10">
            <div class="upload-container">
                <h2 class="text-center mb-4">
                    <i class="fas fa-file-medical me-2"></i>
                    Medical Receipt OCR & Analysis
                </h2>
                <form method="post" action="/analyze" enctype="multipart/form-data" id="uploadForm">
                    <div class="custom-file-upload mb-4">
                        <input type="file" class="form-control" name="files" multiple accept="image/*" id="fileInput" required>
                        <p class="mt-2 text-muted">Drag and drop files here or click to select</p>
                    </div>
                    <div class="image-preview-container" id="imagePreview"></div>
                    <div class="text-center">
                        <button type="submit" class="btn btn-primary btn-analyze">
                            <i class="fas fa-search me-2"></i>Analyze Receipts
                        </button>
                    </div>
                </form>
            </div>
            %RESULT%
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
document.getElementById('fileInput').addEventListener('change', function(e) {
    const preview = document.getElementById('imagePreview');
    preview.innerHTML = '';
    
    for (const file of e.target.files) {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'preview-image';
                preview.appendChild(img);
            }
            reader.readAsDataURL(file);
        }
    }
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE.replace("%RESULT%", "")

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(files: List[UploadFile] = File(...)):
    results = []
    
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        preprocessed = preprocess_image(image_np)
        result = ocr_model.ocr(preprocessed, cls=True)
        text_lines = [line[1][0] for line in result[0]]
        extracted_fields = extract_receipt_fields(text_lines)
        results.append(extracted_fields)

    # Create HTML table for results
    result_html = """
    <div class="result-container">
        <h4 class="mb-4"><i class="fas fa-table me-2"></i>Analysis Results</h4>
        <div class="table-responsive">
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Receipt #</th>
                        <th>Name</th>
                        <th>Email</th>
                        <th>Address</th>
                        <th>Total Amount</th>
                        <th>Date</th>
                        <th>Hospital/Clinic</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for i, fields in enumerate(results, 1):
        result_html += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{fields['Name']}</td>
                        <td>{fields['Email']}</td>
                        <td>{fields['Address']}</td>
                        <td>{fields['Total Amount']}</td>
                        <td>{fields['Date']}</td>
                        <td>{fields['Hospital/Clinic']}</td>
                    </tr>
        """
    
    result_html += """
                </tbody>
            </table>
        </div>
    </div>
    """
    
    return HTML_TEMPLATE.replace("%RESULT%", result_html)

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_receipt_fields(lines):
    fields = {
        "Name": "",
        "Email": "",
        "Address": "",
        "Total Amount": "",
        "Date": "",
        "Hospital/Clinic": "",
    }

    for line in lines:
        lower = line.lower()

        # Email - Enhanced patterns
        if not fields["Email"]:
            match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", line)
            if match:
                fields["Email"] = match.group()

        # Name - Enhanced patterns for First and Last name only
        if not fields["Name"]:
            # First try to find standalone names (2-3 words)
            standalone_pattern = r"^([A-Za-z]+(?:\s+[A-Za-z]+){1,2})$"
            name_match = re.search(standalone_pattern, line)
            if name_match:
                name = name_match.group(1).strip()
                name = ' '.join(name.split())
                name = ' '.join(word.capitalize() for word in name.split())
                fields["Name"] = name
            else:
                # If no standalone name found, try other patterns
                name_patterns = [
                    # Pattern for "Name: First Last" format
                    r"(?:name|nama|patient|pasien)[:\-]?\s*([a-zA-Z]+(?:\s+[a-zA-Z]+){1,2})",
                    # Pattern for titles followed by name
                    r"(?:mr\.|mrs\.|ms\.|dr\.|dr\.|prof\.)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+){1,2})",
                    # Pattern for "To: First Last" format
                    r"(?:to|kepada|dear)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+){1,2})",
                    # Pattern for names with optional middle initial
                    r"([a-zA-Z]+(?:\s+[a-zA-Z]\.)?\s+[a-zA-Z]+)"
                ]
                for pattern in name_patterns:
                    name_match = re.search(pattern, lower)
                    if name_match:
                        name = name_match.group(1).strip()
                        name = ' '.join(name.split())
                        name = ' '.join(word.capitalize() for word in name.split())
                        fields["Name"] = name
                        break

        # Address - Enhanced patterns
        if not fields["Address"]:
            addr_patterns = [
                r"(address|alamat|lokasi|location)[:\-]?\s*(.+)",
                r"(street|jalan|jl\.|jln\.)\s*(.+)",
                r"(no\.|number|nomor)\s*(\d+[,\s]*[a-zA-Z0-9\s\-]+)",
                r"(rt\.|rw\.|block|blok)\s*(\d+[,\s]*[a-zA-Z0-9\s\-]+)"
            ]
            for pattern in addr_patterns:
                addr_match = re.search(pattern, lower)
                if addr_match:
                    fields["Address"] = addr_match.group(2) if len(addr_match.groups()) > 1 else addr_match.group(1)
                    fields["Address"] = fields["Address"].strip()
                    break

        # Total Amount - Enhanced patterns
        if not fields["Total Amount"]:
            amount_patterns = [
                r"(total|jumlah|total amount|total payment)[:\-]?\s*(rp|usd|idr|inr)?[\s:]*([0-9]+[.,]?[0-9]*)",
                r"(rp|usd|idr|inr)\s*([0-9]+[.,]?[0-9]*)",
                r"([0-9]+[.,]?[0-9]*)\s*(rp|usd|idr|inr)",
                r"(total|jumlah)[:\-]?\s*([0-9]+[.,]?[0-9]*)"
            ]
            for pattern in amount_patterns:
                amount_match = re.search(pattern, lower)
                if amount_match:
                    fields["Total Amount"] = amount_match.group(0)
                    break

        # Date - Enhanced patterns
        if not fields["Date"]:
            date_patterns = [
                r"(date|tanggal|issued|dikeluarkan)[:\-]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
                r"(date|tanggal|issued|dikeluarkan)[:\-]?\s*(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})",
                r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
                r"\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})\b"
            ]
            for pattern in date_patterns:
                date_match = re.search(pattern, lower)
                if date_match:
                    fields["Date"] = date_match.group(2) if len(date_match.groups()) > 1 else date_match.group(1)
                    break

        # Hospital/Clinic - Enhanced patterns
        if not fields["Hospital/Clinic"]:
            hospital_patterns = [
                r"(hospital|clinic|rs|rumah sakit|klinik|medical center|puskesmas)[:\-]?\s*(.+)",
                r"(dr\.|dr\.|prof\.)\s*([a-zA-Z\s]+)(?=\s*(?:hospital|clinic|rs|klinik))",
                r"(medical|healthcare|health care|health-care)\s*(center|centre|facility|institution)",
                r"(specialist|spesialis)\s*(clinic|klinik|practice|praktek)"
            ]
            for pattern in hospital_patterns:
                hospital_match = re.search(pattern, lower)
                if hospital_match:
                    fields["Hospital/Clinic"] = hospital_match.group(2) if len(hospital_match.groups()) > 1 else hospital_match.group(1)
                    fields["Hospital/Clinic"] = fields["Hospital/Clinic"].strip()
                    break

    return fields

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
