from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any, TypedDict, Optional
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool
from langchain.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import shutil
import uuid
import google.generativeai as genai
from markdown import markdown
from html import escape
from langchain_core.messages import AIMessage
import json
from datetime import datetime
import re

# ----------------------------
# 🔐 Load and Configure Keys
# ----------------------------
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

tavily_api_key = os.getenv("TAVILY_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# ----------------------------
# ⚙️ Graph State Definition
# ----------------------------
class GraphState(TypedDict, total=False):
    question: str
    tools: List[str]
    results: Dict[str, Any]
    final_answer: str
    document_type: str
    language: str

# ----------------------------
# 🔮 LLM & Tools
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_retries=3,
)

tools = {
    "Wikipedia": Tool(
        name="Wikipedia",
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
        description="Use for general legal concepts and historical legal information"
    ),
    "arXiv": Tool(
        name="arXiv",
        func=ArxivQueryRun(api_wrapper=ArxivAPIWrapper()).run,
        description="Use for academic legal research and jurisprudence studies"
    ),
    "TavilySearch": Tool(
        name="TavilySearch",
        func=TavilySearchResults(k=3, tavily_api_key=tavily_api_key).run,
        description="Use for current Indonesian legal updates, court decisions, and regulatory changes"
    ),
}

# ----------------------------
# 📋 Indonesian Law Prompt Template
# ----------------------------
INDONESIAN_LAW_PROMPT = PromptTemplate.from_template("""
Anda adalah seorang ahli hukum Indonesia yang sangat berpengalaman dengan keahlian mendalam dalam sistem hukum Indonesia. Berikut adalah konten dokumen hukum yang disediakan:

{text}

Tugas Anda adalah menjawab pertanyaan pengguna secara **detail, terstruktur, dan akurat** menggunakan hanya informasi dari konten yang disediakan.

**INSTRUKSI KHUSUS UNTUK DOKUMEN HUKUM INDONESIA:**

1. **Pemahaman Konteks Hukum**:
   - Identifikasi jenis dokumen hukum (UU, PP, Permen, Putusan Pengadilan, dll.)
   - Pahami hierarki peraturan perundang-undangan Indonesia
   - Perhatikan nomor, tahun, dan tanggal berlaku peraturan
   - Identifikasi pasal, ayat, dan huruf yang relevan

2. **Analisis Substansi Hukum**:
   - Jelaskan konsep hukum dengan bahasa yang mudah dipahami
   - Berikan konteks historis jika relevan
   - Identifikasi asas-asas hukum yang terkandung
   - Jelaskan implikasi praktis dari ketentuan hukum

3. **Jika pertanyaan melibatkan perbandingan hukum**:
   - Bandingkan ketentuan antar pasal atau peraturan
   - Identifikasi konsistensi atau kontradiksi
   - Jelaskan mana yang lebih tinggi dalam hierarki hukum
   - Berikan rekomendasi interpretasi yang tepat

4. **Jika pertanyaan meminta interpretasi hukum**:
   - Gunakan metode interpretasi sistematis, gramatikal, dan teleologis
   - Rujuk pada asas-asas hukum umum
   - Pertimbangkan maksud dan tujuan pembentuk undang-undang
   - Berikan alternatif interpretasi jika ada ambiguitas

5. **Jika pertanyaan tentang prosedur hukum**:
   - Jelaskan tahapan-tahapan yang harus dilalui
   - Identifikasi persyaratan, tenggang waktu, dan dokumen yang diperlukan
   - Jelaskan konsekuensi hukum dari setiap langkah
   - Berikan peringatan tentang sanksi atau akibat hukum

6. **Format Jawaban yang Diharapkan**:
   - Gunakan struktur yang jelas dengan heading dan subheading
   - Kutip pasal dan ayat yang relevan dengan format yang benar
   - Berikan penjelasan dalam bahasa Indonesia yang baik dan benar
   - Sertakan referensi lengkap (nama peraturan, nomor, tahun)
   - Gunakan bullet points untuk daftar persyaratan atau tahapan
   - Berikan kesimpulan dan rekomendasi praktis

7. **Catatan Penting**:
   - Jangan memberikan nasihat hukum yang bersifat final
   - Sarankan untuk berkonsultasi dengan ahli hukum untuk kasus spesifik
   - Jelaskan jika ada ketentuan yang memerlukan interpretasi lebih lanjut
   - Berikan disclaimer jika informasi mungkin sudah tidak aktual

**Pertanyaan Pengguna: {question}**

Berikan jawaban yang komprehensif dan mudah dipahami berdasarkan analisis mendalam terhadap dokumen hukum yang disediakan.
""")

# ----------------------------
# 🧠 Enhanced LangGraph Nodes
# ----------------------------
def classify(state):
    q = state["question"].lower()
    state["tools"] = []
    state["document_type"] = "general"
    state["language"] = "indonesian"

    # Detect Indonesian legal keywords
    legal_keywords = ["undang-undang", "peraturan", "pasal", "ayat", "hukum", "sanksi", "pidana", "perdata"]
    if any(word in q for word in legal_keywords):
        state["document_type"] = "legal"

    if any(w in q for w in ["sejarah", "latar belakang", "asal usul", "perkembangan"]):
        state["tools"].append("Wikipedia")
    if any(w in q for w in ["penelitian", "jurnal", "akademik", "studi"]):
        state["tools"].append("arXiv")
    if any(w in q for w in ["terbaru", "2024", "2025", "update", "perubahan", "keputusan terbaru"]):
        state["tools"].append("TavilySearch")

    if not state["tools"]:
        state["tools"].append("Wikipedia")

    return state

def search(state):
    question = state["question"]
    outputs = {}
    for tool_name in state["tools"]:
        tool = tools[tool_name]
        try:
            # Add Indonesian context to search queries
            search_query = f"{question} Indonesia hukum" if state["document_type"] == "legal" else question
            outputs[tool_name] = tool.run(search_query)
        except Exception as e:
            outputs[tool_name] = f"Error from {tool_name}: {e}"
    state["results"] = outputs
    return state

def summarize(state):
    question = state["question"]
    results = state["results"]
    combined = "\n\n".join([f"{k} result:\n{v}" for k, v in results.items()])
    
    if state["document_type"] == "legal":
        # Use Indonesian Law specialized prompt
        prompt = INDONESIAN_LAW_PROMPT.format(
            text=combined,
            question=question
        )
    else:
        # Use general prompt
        prompt = f"""Anda adalah asisten peneliti yang membantu. 

Berdasarkan informasi berikut, berikan jawaban yang jelas dan ringkas untuk pertanyaan:

Pertanyaan: {question}

Sumber:
{combined}

Jawaban:"""
    
    state["final_answer"] = llm.invoke(prompt)
    return state

# ----------------------------
# 🔗 LangGraph DAG Assembly
# ----------------------------
graph = StateGraph(state_schema=GraphState)
graph.add_node("classify", classify)
graph.add_node("search", search)
graph.add_node("summarize", summarize)

graph.set_entry_point("classify")
graph.add_edge("classify", "search")
graph.add_edge("search", "summarize")
graph.add_edge("summarize", END)

app_graph = graph.compile()

# ----------------------------
# 🎨 Enhanced UI
# ----------------------------
def get_enhanced_ui():
    return """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sistem Analisis Dokumen Hukum Indonesia</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            .gradient-bg {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .card-hover:hover {
                transform: translateY(-5px);
                transition: transform 0.3s ease;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }
            .feature-icon {
                font-size: 3rem;
                color: #667eea;
            }
            .upload-area {
                border: 2px dashed #667eea;
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                background-color: #f8f9ff;
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                border-color: #764ba2;
                background-color: #f0f2ff;
            }
            .legal-badge {
                background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <!-- Header -->
        <nav class="navbar navbar-expand-lg gradient-bg">
            <div class="container">
                <a class="navbar-brand fw-bold" href="#">
                    <i class="fas fa-balance-scale me-2"></i>
                    Sistem Analisis Dokumen Hukum Indonesia
                </a>
            </div>
        </nav>

        <!-- Main Content -->
        <div class="container mt-5">
            <!-- Hero Section -->
            <div class="row mb-5">
                <div class="col-12 text-center">
                    <h1 class="display-5 fw-bold mb-3">
                        <span class="legal-badge">AI-Powered</span> Legal Document Analysis
                    </h1>
                    <p class="lead text-muted">
                        Analisis dokumen hukum Indonesia dengan teknologi AI terdepan menggunakan Google Gemini dan LangGraph
                    </p>
                </div>
            </div>

            <!-- Features -->
            <div class="row mb-5">
                <div class="col-md-4 mb-4">
                    <div class="card h-100 card-hover border-0 shadow-sm">
                        <div class="card-body text-center">
                            <i class="fas fa-file-contract feature-icon mb-3"></i>
                            <h5 class="card-title">Multi-Format Support</h5>
                            <p class="card-text">Support untuk PDF, DOCX, TXT, CSV, dan Excel files</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4 mb-4">
                    <div class="card h-100 card-hover border-0 shadow-sm">
                        <div class="card-body text-center">
                            <i class="fas fa-brain feature-icon mb-3"></i>
                            <h5 class="card-title">AI Legal Analysis</h5>
                            <p class="card-text">Analisis mendalam dengan pemahaman konteks hukum Indonesia</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4 mb-4">
                    <div class="card h-100 card-hover border-0 shadow-sm">
                        <div class="card-body text-center">
                            <i class="fas fa-search feature-icon mb-3"></i>
                            <h5 class="card-title">Real-time Research</h5>
                            <p class="card-text">Akses ke database hukum terkini dan jurisprudensi</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Upload Form -->
            <div class="row justify-content-center">
                <div class="col-lg-8">
                    <div class="card shadow-lg border-0">
                        <div class="card-header gradient-bg text-center">
                            <h3 class="mb-0"><i class="fas fa-upload me-2"></i>Upload Dokumen Hukum</h3>
                        </div>
                        <div class="card-body p-4">
                            <form method="post" enctype="multipart/form-data" action="/analyze/" id="analysisForm">
                                <div class="mb-4">
                                    <label class="form-label fw-bold">
                                        <i class="fas fa-file-upload me-2"></i>Pilih Dokumen Hukum:
                                    </label>
                                    <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                                        <i class="fas fa-cloud-upload-alt fa-3x mb-3 text-muted"></i>
                                        <p class="mb-0">Klik untuk memilih file atau drag & drop</p>
                                        <small class="text-muted">Format yang didukung: PDF, DOCX, TXT, CSV, XLSX</small>
                                    </div>
                                    <input type="file" class="form-control d-none" name="files" id="fileInput" multiple required>
                                    <div id="fileList" class="mt-2"></div>
                                </div>
                                
                                <div class="mb-4">
                                    <label for="question" class="form-label fw-bold">
                                        <i class="fas fa-question-circle me-2"></i>Pertanyaan Hukum:
                                    </label>
                                    <textarea class="form-control" name="question" id="question" rows="4" required 
                                              placeholder="Contoh: Jelaskan sanksi pidana dalam Pasal 378 KUHP tentang penipuan dan bagaimana penerapannya dalam praktik..."></textarea>
                                    <div class="form-text">
                                        <strong>Tips:</strong> Semakin spesifik pertanyaan Anda, semakin akurat analisis yang diberikan.
                                    </div>
                                </div>

                                <div class="mb-4">
                                    <label class="form-label fw-bold">
                                        <i class="fas fa-cogs me-2"></i>Jenis Analisis:
                                    </label>
                                    <div class="row">
                                        <div class="col-md-6">
                                            <div class="form-check">
                                                <input class="form-check-input" type="radio" name="analysisType" id="interpretation" value="interpretation" checked>
                                                <label class="form-check-label" for="interpretation">
                                                    Interpretasi Hukum
                                                </label>
                                            </div>
                                            <div class="form-check">
                                                <input class="form-check-input" type="radio" name="analysisType" id="comparison" value="comparison">
                                                <label class="form-check-label" for="comparison">
                                                    Perbandingan Pasal
                                                </label>
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <div class="form-check">
                                                <input class="form-check-input" type="radio" name="analysisType" id="procedure" value="procedure">
                                                <label class="form-check-label" for="procedure">
                                                    Prosedur Hukum
                                                </label>
                                            </div>
                                            <div class="form-check">
                                                <input class="form-check-input" type="radio" name="analysisType" id="summary" value="summary">
                                                <label class="form-check-label" for="summary">
                                                    Ringkasan Dokumen
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div class="d-grid">
                                    <button type="submit" class="btn btn-lg gradient-bg text-white">
                                        <i class="fas fa-search me-2"></i>Analisis Dokumen
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Sample Questions -->
            <div class="row mt-5">
                <div class="col-12">
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-light">
                            <h5 class="mb-0"><i class="fas fa-lightbulb me-2"></i>Contoh Pertanyaan</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <ul class="list-unstyled">
                                        <li class="mb-2"><i class="fas fa-chevron-right text-primary me-2"></i>Jelaskan perbedaan antara delik aduan dan delik biasa</li>
                                        <li class="mb-2"><i class="fas fa-chevron-right text-primary me-2"></i>Apa saja unsur-unsur tindak pidana korupsi menurut UU No. 31 Tahun 1999?</li>
                                        <li class="mb-2"><i class="fas fa-chevron-right text-primary me-2"></i>Bagaimana prosedur pengajuan kasasi ke Mahkamah Agung?</li>
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <ul class="list-unstyled">
                                        <li class="mb-2"><i class="fas fa-chevron-right text-primary me-2"></i>Analisis pasal tentang hak waris dalam KUHPerdata</li>
                                        <li class="mb-2"><i class="fas fa-chevron-right text-primary me-2"></i>Sanksi administrasi vs sanksi pidana dalam hukum lingkungan</li>
                                        <li class="mb-2"><i class="fas fa-chevron-right text-primary me-2"></i>Interpretasi asas praduga tak bersalah dalam KUHAP</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="gradient-bg text-white mt-5 py-4">
            <div class="container text-center">
                <p class="mb-0">&copy; 2025 Sistem Analisis Dokumen Hukum Indonesia - Powered by AI</p>
            </div>
        </footer>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // File upload handling
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const fileList = document.getElementById('fileList');
                fileList.innerHTML = '';
                
                for (let i = 0; i < e.target.files.length; i++) {
                    const file = e.target.files[i];
                    const fileItem = document.createElement('div');
                    fileItem.className = 'alert alert-info alert-dismissible fade show mt-2';
                    fileItem.innerHTML = `
                        <i class="fas fa-file me-2"></i>${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    `;
                    fileList.appendChild(fileItem);
                }
            });

            // Form submission with loading state
            document.getElementById('analysisForm').addEventListener('submit', function(e) {
                const submitBtn = this.querySelector('button[type="submit"]');
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Menganalisis...';
                submitBtn.disabled = true;
            });
        </script>
    </body>
    </html>
    """

# ----------------------------
# 🚀 FastAPI App
# ----------------------------
app = FastAPI(title="Indonesian Law Document QA System", version="2.0.0")

@app.get("/", response_class=HTMLResponse)
async def upload_form():
    return get_enhanced_ui()

@app.post("/analyze/", response_class=HTMLResponse)
async def analyze(files: List[UploadFile] = File(...), question: str = Form(...), analysisType: str = Form("interpretation")):
    temp_dir = f"temp_uploads/{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)

    file_paths = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in [".csv", ".xls", ".xlsx", ".pdf", ".docx", ".txt"]:
            return HTMLResponse(f"""
            <div class="container mt-5">
                <div class="alert alert-danger">
                    <h4>Format File Tidak Didukung</h4>
                    <p>File <strong>{f.filename}</strong> memiliki format yang tidak didukung.</p>
                    <p>Format yang didukung: PDF, DOCX, TXT, CSV, XLSX</p>
                    <a href="/" class="btn btn-primary">Kembali</a>
                </div>
            </div>
            """, status_code=400)
        
        dest = os.path.join(temp_dir, f.filename)
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        file_paths.append(dest)

    docs = []
    for file_path in file_paths:
        try:
            loader = UnstructuredFileLoader(file_path)
            docs.extend(loader.load())
        except Exception as e:
            return HTMLResponse(f"""
            <div class="container mt-5">
                <div class="alert alert-danger">
                    <h4>Gagal Memproses File</h4>
                    <p>Tidak dapat memproses file <strong>{os.path.basename(file_path)}</strong></p>
                    <p>Error: {str(e)}</p>
                    <a href="/" class="btn btn-primary">Kembali</a>
                </div>
            </div>
            """, status_code=500)

    combined_text = "\n\n".join([doc.page_content for doc in docs])
    
    # Enhanced question with analysis type context
    enhanced_question = f"[Jenis Analisis: {analysisType}] {question}"
    
    state = {
        "question": f"{enhanced_question}\n\nKonten dokumen untuk dianalisis:\n{combined_text[:3000]}",
        "document_type": "legal",
        "language": "indonesian"
    }
    
    try:
        result = app_graph.invoke(state)
        shutil.rmtree(temp_dir)

        # Process the final answer
        final = result["final_answer"]
        if isinstance(final, AIMessage):
            raw_answer = final.content
        else:
            raw_answer = str(final)

        raw_answer = raw_answer.strip()
        html_answer = markdown(raw_answer)

        uploaded_files_list = "".join(f"""
        <div class="col-md-6 mb-2">
            <div class="card border-0 bg-light">
                <div class="card-body py-2">
                    <small><i class="fas fa-file me-2"></i>{escape(os.path.basename(f))}</small>
                </div>
            </div>
        </div>
        """ for f in file_paths)

        return f"""
        <!DOCTYPE html>
        <html lang="id">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Hasil Analisis - Sistem Hukum Indonesia</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
            <style>
                .gradient-bg {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
                .answer-content {{ 
                    line-height: 1.8; 
                    font-size: 1.1rem;
                    background: #f8f9ff;
                    border-left: 4px solid #667eea;
                    padding: 2rem;
                    border-radius: 0 10px 10px 0;
                }}
                .legal-highlight {{ background: linear-gradient(45deg, #ff6b6b, #ee5a24); color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.9rem; }}
                .timestamp {{ color: #6c757d; font-size: 0.9rem; }}
            </style>
        </head>
        <body>
            <nav class="navbar navbar-expand-lg gradient-bg">
                <div class="container">
                    <a class="navbar-brand fw-bold" href="/">
                        <i class="fas fa-balance-scale me-2"></i>
                        Sistem Analisis Dokumen Hukum Indonesia
                    </a>
                </div>
            </nav>

            <div class="container mt-4">
                <div class="row">
                    <div class="col-lg-3">
                        <div class="card shadow-sm border-0 mb-4">
                            <div class="card-header bg-light">
                                <h6 class="mb-0"><i class="fas fa-info-circle me-2"></i>Informasi Analisis</h6>
                            </div>
                            <div class="card-body">
                                <p class="mb-2"><strong>Jenis:</strong> <span class="legal-highlight">{analysisType.title()}</span></p>
                                <p class="mb-2"><strong>Waktu:</strong><br><small class="timestamp">{datetime.now().strftime('%d/%m/%Y %H:%M WIB')}</small></p>
                                <p class="mb-0"><strong>File:</strong> {len(file_paths)} dokumen</p>
                            </div>
                        </div>

                        <div class="card shadow-sm border-0">
                            <div class="card-header bg-light">
                                <h6 class="mb-0"><i class="fas fa-file me-2"></i>Dokumen Dianalisis</h6>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    {uploaded_files_list}
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="col-lg-9">
                        <div class="card shadow-lg border-0">
                            <div class="card-header gradient-bg">
                                <h4 class="mb-0"><i class="fas fa-gavel me-2"></i>Hasil Analisis Hukum</h4>
                            </div>
                            <div class="card-body">
                                <div class="mb-4">
                                    <h5 class="text-primary"><i class="fas fa-question-circle me-2"></i>Pertanyaan:</h5>
                                    <div class="bg-light p-3 rounded">
                                        <p class="mb-0">{escape(question)}</p>
                                    </div>
                                </div>

                                <div class="mb-4">
                                    <h5 class="text-success"><i class="fas fa-lightbulb me-2"></i>Analisis & Jawaban:</h5>
                                    <div class="answer-content">
                                        {html_answer}
                                    </div>
                                </div>

                                <div class="alert alert-warning">
                                    <h6><i class="fas fa-exclamation-triangle me-2"></i>Disclaimer Hukum</h6>
                                    <p class="mb-0">Analisis ini bersifat informatif dan tidak menggantikan konsultasi hukum profesional. 
                                    Untuk keperluan hukum yang mengikat, silakan berkonsultasi dengan advokat atau konsultan hukum berlisensi.</p>
                                </div>

                                <div class="d-flex gap-2">
                                    <a href="/" class="btn btn-primary">
                                        <i class="fas fa-plus me-2"></i>Analisis Baru
                                    </a>
                                    <button class="btn btn-outline-secondary" onclick="window.print()">
                                        <i class="fas fa-print me-2"></i>Cetak Hasil
                                    </button>
                                    <button class="btn btn-outline-info" onclick="copyToClipboard()">
                                        <i class="fas fa-copy me-2"></i>Salin Teks
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <footer class="gradient-bg text-white mt-5 py-3">
                <div class="container text-center">
                    <p class="mb-0">&copy; 2025 Sistem Analisis Dokumen Hukum Indonesia - Powered by AI</p>
                </div>
            </footer>

            <script>
                function copyToClipboard() {{
                    const answerContent = document.querySelector('.answer-content').innerText;
                    navigator.clipboard.writeText(answerContent).then(function() {{
                        alert('Hasil analisis berhasil disalin ke clipboard!');
                    }});
                }}
            </script>
        </body>
        </html>
        """

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return HTMLResponse(f"""
        <div class="container mt-5">
            <div class="alert alert-danger">
                <h4><i class="fas fa-exclamation-triangle me-2"></i>Error dalam Analisis</h4>
                <p>Terjadi kesalahan saat menganalisis dokumen:</p>
                <pre>{str(e)}</pre>
                <a href="/" class="btn btn-primary mt-3">Coba Lagi</a>
            </div>
        </div>
        """, status_code=500)

# ----------------------------
# 📊 API Endpoints for Advanced Features
# ----------------------------
@app.get("/api/legal-categories")
async def get_legal_categories():
    """Get available legal document categories"""
    categories = {
        "pidana": ["KUHP", "KUHAP", "UU Tipikor", "UU Narkotika"],
        "perdata": ["KUHPerdata", "UU Perkawinan", "UU Hak Cipta", "UU Merek"],
        "tata_negara": ["UUD 1945", "UU Pemilu", "UU Parpol", "UU MPR/DPR/DPD"],
        "administrasi": ["UU ASN", "UU Pelayanan Publik", "UU Keterbukaan Informasi"],
        "ekonomi": ["UU Persaingan Usaha", "UU Perbankan", "UU Pasar Modal"]
    }
    return JSONResponse(categories)

@app.post("/api/quick-search")
async def quick_legal_search(request: Request):
    """Quick search for legal terms"""
    data = await request.json()
    search_term = data.get("term", "")
    
    # Simple legal term definitions (can be expanded)
    legal_terms = {
        "delik": "Perbuatan yang dilarang dan diancam pidana oleh undang-undang",
        "novasi": "Pembaharuan utang dengan mengganti utang lama dengan utang baru",
        "subrogasi": "Penggantian kedudukan kreditur oleh pihak ketiga yang melunasi utang",
        "cessie": "Pengalihan piutang atas nama dari kreditur kepada pihak ketiga"
    }
    
    result = legal_terms.get(search_term.lower(), "Istilah tidak ditemukan dalam database")
    return JSONResponse({"term": search_term, "definition": result})

@app.get("/api/statistics")
async def get_usage_statistics():
    """Get system usage statistics"""
    # In production, this would connect to a database
    stats = {
        "total_analyses": 1247,
        "documents_processed": 3891,
        "popular_categories": ["Pidana", "Perdata", "Administrasi"],
        "success_rate": 95.8
    }
    return JSONResponse(stats)

# ----------------------------
# 🎯 Additional Features Ideas Implementation
# ----------------------------

# 1. Legal Document Template Generator
@app.post("/api/generate-template")
async def generate_legal_template(request: Request):
    """Generate legal document templates"""
    data = await request.json()
    template_type = data.get("type")
    
    templates = {
        "surat_kuasa": """
        SURAT KUASA
        
Yang bertanda tangan di bawah ini:
Nama: ________________
Alamat: _______________
Dengan ini memberikan kuasa kepada:
Nama: ________________
Alamat: _______________
Untuk dan atas nama pemberi kuasa melakukan:
[Uraian tindakan hukum]
        """,
        "kontrak_sederhana": """
        KONTRAK KERJASAMA
        
Pada hari ini _______, tanggal _____, bulan ______, tahun _____
Telah dibuat dan ditandatangani perjanjian kerjasama antara:
PIHAK PERTAMA: _____________
PIHAK KEDUA: ______________
        """
    }
    
    return JSONResponse({
        "template": templates.get(template_type, "Template tidak tersedia"),
        "type": template_type
    })

# 2. Legal Citation Formatter
@app.post("/api/format-citation")
async def format_legal_citation(request: Request):
    """Format legal citations properly"""
    data = await request.json()
    citation_data = data.get("citation")
    
    # Format Indonesian legal citations
    if citation_data.get("type") == "undang-undang":
        formatted = f"Undang-Undang Nomor {citation_data.get('number')} Tahun {citation_data.get('year')} tentang {citation_data.get('title')}"
    elif citation_data.get("type") == "putusan":
        formatted = f"Putusan {citation_data.get('court')} Nomor {citation_data.get('number')}"
    else:
        formatted = "Format sitasi tidak dikenali"
    
    return JSONResponse({"formatted_citation": formatted})

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Indonesian Law Document QA System...")
    print("📚 Features enabled:")
    print("   ✅ Indonesian Legal Document Analysis")
    print("   ✅ Multi-format file support")
    print("   ✅ Enhanced UI with legal context")
    print("   ✅ Real-time legal research")
    print("   ✅ Document template generation")
    print("   ✅ Legal citation formatting")
    print("   ✅ Usage statistics")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=9000,
        timeout_keep_alive=600,
        log_level="info",
        access_log=True,
    )