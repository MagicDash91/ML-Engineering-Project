import os
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import markdown
from typing import Annotated
from typing_extensions import TypedDict
import google.generativeai as genai
from werkzeug.utils import secure_filename
from typing import List, Dict, Optional, Any
import re
from datetime import datetime, timedelta
import io
import tempfile
import json

load_dotenv()

# ---- FastAPI Setup ----
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---- Configuration ----
EMBEDDING_DIR = "vectorstore"
MODEL_NAME = "gemini-2.0-flash"

os.makedirs(EMBEDDING_DIR, exist_ok=True)

api_key = os.getenv("GOOGLE_API_KEY")
assert api_key, "GOOGLE_API_KEY not found"
genai.configure(api_key=api_key)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize an empty vector DB
vector_db = None

# ---- In-memory storage for uploaded files and their analyses ----
temp_files = {}  # Dictionary to store file contents in memory
file_analyses = {}  # Store analysis results for each file

# ---- LangGraph State ----
class State(TypedDict):
    messages: list[str]
    user_input: str
    file_info: Optional[Dict[str, Any]]

# ---- Document Processing ----
def process_pdf_files(files: List[UploadFile]):
    documents = []
    file_index = 0
    
    for file in files:
        try:
            # Create a temporary file
            temp_file_path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
            
            # Read the uploaded file content
            file_content = file.file.read()
            
            # Write to temp file
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content)

            # Store file info in memory for future reference
            temp_files[file.filename] = {
                "path": temp_file_path,
                "content": file_content,
                "index": file_index
            }
            
            # Load the PDF
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            
            # Add file metadata to each document
            for doc in docs:
                doc.metadata["source"] = file.filename
                doc.metadata["file_index"] = file_index
            
            documents.extend(docs)
            file_index += 1
            
            # Reset file pointer for potential reuse
            file.file.seek(0)
            
        except Exception as e:
            logging.error(f"Failed to load {file.filename}: {e}")
    
    return documents

def build_vectorstore_from_files(files: List[UploadFile]):
    logging.info(f"Processing {len(files)} PDF files.")
    documents = process_pdf_files(files)
    
    if not documents:
        logging.error("No documents were loaded from the provided files.")
        return None
    
    logging.info(f"Loaded {len(documents)} documents from PDFs.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    logging.info("Splitting documents...")
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(chunks)} chunks.")

    logging.info("Starting embedding...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    logging.info("Embedding done.")
    return vector_db

# ---- Generate detailed analysis for each document ----
def generate_document_analysis(filename, content):
    """Generate a detailed analysis for a specific medical/insurance claim using Gemini."""
    try:
        prompt = f"""
        You are a medical claims fraud detection expert reviewing a claim document named {filename}.
        
        Analyze the following content carefully:
        
        {content[:15000]}  # Limit content to avoid token limits
        
        Provide a THOROUGH and IN-DEPTH analysis of this medical/insurance claim. Your analysis should be objective and evidence-based, focusing on:

        1. Claim Documentation:
           - Are all required medical records and supporting documents present?
           - Is there proper documentation of diagnosis and treatment?
           - Are there any missing or incomplete medical records?

        2. Billing Analysis:
           - Are the billed services consistent with the diagnosis?
           - Are the charges within normal ranges for the procedures?
           - Are there any duplicate charges or services?
           - Are the dates of service logical and consistent?

        3. Provider Information:
           - Is the provider properly licensed and credentialed?
           - Are there any unusual patterns in provider behavior?
           - Is the provider's specialty consistent with the services billed?

        4. Patient Information:
           - Is the patient's medical history consistent with the claim?
           - Are there any unusual patterns in patient behavior?
           - Is the frequency of visits reasonable for the condition?

        Format your analysis in Markdown with bullet points for each key finding. Include specific examples from the document.
        
        Based on your analysis, classify this claim as "High Risk" ONLY if you find:
        - Multiple instances of billing inconsistencies
        - Evidence of upcoding or unbundling
        - Unusual patterns in service frequency
        - Missing or forged documentation
        - Services inconsistent with provider specialty
        - Duplicate claims or charges
        
        Otherwise, classify as "Low Risk".
        
        Structure your response as follows:
        
        ## Risk Assessment: [High/Low]
        
        ### Key Findings:
        * **[Finding Title 1]:** [Detailed explanation with specific examples]
        * **[Finding Title 2]:** [Detailed explanation with specific examples]
        * **[Finding Title 3]:** [Detailed explanation with specific examples]
        
        ### Conclusion:
        [Summarize why this claim represents high or low risk, providing specific evidence and reasoning]
        """
        
        # Get analysis from Gemini
        analysis_result = llm.invoke(prompt).content
        
        # Extract risk level
        risk_match = re.search(r'## Risk Assessment:\s*(High|Low)', analysis_result, re.IGNORECASE)
        risk_level = risk_match.group(1) if risk_match else "Low"  # Default to low if parsing fails
        
        return {
            "filename": filename,
            "risk_level": risk_level,
            "analysis": analysis_result
        }
    except Exception as e:
        logging.exception(f"Failed to generate analysis for {filename}")
        return {
            "filename": filename,
            "risk_level": "Low",  # Default to low risk on error
            "analysis": f"## Risk Assessment: Low\n\n### Key Findings:\n* **Error in Analysis:** Failed to properly analyze this claim due to technical issues: {str(e)}"
        }

# ---- LangGraph Tasks ----
def ask_question_with_rag(state: State):
    global vector_db
    user_input = state["user_input"]
    
    if not vector_db:
        return {"messages": ["Please upload PDF files first before asking questions."], "user_input": user_input}

    # Perform similarity search based on user input
    retrieved_docs = vector_db.similarity_search(user_input, k=5)

    # Enhanced prompt for proposal analysis
    prompt_template = """You are a proposal analysis expert. Analyze the following proposal documents for potential corruption or suspicious patterns:

{docs}

Based on these documents, identify any suspicious patterns or potential corruption indicators. Focus on:
1. Inconsistencies in financial figures or calculations
2. Unusual formatting or structural issues
3. Missing or incomplete sections
4. Suspicious patterns in dates, signatures, or approvals
5. Unusual changes in writing style or content

Question: {question}

Provide a detailed analysis focusing on these aspects. If you find any suspicious patterns, explain why they are concerning. 
Make your assessment based on concrete evidence in the documents.
When discussing risk levels, only use "High Risk" or "Low Risk" categories."""

    prompt = PromptTemplate(template=prompt_template, input_variables=["docs", "question"])
    
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    result = llm_chain.run(docs=context, question=user_input)

    # Include file context in the state
    file_sources = list(set([doc.metadata.get("source", "Unknown") for doc in retrieved_docs]))
    
    return {
        "messages": [result], 
        "user_input": user_input,
        "file_info": {
            "sources": file_sources,
            "analyses": {filename: file_analyses.get(filename, {}) for filename in file_sources}
        }
    }

def render_result_as_html(state: State):
    answer = state["messages"][-1]
    html = markdown.markdown(answer)
    
    return {"messages": [html], "user_input": state["user_input"], "file_info": state.get("file_info")}

def assess_risk(state: State):
    # Enhanced risk assessment logic for medical claims
    message = state["messages"][0].lower()
    
    # Define high-risk indicators with context
    high_risk_indicators = {
        "billing": ["upcoding", "unbundling", "duplicate charges", "excessive billing", "unusual charges"],
        "documentation": ["missing records", "forged documents", "incomplete documentation", "altered records"],
        "provider": ["unlicensed provider", "specialty mismatch", "suspicious provider", "credentialing issues"],
        "patient": ["patient mismatch", "identity fraud", "unusual frequency", "suspicious pattern"]
    }
    
    # Count high-risk indicators with context
    high_risk_count = 0
    for category, indicators in high_risk_indicators.items():
        if any(indicator in message for indicator in indicators):
            high_risk_count += 1
    
    # Define low-risk indicators
    low_risk_indicators = [
        "proper documentation", "consistent billing", "appropriate charges",
        "verified provider", "legitimate patient", "standard treatment"
    ]
    
    # Count low-risk indicators
    low_risk_count = sum(1 for indicator in low_risk_indicators if indicator in message)
    
    # Risk assessment logic
    if high_risk_count >= 2:  # Require multiple high-risk indicators
        risk_level = "High"
    elif low_risk_count >= 3 and high_risk_count == 0:  # Strong evidence of low risk
        risk_level = "Low"
    else:
        risk_level = "Low"  # Default to low risk if uncertain
    
    return {
        "messages": state["messages"], 
        "risk_level": risk_level, 
        "user_input": state["user_input"],
        "file_info": state.get("file_info")
    }

# ---- LangGraph Definition ----
graph_builder = StateGraph(State)
graph_builder.add_node("qa_rag", ask_question_with_rag)
graph_builder.add_node("assess_risk", assess_risk)
graph_builder.add_node("render_html", render_result_as_html)

graph_builder.set_entry_point("qa_rag")
graph_builder.add_edge("qa_rag", "assess_risk")
graph_builder.add_edge("assess_risk", "render_html")
graph_builder.set_finish_point("render_html")

graph = graph_builder.compile()

# ---- Routes ----
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head>
        <title>Medical Claims Fraud Detection</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            .upload-area {
                border: 2px dashed #dee2e6;
                padding: 20px;
                text-align: center;
                margin: 20px 0;
                border-radius: 8px;
            }
            .upload-area.dragover {
                background: #e9ecef;
                border-color: #007bff;
            }
            .result-container {
                margin-top: 20px;
            }
            .question-form {
                margin-top: 20px;
                display: none;
            }
            .file-list {
                text-align: left;
                margin-top: 15px;
            }
            .risk-high {
                color: #dc3545;
                font-weight: bold;
            }
            .risk-low {
                color: #28a745;
                font-weight: bold;
            }
            .analysis-toggle {
                cursor: pointer;
                transition: all 0.3s ease;
                border: 1px solid #e0e0e0;
                background-color: #ffffff;
                color: #333333;
                font-weight: 500;
            }
            .analysis-toggle:hover {
                background-color: #f8f9fa;
                border-color: #0d6efd;
                color: #0d6efd;
            }
            .analysis-content {
                display: none;
                margin-top: 10px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
                border-left: 4px solid #6c757d;
            }
            .markdown-content h2 {
                font-size: 1.4rem;
                margin-top: 1.5rem;
                margin-bottom: 1rem;
                color: #495057;
                border-bottom: 2px solid #e9ecef;
                padding-bottom: 0.5rem;
            }
            .markdown-content h3 {
                font-size: 1.2rem;
                margin-top: 1.2rem;
                margin-bottom: 0.8rem;
                color: #495057;
            }
            .markdown-content ul {
                padding-left: 1.5rem;
                margin-bottom: 1rem;
            }
            .markdown-content li {
                margin-bottom: 0.5rem;
            }
            .markdown-content strong {
                color: #495057;
                font-weight: 600;
            }
            .markdown-content p {
                margin-bottom: 1rem;
                line-height: 1.6;
            }
            .markdown-content blockquote {
                border-left: 4px solid #e9ecef;
                padding-left: 1rem;
                margin-left: 0;
                color: #6c757d;
            }
            .chat-container {
                max-height: calc(100vh - 300px);
                overflow-y: auto;
                padding: 20px;
                background: transparent;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .chat-message {
                margin-bottom: 20px;
                display: flex;
                flex-direction: column;
            }
            .user-message {
                align-items: flex-end;
            }
            .assistant-message {
                align-items: flex-start;
            }
            .message-bubble {
                max-width: 80%;
                padding: 12px 16px;
                border-radius: 12px;
                margin-bottom: 4px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }
            .user-message .message-bubble {
                background-color: #007bff;
                color: white;
            }
            .assistant-message .message-bubble {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
            }
            .message-time {
                font-size: 0.75rem;
                color: #6c757d;
                margin-top: 4px;
            }
            .chat-input-container {
                background: transparent;
                padding: 0;
                border-top: 1px solid #dee2e6;
                margin-top: 20px;
            }
            .question-form {
                margin-top: 20px;
                display: none;
                background: #fff;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .container {
                padding-bottom: 80px; /* Space for the fixed input */
            }
        </style>
    </head>
    <body class="bg-light">
        <div class="container py-5">
            <h2 class="text-center mb-4">Medical Claims Fraud Detection</h2>
            
            <div class="upload-area" id="uploadArea">
                <h4>Upload Medical Claims</h4>
                <p class="text-muted">Drag and drop medical claim documents here or click to select</p>
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="fileInput" name="files" multiple accept=".pdf" class="d-none">
                    <button type="button" class="btn btn-primary" onclick="document.getElementById('fileInput').click()">
                        Select Files
                    </button>
                </form>
                <div class="file-list" id="fileList"></div>
            </div>

            <div class="question-form" id="questionForm">
                <h4>Chat About These Claims</h4>
                <div class="chat-container" id="chatContainer">
                    <div class="chat-message assistant-message">
                        <div class="message-bubble">
                            Hello! I'm your medical claims analysis assistant. You can ask me questions about the uploaded claims, and I'll help you identify any potential fraud patterns or concerns.
                        </div>
                        <div class="message-time">System</div>
                    </div>
                </div>
                <div class="chat-input-container">
                    <form id="askForm" class="d-flex gap-2">
                        <input type="text" id="questionInput" class="form-control" placeholder="Ask about the claims...">
                        <button class="btn btn-primary" type="submit">Send</button>
                    </form>
                </div>
            </div>

            <div class="result-container" id="resultContainer" style="display: none;">
                <h4>Analysis Results</h4>
                <div id="analysisResults"></div>
            </div>
        </div>

        <script>
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const fileList = document.getElementById('fileList');
            const questionForm = document.getElementById('questionForm');
            const resultContainer = document.getElementById('resultContainer');
            const analysisResults = document.getElementById('analysisResults');

            // Drag and drop handlers
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
                handleFiles(e.dataTransfer.files);
            });

            fileInput.addEventListener('change', (e) => {
                handleFiles(e.target.files);
            });

            function handleFiles(files) {
                if (files.length === 0) return;

                const formData = new FormData();
                Array.from(files).forEach(file => {
                    formData.append('files', file);
                });

                // Show loading state
                fileList.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"></div><p class="mt-2">Uploading and analyzing files...</p></div>';

                // Upload and analyze
                fetch('/analyze/', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    let filesHtml = '<ul class="list-group">';
                    data.results.forEach(result => {
                        const riskClass = `risk-${result.risk_level.toLowerCase()}`;
                        filesHtml += `
                            <li class="list-group-item">
                                <div class="d-flex justify-content-between align-items-center">
                                    <span>${result.filename}</span>
                                    <span class="badge bg-${result.risk_level.toLowerCase() === 'high' ? 'danger' : 'success'}">${result.risk_level} Risk</span>
                                </div>
                                <div>
                                    <button class="btn btn-sm analysis-toggle" onclick="toggleAnalysis('analysis-${result.filename.replace(/[^a-zA-Z0-9]/g, '')}')">Show Analysis ▼</button>
                                    <div id="analysis-${result.filename.replace(/[^a-zA-Z0-9]/g, '')}" class="analysis-content">
                                        <div class="markdown-content">${marked.parse(result.analysis)}</div>
                                    </div>
                                </div>
                            </li>
                        `;
                    });
                    filesHtml += '</ul>';
                    
                    fileList.innerHTML = filesHtml;
                    
                    // Show question form after files are uploaded
                    questionForm.style.display = 'block';
                })
                .catch(error => {
                    fileList.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
                });
            }

            // Toggle analysis details
            function toggleAnalysis(id) {
                const element = document.getElementById(id);
                const isHidden = element.style.display === 'none' || element.style.display === '';
                element.style.display = isHidden ? 'block' : 'none';
                
                // Update the toggle button text
                const toggleElement = element.previousElementSibling;
                toggleElement.textContent = isHidden ? 'Hide Analysis ▲' : 'Show Analysis ▼';
            }

            // Add message to chat
            function addMessage(content, isUser = false) {
                const chatContainer = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `chat-message ${isUser ? 'user-message' : 'assistant-message'}`;
                
                const bubble = document.createElement('div');
                bubble.className = 'message-bubble';
                bubble.innerHTML = content;
                
                const timeDiv = document.createElement('div');
                timeDiv.className = 'message-time';
                timeDiv.textContent = isUser ? 'You' : 'Assistant';
                
                messageDiv.appendChild(bubble);
                messageDiv.appendChild(timeDiv);
                chatContainer.appendChild(messageDiv);
                
                // Scroll to bottom
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            // Modified ask question handler
            document.getElementById('askForm').addEventListener('submit', (e) => {
                e.preventDefault();
                const question = document.getElementById('questionInput').value.trim();
                
                if (!question) return;

                // Add user message to chat
                addMessage(question, true);
                
                // Clear input
                document.getElementById('questionInput').value = '';

                // Show loading state
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'chat-message assistant-message';
                loadingDiv.innerHTML = `
                    <div class="message-bubble">
                        <div class="spinner-border spinner-border-sm" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        Analyzing...
                    </div>
                `;
                document.getElementById('chatContainer').appendChild(loadingDiv);

                // Send question
                fetch('/ask/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `question=${encodeURIComponent(question)}`
                })
                .then(response => response.text())
                .then(html => {
                    // Remove loading message
                    loadingDiv.remove();
                    // Add assistant response
                    addMessage(html);
                })
                .catch(error => {
                    // Remove loading message
                    loadingDiv.remove();
                    // Add error message
                    addMessage(`<div class="text-danger">Error: ${error.message}</div>`);
                });
            });

            // Configure marked options
            marked.setOptions({
                breaks: true,
                gfm: true,
                headerIds: true,
                mangle: false
            });
        </script>
    </body>
    </html>
    """

@app.post("/ask/", response_class=HTMLResponse)
async def ask_question(question: str = Form(...)):
    try:
        state = {"messages": [], "user_input": question}
        result = graph.invoke(state)
        return HTMLResponse(content=result["messages"][-1])
    except Exception as e:
        logging.exception("Failed to process RAG")
        return HTMLResponse(content=f"<div class='text-danger'>Error: {str(e)}</div>")

@app.post("/analyze/")
async def analyze_documents(files: List[UploadFile] = File(...)):
    try:
        global file_analyses
        results = []
        documents = process_pdf_files(files)
        
        if not documents:
            return JSONResponse(
                status_code=400,
                content={"error": "No valid documents were found in the uploaded files"}
            )
        
        # Group documents by file
        docs_by_file = {}
        for doc in documents:
            filename = doc.metadata.get('source')
            if filename not in docs_by_file:
                docs_by_file[filename] = []
            docs_by_file[filename].append(doc)
        
        for filename, docs in docs_by_file.items():
            try:
                # Combine all document content for this file
                full_content = " ".join([doc.page_content for doc in docs])
                
                # Basic analysis (quick checks)
                content = full_content.lower()
                
                # Initial suspicious indicators
                flags = []
                
                # Proposal-specific checks
                required_sections = ["executive summary", "scope of work", "budget", "timeline"]
                missing_sections = [section for section in required_sections if section not in content]
                if missing_sections:
                    flags.append(f"Missing essential sections: {', '.join(missing_sections)}")
                
                # Financial consistency check
                if "budget" in content or "cost" in content:
                    numbers = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d{2})?', content)
                    if len(numbers) > 0:
                        try:
                            numbers = [float(n.replace('$', '').replace(',', '')) for n in numbers]
                            if max(numbers) / min(numbers) > 100:
                                flags.append("Large discrepancies in financial figures detected")
                        except ValueError:
                            pass
                
                # Date consistency check
                dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', content)
                if dates:
                    try:
                        date_formats = ['%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y', '%d-%m-%Y']
                        parsed_dates = []
                        
                        for d in dates:
                            for fmt in date_formats:
                                try:
                                    parsed_dates.append(datetime.strptime(d, fmt))
                                    break
                                except ValueError:
                                    continue
                        
                        if parsed_dates and max(parsed_dates) - min(parsed_dates) > timedelta(days=365*2):  # More lenient - 2 years
                            flags.append("Unusual date range detected")
                    except:
                        flags.append("Inconsistent date formats")
                
                # Get initial risk level based on basic checks
                initial_risk = "High" if len(flags) >= 2 else "Low"
                
                # Get detailed analysis from Gemini
                detailed_analysis = generate_document_analysis(filename, full_content)
                
                # Store the analysis for future reference
                file_analyses[filename] = detailed_analysis
                
                results.append({
                    "filename": filename,
                    "risk_level": detailed_analysis["risk_level"],
                    "analysis": detailed_analysis["analysis"]
                })
                
            except Exception as e:
                results.append({
                    "filename": filename,
                    "risk_level": "High",
                    "analysis": f"Error analyzing document: {str(e)}"
                })
        
        # Build global vectorstore from the processed files
        global vector_db
        vector_db = build_vectorstore_from_files(files)
        
        return JSONResponse(content={"results": results})
        
    except Exception as e:
        logging.exception("Failed to analyze documents")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/upload/", response_class=JSONResponse)
async def upload_pdf(files: List[UploadFile] = File(...)):
    try:
        # Process files
        global vector_db
        vector_db = build_vectorstore_from_files(files)
        
        if not vector_db:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Failed to process uploaded files"}
            )
        
        return JSONResponse(content={"success": True, "message": f"Successfully processed {len(files)} files"})
    except Exception as e:
        logging.exception("Failed to upload PDF")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

# Cleanup function to remove temporary files when the server shuts down
@app.on_event("shutdown")
def cleanup():
    for file_info in temp_files.values():
        try:
            if os.path.exists(file_info["path"]):
                os.remove(file_info["path"])
        except Exception as e:
            logging.error(f"Failed to remove temporary file: {e}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=9000,
        timeout_keep_alive=600,
        log_level="info",
        access_log=True,
    )