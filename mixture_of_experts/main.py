# Gemini Mixture of Experts Document Analyzer
# FastAPI Backend with Multi-file Support and Chat Interface

import os
import io
import uuid
import asyncio
import tempfile
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Google Gemini
import google.generativeai as genai

# LangChain imports
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# ML/NLP tools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import markdown
import re

# Create FastAPI app
app = FastAPI(title="Gemini MoE Document Analyzer")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global configuration
GEMINI_API_KEY = "AIzaSyAMAYxkjP49QZRCg21zImWWAu7c3YHJ0a8"  # Replace with your key
genai.configure(api_key=GEMINI_API_KEY)

# Global variables
sessions = {}
moe_analyzer = None

# Pydantic models
class AnalysisRequest(BaseModel):
    session_id: str
    question: str
    use_moe: bool = True

class ChatRequest(BaseModel):
    session_id: str
    message: str

class SessionInfo(BaseModel):
    session_id: str
    file_count: int
    total_chunks: int
    status: str

class GeminiMoEAnalyzer:
    """Gemini-based Mixture of Experts for document analysis"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.3
        )
        
        # Define expert personas
        self.experts = {
            "summarization_expert": {
                "persona": """You are a Senior Content Summarization Specialist with expertise in:
                - Creating concise, comprehensive summaries of complex documents
                - Extracting key points, main arguments, and critical insights
                - Structuring information hierarchically (executive summary → key points → details)
                - Identifying patterns, trends, and relationships across multiple sources
                - Presenting information in clear, actionable formats for decision-makers
                Focus on clarity, brevity, and capturing the essence of the content.""",
                "keywords": ["summary", "summarize", "overview", "key points", "main ideas", "extract", "brief"],
                "temperature": 0.2
            },
            
            "insight_expert": {
                "persona": """You are a Strategic Business Analyst specializing in:
                - Deep analytical thinking and pattern recognition
                - Generating actionable insights from complex data and documents
                - Identifying business implications, opportunities, and risks
                - Cross-referencing information to reveal hidden connections
                - Strategic recommendations based on evidence
                - Market analysis and competitive intelligence
                Provide strategic, forward-thinking insights that drive decision-making.""",
                "keywords": ["insight", "analysis", "implications", "recommendations", "strategy", "opportunities", "trends"],
                "temperature": 0.4
            },
            
            "research_expert": {
                "persona": """You are a Research Methodology Expert with skills in:
                - Comprehensive literature review and research synthesis
                - Academic and technical document analysis
                - Evidence evaluation and source credibility assessment
                - Research gap identification and methodology critique
                - Comparative analysis across multiple studies or documents
                - Scientific reasoning and hypothesis evaluation
                Focus on thorough, evidence-based analysis with academic rigor.""",
                "keywords": ["research", "study", "evidence", "methodology", "findings", "conclusion", "academic"],
                "temperature": 0.3
            },
            
            "financial_expert": {
                "persona": """You are a Senior Financial Analyst specializing in:
                - Financial statement analysis and performance evaluation
                - Market trends, investment analysis, and risk assessment
                - Business valuation and financial modeling
                - Economic indicators and their business impact
                - Cost-benefit analysis and ROI calculations
                - Financial planning and forecasting
                Provide precise financial insights with quantitative support when available.""",
                "keywords": ["financial", "revenue", "profit", "investment", "cost", "budget", "economic", "money"],
                "temperature": 0.2
            },
            
            "technical_expert": {
                "persona": """You are a Technical Documentation Specialist with expertise in:
                - Complex technical concept explanation and clarification
                - System architecture analysis and technical requirements
                - Process documentation and workflow optimization
                - Technical risk assessment and solution evaluation
                - Integration analysis and compatibility assessment
                - Best practices and industry standards compliance
                Translate technical complexity into clear, actionable information.""",
                "keywords": ["technical", "system", "process", "implementation", "architecture", "specification", "development"],
                "temperature": 0.3
            }
        }
        
        # Initialize expert selection system
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        self._initialize_expert_selection()
    
    def _initialize_expert_selection(self):
        """Initialize TF-IDF for expert selection"""
        expert_texts = []
        self.expert_names = list(self.experts.keys())
        
        for expert_config in self.experts.values():
            text = expert_config["persona"] + " " + " ".join(expert_config["keywords"])
            expert_texts.append(text)
        
        self.expert_vectors = self.vectorizer.fit_transform(expert_texts)
    
    def select_experts(self, question: str, max_experts: int = 2) -> List[str]:
        """Select most relevant experts based on question"""
        # TF-IDF similarity
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.expert_vectors)[0]
        
        # Keyword-based boosting
        boosted_scores = similarities.copy()
        for i, expert_name in enumerate(self.expert_names):
            expert_config = self.experts[expert_name]
            keyword_matches = sum(1 for keyword in expert_config["keywords"] 
                                if keyword.lower() in question.lower())
            if keyword_matches > 0:
                boosted_scores[i] += 0.1 * keyword_matches
        
        # Select top experts
        expert_indices = np.argsort(boosted_scores)[-max_experts:][::-1]
        selected_experts = [self.expert_names[i] for i in expert_indices if boosted_scores[i] > 0.1]
        
        # Ensure at least one expert is selected
        if not selected_experts:
            selected_experts = [self.expert_names[np.argmax(boosted_scores)]]
        
        return selected_experts
    
    def query_expert(self, expert_name: str, question: str, context: str) -> Dict[str, Any]:
        """Query a single expert"""
        expert_config = self.experts[expert_name]
        
        prompt = f"""{expert_config["persona"]}

Document Context:
{context}

User Question: {question}

Please provide a comprehensive response based on your expertise. Be specific, actionable, and thorough in your analysis."""
        
        try:
            response = self.llm.invoke(prompt)
            return {
                "expert": expert_name,
                "response": response.content,
                "success": True
            }
        except Exception as e:
            return {
                "expert": expert_name,
                "response": f"Error: {str(e)}",
                "success": False
            }
    
    def query_multiple_experts(self, expert_names: List[str], question: str, context: str) -> List[Dict[str, Any]]:
        """Query multiple experts in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=len(expert_names)) as executor:
            future_to_expert = {
                executor.submit(self.query_expert, expert, question, context): expert 
                for expert in expert_names
            }
            
            for future in future_to_expert:
                result = future.result()
                results.append(result)
        
        return results
    
    def synthesize_responses(self, expert_results: List[Dict[str, Any]], question: str) -> str:
        """Synthesize multiple expert responses"""
        successful_results = [r for r in expert_results if r["success"]]
        
        if not successful_results:
            return "No expert responses were generated successfully."
        
        if len(successful_results) == 1:
            expert_name = successful_results[0]["expert"].replace("_", " ").title()
            return f"## {expert_name} Analysis\n\n{successful_results[0]['response']}"
        
        # Multi-expert synthesis
        combined_response = f"# Multi-Expert Analysis: {question}\n\n"
        
        for result in successful_results:
            expert_name = result["expert"].replace("_", " ").title()
            combined_response += f"## {expert_name}\n\n{result['response']}\n\n"
        
        # Generate synthesis
        if len(successful_results) > 1:
            synthesis_prompt = f"""Based on these expert analyses for the question "{question}":

{chr(10).join([f"{r['expert']}: {r['response']}" for r in successful_results])}

Provide a concise integrated summary that:
1. Highlights key agreements between experts
2. Identifies complementary insights
3. Provides actionable recommendations
4. Notes any important differences in perspective

Keep the synthesis focused and under 200 words."""
            
            try:
                synthesis_response = self.llm.invoke(synthesis_prompt)
                combined_response += f"## Integrated Summary\n\n{synthesis_response.content}\n"
            except Exception as e:
                combined_response += f"## Integrated Summary\n\nMultiple expert perspectives provided above offer comprehensive insights for your question.\n"
        
        return combined_response
    
    async def analyze_documents(self, question: str, documents: List[Document], use_moe: bool = True) -> Dict[str, Any]:
        """Main analysis method"""
        # Prepare context from documents
        context = "\n\n".join([f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(documents[:10])])
        
        if use_moe:
            # Select experts
            selected_experts = self.select_experts(question, max_experts=3)
            
            # Query experts
            expert_results = self.query_multiple_experts(selected_experts, question, context)
            
            # Synthesize responses
            final_response = self.synthesize_responses(expert_results, question)
            
            return {
                "question": question,
                "response": final_response,
                "experts_used": selected_experts,
                "expert_results": expert_results,
                "method": "mixture_of_experts"
            }
        else:
            # Single expert approach (use summarization expert as default)
            result = self.query_expert("summarization_expert", question, context)
            
            return {
                "question": question,
                "response": result["response"],
                "experts_used": ["summarization_expert"],
                "expert_results": [result],
                "method": "single_expert"
            }

class DocumentProcessor:
    """Handle document loading and processing"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
    
    async def process_files(self, files: List[UploadFile], session_id: str) -> Dict[str, Any]:
        """Process uploaded files and create vector store"""
        try:
            all_documents = []
            processed_files = []
            
            # Create temp directory for this session
            temp_dir = tempfile.mkdtemp(prefix=f"session_{session_id}_")
            
            for file in files:
                # Save file temporarily
                file_content = await file.read()
                temp_file_path = os.path.join(temp_dir, file.filename)
                
                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(file_content)
                
                # Load document
                try:
                    loader = UnstructuredFileLoader(temp_file_path)
                    documents = loader.load()
                    
                    # Add metadata
                    for doc in documents:
                        doc.metadata["source_file"] = file.filename
                        doc.metadata["session_id"] = session_id
                    
                    all_documents.extend(documents)
                    processed_files.append({
                        "filename": file.filename,
                        "status": "success",
                        "chunks": len(documents)
                    })
                    
                except Exception as e:
                    processed_files.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": str(e)
                    })
                
                # Clean up temp file
                try:
                    os.remove(temp_file_path)
                except:
                    pass
            
            # Clean up temp directory
            try:
                os.rmdir(temp_dir)
            except:
                pass
            
            if not all_documents:
                raise Exception("No documents were successfully processed")
            
            # Split documents into chunks
            document_chunks = self.text_splitter.split_documents(all_documents)
            
            # Create vector store for chat functionality
            vectorstore = FAISS.from_documents(document_chunks, self.embeddings)
            
            # Store in session
            sessions[session_id] = {
                "documents": document_chunks,
                "vectorstore": vectorstore,
                "processed_files": processed_files,
                "created_at": pd.Timestamp.now(),
                "chat_history": []
            }
            
            return {
                "session_id": session_id,
                "total_documents": len(all_documents),
                "total_chunks": len(document_chunks),
                "processed_files": processed_files,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "session_id": session_id,
                "status": "error",
                "error": str(e)
            }

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global moe_analyzer
    moe_analyzer = GeminiMoEAnalyzer(GEMINI_API_KEY)
    print("Gemini MoE Document Analyzer initialized")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Main interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemini MoE Document Analyzer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        
        body {
            background: var(--primary-gradient);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .main-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            margin: 20px;
            overflow: hidden;
        }
        
        .header-section {
            background: linear-gradient(45deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            background: rgba(102, 126, 234, 0.05);
            margin: 20px 0;
        }
        
        .upload-area:hover, .upload-area.dragover {
            border-color: #38ef7d;
            background: rgba(56, 239, 125, 0.1);
            transform: translateY(-2px);
        }
        
        .btn-primary {
            background: var(--primary-gradient);
            border: none;
            border-radius: 25px;
            padding: 12px 30px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .btn-success {
            background: var(--success-gradient);
            border: none;
            border-radius: 25px;
            padding: 12px 30px;
            font-weight: 600;
        }
        
        .form-control {
            border-radius: 15px;
            border: 2px solid #e9ecef;
            padding: 15px 20px;
            transition: all 0.3s ease;
        }
        
        .form-control:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
        }
        
        .card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        .file-list {
            max-height: 200px;
            overflow-y: auto;
        }
        
        .file-item {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 10px 15px;
            margin-bottom: 8px;
            display: flex;
            justify-content: between;
            align-items: center;
        }
        
        .chat-container {
            height: 700px;
            overflow-y: auto;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 15px;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 10px;
            animation: fadeInUp 0.3s ease-out;
        }
        
        .user-message {
            background: var(--primary-gradient);
            color: white;
            margin-left: 20%;
            text-align: right;
        }
        
        .bot-message {
            background: white;
            border: 1px solid #e9ecef;
            margin-right: 20%;
            border-left: 4px solid #28a745;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
            font-style: italic;
        }
        
        .nav-tabs .nav-link {
            border-radius: 10px 10px 0 0;
            border: none;
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
            margin-right: 5px;
            font-weight: 500;
        }
        
        .nav-tabs .nav-link.active {
            background: #667eea;
            color: white;
        }
        
        .analysis-result {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            border-left: 4px solid #28a745;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="main-container">
            <div class="header-section">
                <h1><i class="fas fa-brain"></i> Gemini MoE Document Analyzer</h1>
                <p class="mb-0">Advanced AI-powered document analysis with Mixture of Experts</p>
            </div>
            
            <div class="container-fluid p-4">
                <div class="row">
                    <!-- Left Panel: Upload and Analysis -->
                    <div class="col-lg-6">
                        <div class="card">
                            <div class="card-header">
                                <h4><i class="fas fa-upload"></i> Document Upload</h4>
                            </div>
                            <div class="card-body">
                                <div class="upload-area" id="uploadArea">
                                    <i class="fas fa-cloud-upload-alt fa-3x mb-3" style="color: #667eea;"></i>
                                    <h5>Upload Your Documents</h5>
                                    <p class="mb-3">Drag and drop files here or click to browse</p>
                                    <p class="small text-muted">Supports: PDF, DOC, DOCX, PPT, PPTX, XLS, XLSX, CSV, TXT</p>
                                    <input type="file" id="fileInput" multiple accept=".pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.csv,.txt" style="display: none;">
                                </div>
                                
                                <div id="fileList" class="file-list" style="display: none;">
                                    <h6>Selected Files:</h6>
                                    <div id="files"></div>
                                </div>
                                
                                <button id="processBtn" class="btn btn-primary w-100 mt-3" disabled>
                                    <i class="fas fa-cogs"></i> Process Documents
                                </button>
                            </div>
                        </div>
                        
                        <div class="card" id="analysisCard" style="display: none;">
                            <div class="card-header">
                                <h4><i class="fas fa-search"></i> AI Analysis</h4>
                            </div>
                            <div class="card-body">
                                <div class="mb-3">
                                    <label class="form-label">Your Question:</label>
                                    <textarea id="questionInput" class="form-control" rows="3" placeholder="What would you like to know about your documents? (e.g., 'Summarize the key findings', 'What are the main insights?', 'Provide a technical analysis')"></textarea>
                                </div>
                                
                                <div class="form-check mb-3">
                                    <input class="form-check-input" type="checkbox" id="useMoE" checked>
                                    <label class="form-check-label" for="useMoE">
                                        Use Mixture of Experts (Multiple AI specialists)
                                    </label>
                                </div>
                                
                                <button id="analyzeBtn" class="btn btn-success w-100">
                                    <i class="fas fa-brain"></i> Analyze Documents
                                </button>
                                
                                <div id="analysisResult" class="analysis-result" style="display: none;"></div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Right Panel: Chat -->
                    <div class="col-lg-6">
                        <div class="card">
                            <div class="card-header">
                                <h4><i class="fas fa-comments"></i> Chat with Documents</h4>
                            </div>
                            <div class="card-body">
                                <div id="chatContainer" class="chat-container">
                                    <div class="message bot-message">
                                        <strong>AI Assistant:</strong><br>
                                        Upload and process your documents first, then I'll be ready to answer any questions about them!
                                    </div>
                                </div>
                                
                                <div class="loading" id="chatLoading" style="display: none;">
                                    AI is thinking...
                                </div>
                                
                                <div class="input-group mt-3">
                                    <input type="text" id="chatInput" class="form-control" placeholder="Ask questions about your documents..." disabled>
                                    <button id="sendBtn" class="btn btn-primary" disabled>
                                        <i class="fas fa-paper-plane"></i>
                                    </button>
                                </div>
                            </div>
                        </div>
                        
                        <div id="sessionInfo" class="card" style="display: none;">
                            <div class="card-body">
                                <h6>Session Info</h6>
                                <div id="sessionDetails"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        let sessionId = null;
        let selectedFiles = [];
        
        // Generate session ID
        function generateSessionId() {
            return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }
        
        // Initialize drag and drop
        function initializeDragDrop() {
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            
            uploadArea.addEventListener('click', () => fileInput.click());
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, preventDefaults, false);
            });
            
            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
            });
            
            uploadArea.addEventListener('drop', handleDrop, false);
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            function handleDrop(e) {
                const files = Array.from(e.dataTransfer.files);
                handleFileSelection(files);
            }
        }
        
        // Handle file selection
        function handleFileSelection(files) {
            selectedFiles = files;
            displaySelectedFiles();
            document.getElementById('processBtn').disabled = files.length === 0;
        }
        
        // Display selected files
        function displaySelectedFiles() {
            const fileListDiv = document.getElementById('fileList');
            const filesDiv = document.getElementById('files');
            
            if (selectedFiles.length > 0) {
                fileListDiv.style.display = 'block';
                
                let html = '';
                selectedFiles.forEach((file, index) => {
                    const sizeKB = Math.round(file.size / 1024);
                    html += `
                        <div class="file-item">
                            <div>
                                <i class="fas fa-file"></i>
                                <span class="ms-2">${file.name}</span>
                                <small class="text-muted ms-2">(${sizeKB} KB)</small>
                            </div>
                            <button class="btn btn-sm btn-outline-danger" onclick="removeFile(${index})">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                    `;
                });
                
                filesDiv.innerHTML = html;
            } else {
                fileListDiv.style.display = 'none';
            }
        }
        
        // Remove file
        function removeFile(index) {
            selectedFiles.splice(index, 1);
            displaySelectedFiles();
            document.getElementById('processBtn').disabled = selectedFiles.length === 0;
        }
        
        // Process documents
        async function processDocuments() {
            if (selectedFiles.length === 0) return;
            
            sessionId = generateSessionId();
            
            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('files', file);
            });
            formData.append('session_id', sessionId);
            
            const processBtn = document.getElementById('processBtn');
            processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            processBtn.disabled = true;
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    // Show analysis card
                    document.getElementById('analysisCard').style.display = 'block';
                    
                    // Enable chat
                    document.getElementById('chatInput').disabled = false;
                    document.getElementById('sendBtn').disabled = false;
                    
                    // Show session info
                    document.getElementById('sessionInfo').style.display = 'block';
                    document.getElementById('sessionDetails').innerHTML = `
                        <p><strong>Session ID:</strong> ${sessionId}</p>
                        <p><strong>Files Processed:</strong> ${result.total_documents}</p>
                        <p><strong>Document Chunks:</strong> ${result.total_chunks}</p>
                    `;
                    
                    // Update chat message
                    addMessage(`Documents processed successfully! ${result.total_documents} files loaded with ${result.total_chunks} text chunks. You can now ask me questions or request analysis.`, false);
                    
                    processBtn.innerHTML = '<i class="fas fa-check"></i> Processed';
                    processBtn.classList.remove('btn-primary');
                    processBtn.classList.add('btn-success');
                    
                } else {
                    throw new Error(result.error || 'Processing failed');
                }
                
            } catch (error) {
                alert(`Error processing documents: ${error.message}`);
                processBtn.innerHTML = '<i class="fas fa-cogs"></i> Process Documents';
                processBtn.disabled = false;
            }
        }
        
        // Analyze documents
        async function analyzeDocuments() {
            const question = document.getElementById('questionInput').value.trim();
            if (!question) {
                alert('Please enter a question for analysis');
                return;
            }
            
            const useMoE = document.getElementById('useMoE').checked;
            const analyzeBtn = document.getElementById('analyzeBtn');
            
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            analyzeBtn.disabled = true;
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        session_id: sessionId,
                        question: question,
                        use_moe: useMoE
                    })
                });
                
                const result = await response.json();
                
                if (result.response) {
                    const analysisResult = document.getElementById('analysisResult');
                    analysisResult.style.display = 'block';
                    
                    // Convert markdown to HTML
                    const htmlContent = marked.parse(result.response);
                    
                    let expertInfo = '';
                    if (result.experts_used && result.experts_used.length > 0) {
                        expertInfo = `<div class="mb-3"><small class="text-muted"><strong>Experts Used:</strong> ${result.experts_used.map(e => e.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())).join(', ')}</small></div>`;
                    }
                    
                    analysisResult.innerHTML = `
                        <h5>Analysis Results</h5>
                        ${expertInfo}
                        <div class="analysis-content">${htmlContent}</div>
                    `;
                    
                    // Add to chat
                    addMessage(question, true);
                    addMessage(result.response, false);
                    
                } else {
                    throw new Error(result.error || 'Analysis failed');
                }
                
            } catch (error) {
                alert(`Analysis error: ${error.message}`);
            } finally {
                analyzeBtn.innerHTML = '<i class="fas fa-brain"></i> Analyze Documents';
                analyzeBtn.disabled = false;
            }
        }
        
        // Add message to chat
        function addMessage(message, isUser) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            
            if (isUser) {
                messageDiv.innerHTML = `<strong>You:</strong><br>${message}`;
            } else {
                const htmlMessage = marked.parse(message);
                messageDiv.innerHTML = `<strong>AI Assistant:</strong><br>${htmlMessage}`;
            }
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Send chat message
        async function sendMessage() {
            const chatInput = document.getElementById('chatInput');
            const message = chatInput.value.trim();
            
            if (!message || !sessionId) return;
            
            addMessage(message, true);
            chatInput.value = '';
            
            const chatLoading = document.getElementById('chatLoading');
            chatLoading.style.display = 'block';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        session_id: sessionId,
                        message: message
                    })
                });
                
                const result = await response.json();
                
                if (result.response) {
                    addMessage(result.response, false);
                } else {
                    addMessage(`Error: ${result.error || 'Chat failed'}`, false);
                }
                
            } catch (error) {
                addMessage(`Connection error: ${error.message}`, false);
            } finally {
                chatLoading.style.display = 'none';
            }
        }
        
        // Event listeners
        document.addEventListener('DOMContentLoaded', function() {
            initializeDragDrop();
            
            document.getElementById('fileInput').addEventListener('change', function(e) {
                handleFileSelection(Array.from(e.target.files));
            });
            
            document.getElementById('processBtn').addEventListener('click', processDocuments);
            document.getElementById('analyzeBtn').addEventListener('click', analyzeDocuments);
            document.getElementById('sendBtn').addEventListener('click', sendMessage);
            
            document.getElementById('chatInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload")
async def upload_files(request: Request):
    """Handle multiple file uploads"""
    form = await request.form()
    session_id = form.get("session_id")
    files = form.getlist("files")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    processor = DocumentProcessor()
    result = await processor.process_files(files, session_id)
    
    return JSONResponse(result)

@app.post("/analyze")
async def analyze_documents(request: AnalysisRequest):
    """Analyze documents using MoE"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not found")
    
    session = sessions[request.session_id]
    documents = session["documents"]
    
    try:
        result = await moe_analyzer.analyze_documents(
            request.question, 
            documents, 
            request.use_moe
        )
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/chat")
async def chat_with_documents(request: ChatRequest):
    """Chat with uploaded documents"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not found")
    
    session = sessions[request.session_id]
    vectorstore = session["vectorstore"]
    
    try:
        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.get_relevant_documents(request.message)
        
        # Prepare context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate response using single expert (chat mode)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3
        )
        
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided document context.

Context from documents:
{context}

User question: {request.message}

Please provide a comprehensive answer based on the document context. If the question cannot be answered from the provided context, say so clearly."""
        
        response = llm.invoke(prompt)
        
        # Store in chat history
        session["chat_history"].append({
            "user": request.message,
            "assistant": response.content,
            "timestamp": pd.Timestamp.now()
        })
        
        return JSONResponse({
            "response": response.content,
            "relevant_docs_count": len(relevant_docs)
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get session information"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return SessionInfo(
        session_id=session_id,
        file_count=len(session["processed_files"]),
        total_chunks=len(session["documents"]),
        status="active"
    )

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
        return JSONResponse({"message": "Session deleted successfully"})
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "active_sessions": len(sessions),
        "gemini_configured": GEMINI_API_KEY is not None
    })

if __name__ == "__main__":
    print("Starting Gemini MoE Document Analyzer...")
    print("Features:")
    print("- Multi-file upload (PDF, DOC, PPT, XLS, CSV, TXT, etc.)")
    print("- Mixture of Experts analysis")
    print("- Interactive chat with documents")
    print("- Session management")
    print("\nMake sure to set your GEMINI_API_KEY in the code!")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
        log_level="info"
    )