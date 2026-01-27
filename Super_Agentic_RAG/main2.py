"""
Traditional RAG System with RAGAS Evaluation
Simple retrieval-augmented generation using LangChain and Gemini 2.5 Flash
Includes RAGAS evaluation metrics for answer quality assessment
"""

import os
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# RAGAS imports
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Use legacy metrics (more compatible with current setup)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='ragas')

from ragas.metrics import (
    answer_correctness,
    context_precision,
    faithfulness,
    answer_relevancy
)
from ragas import evaluate
from datasets import Dataset

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import uuid

# Load environment variables
load_dotenv()

# ===========================
# Configuration
# ===========================

# Google API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LangSmith tracing (optional)
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# Initialize LLM - Gemini 2.5 Flash
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    max_retries=2,
) if GOOGLE_API_KEY else None

# Initialize Embeddings - HuggingFace for vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize RAGAS LLM and Embeddings (wrap existing LangChain objects)
if GOOGLE_API_KEY and gemini_llm:
    # Wrap LangChain LLM for RAGAS compatibility
    ragas_llm = LangchainLLMWrapper(gemini_llm)

    # Wrap LangChain embeddings for RAGAS compatibility
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
else:
    ragas_llm = None
    ragas_embeddings = None

# Rate limiting for Gemini API
GEMINI_CALL_DELAY = 1  # 1 second delay between calls

# ===========================
# Document Processing
# ===========================

def detect_file_type(file_path: str) -> str:
    """Detect file type based on extension"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return 'pdf'
    elif ext in ['.doc', '.docx']:
        return 'word'
    elif ext in ['.ppt', '.pptx']:
        return 'powerpoint'
    elif ext == '.txt':
        return 'text'
    elif ext == '.csv':
        return 'csv'
    elif ext in ['.xls', '.xlsx']:
        return 'excel'
    else:
        return 'unknown'

def load_document(file_path: str) -> List:
    """Load document based on file type"""
    file_type = detect_file_type(file_path)

    if file_type == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_type == 'word':
        loader = Docx2txtLoader(file_path)
    elif file_type == 'powerpoint':
        loader = UnstructuredPowerPointLoader(file_path)
    elif file_type == 'text':
        loader = TextLoader(file_path)
    elif file_type == 'csv':
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return loader.load()

# ===========================
# Traditional RAG Class
# ===========================

class TraditionalRAG:
    """
    Traditional RAG implementation with RAGAS evaluation
    Simple and straightforward: retrieve -> generate -> evaluate
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.vectorstore = None
        self.documents = []
        self.chunks = []

    def process_documents(self, documents: List) -> str:
        """Process documents and create vector store"""
        # Split documents into chunks
        self.chunks = self.text_splitter.split_documents(documents)
        self.documents = documents

        # Create vector store
        self.vectorstore = FAISS.from_documents(self.chunks, embeddings)

        # Return summary
        return f"Processed {len(documents)} documents into {len(self.chunks)} chunks"

    def retrieve(self, query: str, k: int = 4) -> List[str]:
        """Retrieve relevant documents"""
        if not self.vectorstore:
            return []

        # Perform similarity search
        retrieved_docs = self.vectorstore.similarity_search(query, k=k)

        # Extract content
        contexts = [doc.page_content for doc in retrieved_docs]
        return contexts

    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """Generate answer using LLM with retrieved contexts"""
        if not gemini_llm:
            return "LLM not initialized. Please set GOOGLE_API_KEY"

        # Combine contexts
        context_text = "\n\n".join(contexts)

        # Create prompt
        prompt_template = """You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, concise, and accurate answer
- Base your answer strictly on the provided context
- If the context doesn't contain enough information, say so
- Use markdown formatting for better readability

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create chain
        llm_chain = LLMChain(llm=gemini_llm, prompt=prompt)

        # Generate answer
        answer = llm_chain.run(context=context_text, question=query)

        # Add delay for rate limiting
        time.sleep(GEMINI_CALL_DELAY)

        return answer.strip()

    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        """
        Main query method: retrieve + generate

        Args:
            question: User's question
            k: Number of documents to retrieve

        Returns:
            Dictionary with answer, contexts, and metadata
        """
        # Step 1: Retrieve relevant contexts
        contexts = self.retrieve(question, k=k)

        if not contexts:
            return {
                "question": question,
                "answer": "No relevant documents found.",
                "contexts": [],
                "num_contexts": 0
            }

        # Step 2: Generate answer
        answer = self.generate_answer(question, contexts)

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "num_contexts": len(contexts)
        }

    def evaluate_with_ragas(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate the RAG response using RAGAS metrics

        Args:
            question: User's question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer (for AnswerCorrectness)

        Returns:
            Dictionary with evaluation scores
        """
        if not ragas_llm or not ragas_embeddings:
            return {
                "error": "RAGAS not initialized. Please set GOOGLE_API_KEY"
            }

        # Prepare data for RAGAS
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        # Add ground truth if provided
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        # Create dataset
        dataset = Dataset.from_dict(data)

        # Define metrics using legacy API (already initialized singletons)
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
        ]

        # Add answer_correctness only if ground_truth is provided
        if ground_truth:
            metrics.append(answer_correctness)

        try:
            # Run evaluation with LLM and embeddings
            results = evaluate(
                dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings
            )

            # Convert EvaluationResult to dict - access scores property
            # Results is an EvaluationResult object, convert to pandas then to dict
            results_df = results.to_pandas()

            # Extract scores from the first row (we only have one sample)
            scores = {
                "faithfulness": float(results_df["faithfulness"].iloc[0]) if "faithfulness" in results_df else 0.0,
                "answer_relevancy": float(results_df["answer_relevancy"].iloc[0]) if "answer_relevancy" in results_df else 0.0,
                "context_precision": float(results_df["context_precision"].iloc[0]) if "context_precision" in results_df else 0.0,
            }

            if ground_truth and "answer_correctness" in results_df:
                scores["answer_correctness"] = float(results_df["answer_correctness"].iloc[0])

            return scores

        except Exception as e:
            return {
                "error": f"RAGAS evaluation failed: {str(e)}"
            }

    def query_with_evaluation(
        self,
        question: str,
        k: int = 4,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query with automatic RAGAS evaluation

        Args:
            question: User's question
            k: Number of documents to retrieve
            ground_truth: Optional ground truth answer

        Returns:
            Dictionary with answer, contexts, and evaluation scores
        """
        # Get answer
        result = self.query(question, k=k)

        # Evaluate
        if result.get("contexts"):
            evaluation = self.evaluate_with_ragas(
                question=result["question"],
                answer=result["answer"],
                contexts=result["contexts"],
                ground_truth=ground_truth
            )
            result["evaluation"] = evaluation

        return result

# ===========================
# FastAPI Application
# ===========================

app = FastAPI(
    title="Traditional RAG with RAGAS Evaluation",
    description="Simple RAG system with quality evaluation",
    version="1.0.0"
)

# Session storage
sessions = {}

# Upload directory
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic models
class SessionResponse(BaseModel):
    session_id: str
    file_name: str
    file_type: str
    message: str
    num_chunks: int

class QueryRequest(BaseModel):
    session_id: str
    query: str
    k: int = 4
    evaluate: bool = True
    ground_truth: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    num_contexts: int
    evaluation: Optional[Dict[str, Any]] = None
    error: bool = False

# HTML Frontend
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
    <title>Traditional RAG with RAGAS</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .upload-section, .query-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .upload-section h2, .query-section h2 {
            color: #764ba2;
            margin-top: 0;
        }
        input[type="file"], input[type="text"], textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
            margin-bottom: 10px;
            box-sizing: border-box;
        }
        textarea {
            min-height: 80px;
            resize: vertical;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .response {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            border-left: 4px solid #667eea;
        }
        .response h3 {
            color: #764ba2;
            margin-top: 0;
        }
        .evaluation-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .metric {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric-name {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }
        .metric-bar {
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }
        .metric-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
        }
        .status {
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .checkbox-container {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        .checkbox-container input[type="checkbox"] {
            width: auto;
            margin-right: 10px;
        }
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Traditional RAG with RAGAS Evaluation</h1>
        <p class="subtitle">Upload documents, ask questions, and evaluate answer quality</p>

        <!-- Upload Section -->
        <div class="upload-section">
            <h2>📁 Step 1: Upload Document</h2>
            <input type="file" id="fileInput" accept=".pdf,.docx,.txt,.pptx,.csv">
            <button onclick="uploadFile()">Upload Document</button>
            <div id="uploadStatus"></div>
        </div>

        <!-- Query Section -->
        <div class="query-section">
            <h2>💬 Step 2: Ask Questions</h2>
            <textarea id="queryInput" placeholder="Enter your question here..."></textarea>

            <div class="checkbox-container">
                <input type="checkbox" id="evaluateCheckbox" checked>
                <label for="evaluateCheckbox">Enable RAGAS Evaluation (Faithfulness, Relevancy, Context Precision)</label>
            </div>

            <input type="text" id="groundTruthInput" placeholder="(Optional) Ground truth answer for Answer Correctness metric">

            <label for="kInput">Number of documents to retrieve (k):</label>
            <input type="number" id="kInput" value="4" min="1" max="10">

            <button onclick="askQuestion()" id="queryBtn" disabled>Ask Question</button>
            <div id="queryStatus"></div>
        </div>

        <!-- Response Section -->
        <div id="responseSection"></div>
    </div>

    <script>
        let sessionId = null;

        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];

            if (!file) {
                alert('Please select a file');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            document.getElementById('uploadStatus').innerHTML = '<div class="loader"></div>';

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    sessionId = data.session_id;
                    document.getElementById('uploadStatus').innerHTML =
                        `<div class="status success">✅ ${data.message}<br>
                        📊 Processed into ${data.num_chunks} chunks</div>`;
                    document.getElementById('queryBtn').disabled = false;
                } else {
                    document.getElementById('uploadStatus').innerHTML =
                        `<div class="status error">❌ Upload failed: ${data.detail}</div>`;
                }
            } catch (error) {
                document.getElementById('uploadStatus').innerHTML =
                    `<div class="status error">❌ Error: ${error.message}</div>`;
            }
        }

        async function askQuestion() {
            const query = document.getElementById('queryInput').value.trim();
            const k = parseInt(document.getElementById('kInput').value);
            const evaluate = document.getElementById('evaluateCheckbox').checked;
            const groundTruth = document.getElementById('groundTruthInput').value.trim();

            if (!query) {
                alert('Please enter a question');
                return;
            }

            document.getElementById('queryStatus').innerHTML = '<div class="loader"></div>';
            document.getElementById('responseSection').innerHTML = '';

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        session_id: sessionId,
                        query: query,
                        k: k,
                        evaluate: evaluate,
                        ground_truth: groundTruth || null
                    })
                });

                const data = await response.json();

                document.getElementById('queryStatus').innerHTML = '';

                if (!data.error) {
                    let html = `
                        <div class="response">
                            <h3>📝 Answer</h3>
                            <div>${formatMarkdown(data.answer)}</div>
                            <p style="color: #666; font-size: 12px; margin-top: 15px;">
                                Retrieved ${data.num_contexts} relevant document chunks
                            </p>
                    `;

                    if (data.evaluation && !data.evaluation.error) {
                        html += `
                            <h3 style="margin-top: 25px;">📊 RAGAS Evaluation Metrics</h3>
                            <div class="evaluation-metrics">
                        `;

                        for (const [metric, value] of Object.entries(data.evaluation)) {
                            if (metric !== 'error') {
                                const percentage = (value * 100).toFixed(1);
                                const displayName = metric.replace(/_/g, ' ').toUpperCase();
                                html += `
                                    <div class="metric">
                                        <div class="metric-name">${displayName}</div>
                                        <div class="metric-value">${percentage}%</div>
                                        <div class="metric-bar">
                                            <div class="metric-bar-fill" style="width: ${percentage}%"></div>
                                        </div>
                                    </div>
                                `;
                            }
                        }

                        html += `</div>`;
                    } else if (data.evaluation && data.evaluation.error) {
                        html += `<p style="color: #721c24;">⚠️ Evaluation error: ${data.evaluation.error}</p>`;
                    }

                    html += `</div>`;
                    document.getElementById('responseSection').innerHTML = html;
                } else {
                    document.getElementById('queryStatus').innerHTML =
                        `<div class="status error">❌ ${data.answer}</div>`;
                }
            } catch (error) {
                document.getElementById('queryStatus').innerHTML =
                    `<div class="status error">❌ Error: ${error.message}</div>`;
            }
        }

        function formatMarkdown(text) {
            // Simple markdown formatting
            return text
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/\n/g, '<br>');
        }

        // Allow Enter key to submit query
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!document.getElementById('queryBtn').disabled) {
                    askQuestion();
                }
            }
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the web interface"""
    return HTML_TEMPLATE

@app.post("/api/upload", response_model=SessionResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())

        # Save file
        file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Detect file type
        file_type = detect_file_type(file_path)

        if file_type == 'unknown':
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Load and process documents
        documents = load_document(file_path)

        # Create RAG instance
        rag = TraditionalRAG(chunk_size=1500, chunk_overlap=300)
        summary = rag.process_documents(documents)

        # Store session
        sessions[session_id] = {
            "file_path": file_path,
            "file_name": file.filename,
            "file_type": file_type,
            "rag": rag,
            "created_at": datetime.now().isoformat(),
            "queries": []
        }

        return SessionResponse(
            session_id=session_id,
            file_name=file.filename,
            file_type=file_type,
            message=f"File uploaded successfully. {summary}",
            num_chunks=len(rag.chunks)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Query the document with optional RAGAS evaluation"""
    try:
        # Validate session
        if request.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found. Please upload a file first.")

        session = sessions[request.session_id]
        rag = session["rag"]

        # Process query
        if request.evaluate:
            result = rag.query_with_evaluation(
                question=request.query,
                k=request.k,
                ground_truth=request.ground_truth
            )
        else:
            result = rag.query(question=request.query, k=request.k)

        # Store query in session
        session["queries"].append({
            "query": request.query,
            "answer": result.get("answer"),
            "timestamp": datetime.now().isoformat(),
            "evaluation": result.get("evaluation")
        })

        return QueryResponse(
            answer=result.get("answer", "No answer generated"),
            num_contexts=result.get("num_contexts", 0),
            evaluation=result.get("evaluation"),
            error=False
        )

    except HTTPException:
        raise
    except Exception as e:
        return QueryResponse(
            answer=f"Error processing query: {str(e)}",
            num_contexts=0,
            error=True
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gemini_llm": gemini_llm is not None,
        "ragas_enabled": ragas_llm is not None,
        "active_sessions": len(sessions)
    }

# ===========================
# Main Entry Point
# ===========================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("TRADITIONAL RAG WITH RAGAS EVALUATION")
    print("="*60)
    print("\nStarting server...")
    print("Open your browser and go to: http://localhost:8001")
    print("\nPress Ctrl+C to stop the server\n")
    print("Features:")
    print("  ✅ Simple Traditional RAG (Retrieve + Generate)")
    print("  ✅ RAGAS Evaluation Metrics:")
    print("     - Faithfulness: Answer grounded in context")
    print("     - Answer Relevancy: Answer addresses the question")
    print("     - Context Precision: Retrieved docs are relevant")
    print("     - Answer Correctness: Compared to ground truth (optional)")
    print("\n")

    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
