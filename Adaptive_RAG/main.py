from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any, TypedDict, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import shutil
import uuid
import google.generativeai as genai
from pathlib import Path
import json
from datetime import datetime
import asyncio
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Load environment variables
load_dotenv()

# Configure API keys
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
tavily_api_key = os.getenv("TAVILY_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Create directories
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# ----------------------------
# 🔐 Graph State Definition
# ----------------------------
class GraphState(TypedDict, total=False):
    question: str
    document_context: str
    retrieved_docs: List[str]
    grade: str
    generation: str
    web_search_results: str
    rewritten_question: str
    final_answer: str
    iterations: int
    max_iterations: int

# ----------------------------
# 🤖 LLM Configuration
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    max_retries=3,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# Web search tool
web_search_tool = TavilySearchResults(
    k=3,
    tavily_api_key=tavily_api_key
) if tavily_api_key else None

# ----------------------------
# 📄 Document Processing
# ----------------------------
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vectorstore = None
        self.compression_retriever = None
    
    def process_pdf(self, file_path: str):
        """Process PDF and create vector store"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Combine all text for context
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        return combined_text
    
    def retrieve_docs(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documents"""
        if self.compression_retriever:
            docs = self.compression_retriever.get_relevant_documents(query)
            return [doc.page_content for doc in docs]
        elif self.vectorstore:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        else:
            return []

# Global document processor instance
doc_processor = DocumentProcessor()

# ----------------------------
# 🎯 Prompt Templates
# ----------------------------
GRADING_PROMPT = PromptTemplate.from_template("""
You are a grader assessing relevance of retrieved documents to a user question.

Retrieved Documents:
{documents}

User Question: {question}

If the documents contain information that can help answer the question, grade as "relevant".
If the documents do not contain useful information to answer the question, grade as "not relevant".

Provide a binary score 'relevant' or 'not relevant' based on whether the documents are relevant to the question.

Grade:""")

GENERATION_PROMPT = PromptTemplate.from_template("""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.

Context:
{context}

Question: {question}

Provide a comprehensive and accurate answer based on the context. If the context doesn't contain enough information to fully answer the question, clearly state what information is missing.

Answer:""")

HALLUCINATION_GRADER_PROMPT = PromptTemplate.from_template("""
You are a grader assessing whether an answer is grounded in / supported by a set of facts.

Facts:
{documents}

Answer: {generation}

Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in the facts.

Score:""")

ANSWER_GRADER_PROMPT = PromptTemplate.from_template("""
You are a grader assessing whether an answer addresses the user question.

Question: {question}
Answer: {generation}

Give a binary score 'yes' or 'no' to indicate whether the answer addresses the question.

Score:""")

QUESTION_REWRITER_PROMPT = PromptTemplate.from_template("""
You are a question re-writer. Your task is to re-write the input question to be more specific and better suited for vectorstore retrieval.

Original Question: {question}

Rewritten Question:""")

WEB_SEARCH_PROMPT = PromptTemplate.from_template("""
You are an assistant for question-answering tasks. Use the following web search results and document context to answer the question.

Document Context:
{context}

Web Search Results:
{web_results}

Question: {question}

Provide a comprehensive answer combining information from both the document context and web search results.

Answer:""")

# ----------------------------
# 🔄 LangGraph Nodes
# ----------------------------
def retrieve_node(state: GraphState):
    """Retrieve documents based on the question"""
    question = state["question"]
    
    # Retrieve documents
    documents = doc_processor.retrieve_docs(question)
    
    return {
        **state,
        "retrieved_docs": documents,
        "iterations": state.get("iterations", 0)
    }

def grade_documents_node(state: GraphState):
    """Grade retrieved documents for relevance"""
    question = state["question"]
    documents = state["retrieved_docs"]
    
    # Combine documents for grading
    docs_text = "\n\n".join(documents)
    
    # Create grading chain
    grading_chain = LLMChain(
        llm=llm,
        prompt=GRADING_PROMPT,
        output_key="grade"
    )
    
    # Grade documents
    result = grading_chain.run(
        documents=docs_text,
        question=question
    )
    
    grade = result.strip().lower()
    
    return {
        **state,
        "grade": grade
    }

def generate_node(state: GraphState):
    """Generate answer based on retrieved documents"""
    question = state["question"]
    documents = state["retrieved_docs"]
    
    # Combine documents
    context = "\n\n".join(documents)
    
    # Create generation chain
    generation_chain = LLMChain(
        llm=llm,
        prompt=GENERATION_PROMPT,
        output_key="generation"
    )
    
    # Generate answer
    result = generation_chain.run(
        context=context,
        question=question
    )
    
    return {
        **state,
        "generation": result,
        "iterations": state.get("iterations", 0) + 1
    }

def grade_generation_node(state: GraphState):
    """Grade generation for hallucinations and question answering"""
    question = state["question"]
    documents = state["retrieved_docs"]
    generation = state["generation"]
    
    # Combine documents
    docs_text = "\n\n".join(documents)
    
    # Check for hallucinations
    hallucination_chain = LLMChain(
        llm=llm,
        prompt=HALLUCINATION_GRADER_PROMPT,
        output_key="score"
    )
    
    hallucination_score = hallucination_chain.run(
        documents=docs_text,
        generation=generation
    ).strip().lower()
    
    # Check if answer addresses question
    answer_chain = LLMChain(
        llm=llm,
        prompt=ANSWER_GRADER_PROMPT,
        output_key="score"
    )
    
    answer_score = answer_chain.run(
        question=question,
        generation=generation
    ).strip().lower()
    
    return {
        **state,
        "hallucination_score": hallucination_score,
        "answer_score": answer_score
    }

def rewrite_question_node(state: GraphState):
    """Rewrite question for better retrieval"""
    question = state["question"]
    
    # Create rewriting chain
    rewrite_chain = LLMChain(
        llm=llm,
        prompt=QUESTION_REWRITER_PROMPT,
        output_key="rewritten_question"
    )
    
    rewritten_question = rewrite_chain.run(question=question)
    
    return {
        **state,
        "rewritten_question": rewritten_question,
        "question": rewritten_question  # Update question for next iteration
    }

def web_search_node(state: GraphState):
    """Perform web search and generate answer"""
    question = state["question"]
    context = state["document_context"]
    
    if not web_search_tool:
        return {
            **state,
            "final_answer": "Web search is not available. Please configure TAVILY_API_KEY."
        }
    
    # Perform web search
    web_results = web_search_tool.run(question)
    
    # Create web search chain
    web_search_chain = LLMChain(
        llm=llm,
        prompt=WEB_SEARCH_PROMPT,
        output_key="answer"
    )
    
    # Generate answer with web results
    result = web_search_chain.run(
        context=context,
        web_results=web_results,
        question=question
    )
    
    return {
        **state,
        "web_search_results": web_results,
        "final_answer": result
    }

# ----------------------------
# 🔄 Decision Functions
# ----------------------------
def decide_to_generate(state: GraphState):
    """Decide whether to generate answer or search web"""
    grade = state["grade"]
    
    if grade == "relevant":
        return "generate"
    else:
        return "web_search"

def grade_generation_decision(state: GraphState):
    """Decide based on generation grading"""
    hallucination_score = state.get("hallucination_score", "no")
    answer_score = state.get("answer_score", "no")
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)
    
    # If too many iterations, return current generation
    if iterations >= max_iterations:
        return "end"
    
    # If generation is good, return it
    if hallucination_score == "yes" and answer_score == "yes":
        return "end"
    
    # If generation has issues, rewrite question and try again
    return "rewrite"

# ----------------------------
# 🏗️ Build Graph
# ----------------------------
def build_graph():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_generation", grade_generation_node)
    workflow.add_node("rewrite_question", rewrite_question_node)
    workflow.add_node("web_search", web_search_node)
    
    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "web_search": "web_search"
        }
    )
    workflow.add_edge("generate", "grade_generation")
    workflow.add_conditional_edges(
        "grade_generation",
        grade_generation_decision,
        {
            "end": END,
            "rewrite": "rewrite_question"
        }
    )
    workflow.add_edge("rewrite_question", "retrieve")
    workflow.add_edge("web_search", END)
    
    return workflow.compile()

# Create compiled graph
app_graph = build_graph()

# ----------------------------
# 🌐 FastAPI Application
# ----------------------------
app = FastAPI(title="Adaptive RAG System", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_upload_form():
    """Serve the main upload form"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Adaptive RAG System</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            .gradient-bg {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .upload-area {
                border: 2px dashed #667eea;
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                transition: border-color 0.3s;
            }
            .upload-area:hover {
                border-color: #764ba2;
            }
            .workflow-step {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 1rem;
                margin: 0.5rem 0;
                border-left: 4px solid #667eea;
            }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg gradient-bg">
            <div class="container">
                <a class="navbar-brand" href="#">
                    <i class="fas fa-brain me-2"></i>
                    Adaptive RAG System
                </a>
            </div>
        </nav>

        <div class="container mt-4">
            <div class="row">
                <div class="col-md-8">
                    <div class="card shadow">
                        <div class="card-header gradient-bg">
                            <h3 class="mb-0">
                                <i class="fas fa-upload me-2"></i>
                                Upload PDF Document
                            </h3>
                        </div>
                        <div class="card-body">
                            <form method="post" enctype="multipart/form-data" action="/analyze" id="uploadForm">
                                <div class="mb-3">
                                    <label class="form-label">Select PDF Document:</label>
                                    <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                                        <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                                        <p class="mb-0">Click to select PDF file</p>
                                        <small class="text-muted">Maximum file size: 10MB</small>
                                    </div>
                                    <input type="file" class="form-control d-none" name="file" id="fileInput" accept=".pdf" required>
                                    <div id="fileName" class="mt-2"></div>
                                </div>
                                
                                <div class="mb-3">
                                    <label for="question" class="form-label">Your Question:</label>
                                    <textarea class="form-control" name="question" id="question" rows="3" required 
                                              placeholder="Ask a question about the document..."></textarea>
                                </div>

                                <div class="mb-3">
                                    <label for="maxIterations" class="form-label">Max Iterations:</label>
                                    <select class="form-select" name="max_iterations" id="maxIterations">
                                        <option value="1">1 (Quick)</option>
                                        <option value="2">2 (Balanced)</option>
                                        <option value="3" selected>3 (Thorough)</option>
                                        <option value="4">4 (Comprehensive)</option>
                                    </select>
                                </div>

                                <button type="submit" class="btn btn-primary btn-lg w-100">
                                    <i class="fas fa-search me-2"></i>
                                    Analyze Document
                                </button>
                            </form>
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="card shadow">
                        <div class="card-header bg-light">
                            <h5 class="mb-0">
                                <i class="fas fa-route me-2"></i>
                                Adaptive RAG Workflow
                            </h5>
                        </div>
                        <div class="card-body">
                            <div class="workflow-step">
                                <strong>1. Query Analysis</strong>
                                <p class="mb-0 small">Analyze the input question</p>
                            </div>
                            <div class="workflow-step">
                                <strong>2. Document Retrieval</strong>
                                <p class="mb-0 small">Find relevant document chunks</p>
                            </div>
                            <div class="workflow-step">
                                <strong>3. Relevance Grading</strong>
                                <p class="mb-0 small">Grade retrieved documents</p>
                            </div>
                            <div class="workflow-step">
                                <strong>4. Generation</strong>
                                <p class="mb-0 small">Generate answer or search web</p>
                            </div>
                            <div class="workflow-step">
                                <strong>5. Self-Reflection</strong>
                                <p class="mb-0 small">Check for hallucinations</p>
                            </div>
                            <div class="workflow-step">
                                <strong>6. Answer/Rewrite</strong>
                                <p class="mb-0 small">Return answer or rewrite question</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const fileName = document.getElementById('fileName');
                if (e.target.files.length > 0) {
                    fileName.innerHTML = `<div class="alert alert-info">
                        <i class="fas fa-file-pdf me-2"></i>${e.target.files[0].name}
                    </div>`;
                }
            });

            document.getElementById('uploadForm').addEventListener('submit', function(e) {
                const submitBtn = this.querySelector('button[type="submit"]');
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
                submitBtn.disabled = true;
            });
        </script>
    </body>
    </html>
    """)

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_document(
    file: UploadFile = File(...),
    question: str = Form(...),
    max_iterations: int = Form(3)
):
    """Analyze uploaded PDF document"""
    
    # Create temporary directory
    temp_dir = f"temp_uploads/{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Save uploaded file
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        document_context = doc_processor.process_pdf(file_path)
        
        # Initialize state
        initial_state = {
            "question": question,
            "document_context": document_context,
            "max_iterations": max_iterations,
            "iterations": 0
        }
        
        # Run adaptive RAG workflow
        result = app_graph.invoke(initial_state)
        
        # Get final answer
        final_answer = result.get("final_answer", result.get("generation", "No answer generated"))
        
        # Render answer as markdown
        from markdown import markdown
        final_answer_html = markdown(final_answer)
        
        # Build response HTML
        response_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Analysis Results - Adaptive RAG</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                .gradient-bg {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
                .answer-card {{ border-left: 4px solid #667eea; }}
                .process-step {{ background: #f8f9fa; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }}
                .process-step.active {{ background: #e3f2fd; border-left: 4px solid #2196f3; }}
            </style>
        </head>
        <body>
            <nav class="navbar navbar-expand-lg gradient-bg">
                <div class="container">
                    <a class="navbar-brand" href="/">
                        <i class="fas fa-brain me-2"></i>
                        Adaptive RAG System
                    </a>
                </div>
            </nav>

            <div class="container mt-4">
                <div class="row">
                    <div class="col-md-8">
                        <div class="card shadow answer-card">
                            <div class="card-header gradient-bg">
                                <h4 class="mb-0">
                                    <i class="fas fa-lightbulb me-2"></i>
                                    Analysis Results
                                </h4>
                            </div>
                            <div class="card-body">
                                <div class="mb-3">
                                    <h6 class="text-muted">Question:</h6>
                                    <p class="border-start border-3 border-primary ps-3">{question}</p>
                                </div>
                                
                                <div class="mb-3">
                                    <h6 class="text-muted">Document:</h6>
                                    <p><i class="fas fa-file-pdf text-danger me-2"></i>{file.filename}</p>
                                </div>
                                
                                <div class="mb-3">
                                    <h6 class="text-muted">Answer:</h6>
                                    <div class="bg-light p-3 rounded">
                                        {final_answer_html}
                                    </div>
                                </div>
                                
                                <div class="d-flex gap-2">
                                    <a href="/" class="btn btn-primary">
                                        <i class="fas fa-plus me-2"></i>
                                        New Analysis
                                    </a>
                                    <button class="btn btn-outline-secondary" onclick="window.print()">
                                        <i class="fas fa-print me-2"></i>
                                        Print
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-4">
                        <div class="card shadow">
                            <div class="card-header bg-light">
                                <h5 class="mb-0">
                                    <i class="fas fa-cogs me-2"></i>
                                    Process Information
                                </h5>
                            </div>
                            <div class="card-body">
                                <div class="process-step active">
                                    <strong>Iterations:</strong> {result.get('iterations', 0)}
                                </div>
                                <div class="process-step">
                                    <strong>Max Iterations:</strong> {max_iterations}
                                </div>
                                <div class="process-step">
                                    <strong>Document Grade:</strong> {result.get('grade', 'N/A')}
                                </div>
                                <div class="process-step">
                                    <strong>Hallucination Check:</strong> {result.get('hallucination_score', 'N/A')}
                                </div>
                                <div class="process-step">
                                    <strong>Answer Quality:</strong> {result.get('answer_score', 'N/A')}
                                </div>
                                {"<div class='process-step'><strong>Web Search:</strong> Used</div>" if result.get('web_search_results') else ""}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """
        
        return HTMLResponse(response_html)
        
    except Exception as e:
        return HTMLResponse(f"""
        <div class="container mt-5">
            <div class="alert alert-danger">
                <h4>Error Processing Document</h4>
                <p>An error occurred while processing your document: {str(e)}</p>
                <a href="/" class="btn btn-primary">Try Again</a>
            </div>
        </div>
        """, status_code=500)
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Adaptive RAG System...")
    print("📋 Features:")
    print("   ✅ Self-reflection mechanism")
    print("   ✅ Document relevance grading")
    print("   ✅ Hallucination detection")
    print("   ✅ Question rewriting")
    print("   ✅ Web search fallback")
    print("   ✅ Adaptive iterations")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )