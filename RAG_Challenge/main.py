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
import shutil
from werkzeug.utils import secure_filename

load_dotenv()

# ---- FastAPI Setup ----
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---- Configuration ----
PDF_REPORTS_DIR = "pdf_report"
EMBEDDING_DIR = "vectorstore"
MODEL_NAME = "gemini-2.0-flash"

os.makedirs(PDF_REPORTS_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

api_key = os.getenv("GOOGLE_API_KEY")
assert api_key, "GOOGLE_API_KEY not found"
genai.configure(api_key=api_key)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ---- LangGraph State ----
class State(TypedDict):
    messages: list[str]
    user_input: str

# ---- Document Processing ----
def load_pdf_reports():
    pdf_files = list(Path(PDF_REPORTS_DIR).glob("*.pdf"))
    docs = []
    for pdf in pdf_files:
        try:
            loader = UnstructuredPDFLoader(str(pdf))
            docs.extend(loader.load())
        except Exception as e:
            logging.error(f"Failed to load {pdf}: {e}")
    return docs


def build_vectorstore():
    pdf_files = list(Path(PDF_REPORTS_DIR).glob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files.")
    documents = []
    for pdf in pdf_files:
        try:
            logging.info(f"Loading PDF: {pdf}")
            loader = PyPDFLoader(str(pdf))
            documents.extend(loader.load())
            logging.info(f"Loaded PDF: {pdf}")
        except Exception as e:
            logging.error(f"Failed to load {pdf}: {e}")

    logging.info(f"Loaded {len(documents)} documents from PDFs.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=200)
    logging.info("Splitting documents...")
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(chunks)} chunks.")

    logging.info("Starting embedding...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    logging.info("Embedding done.")
    return vector_db

# ---- Initialize Vector DB ----
vector_db = build_vectorstore()

# ---- LangGraph Tasks ----
def ask_question_with_rag(state: State):
    user_input = state["user_input"]

    # Perform similarity search based on user input
    retrieved_docs = vector_db.similarity_search(user_input)

    # Use the same prompt as the template
    prompt_template = """You are given the following documents: {docs}
Based on these documents, answer the following question: {question}
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["docs", "question"])
    
    # Run LLMChain with the joined text content
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    result = llm_chain.run(docs=context, question=user_input)

    return {"messages": [result], "user_input": user_input}


def render_result_as_html(state: State):
    answer = state["messages"][-1]
    html = markdown.markdown(answer)
    html = f"""
    <div class='mt-4'><b>Question:</b> {state['user_input']}</div>
    <div class='mt-2'><b>Answer:</b><br>{html}</div>
    <br><a href='/' class='btn btn-secondary'>Ask another question</a>
    """
    return {"messages": [html], "user_input": state["user_input"]}

# ---- LangGraph Definition ----
graph_builder = StateGraph(State)
graph_builder.add_node("qa_rag", ask_question_with_rag)
graph_builder.add_node("render_html", render_result_as_html)

graph_builder.set_entry_point("qa_rag")
graph_builder.add_edge("qa_rag", "render_html")
graph_builder.set_finish_point("render_html")

graph = graph_builder.compile()

# ---- Routes ----
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head>
        <title>PDF RAG Chat</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <style>
            .chat-container {
                height: 80vh;
                overflow-y: auto;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .message {
                margin-bottom: 20px;
                display: flex;
                flex-direction: column;
            }
            .message.user {
                align-items: flex-end;
            }
            .message.bot {
                align-items: flex-start;
            }
            .message-content {
                max-width: 80%;
                padding: 10px 15px;
                border-radius: 15px;
                margin: 5px 0;
            }
            .user .message-content {
                background: #007bff;
                color: white;
            }
            .bot .message-content {
                background: white;
                border: 1px solid #dee2e6;
            }
            .input-container {
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                width: 90%;
                max-width: 800px;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            }
            .typing-indicator {
                display: none;
                padding: 10px;
                color: #6c757d;
            }
        </style>
    </head>
    <body class="bg-light">
        <div class="container py-5">
            <h2 class="mb-4 text-center">PDF RAG Chat</h2>
            
            <div class="chat-container" id="chatContainer">
                <div class="message bot">
                    <div class="message-content">
                        Hello! I'm your PDF assistant. Ask me anything about your documents.
                    </div>
                </div>
            </div>

            <div class="input-container">
                <form id="chatForm" class="d-flex gap-2">
                    <textarea class="form-control" id="question" rows="1" placeholder="Type your message..." required></textarea>
                    <button type="submit" class="btn btn-primary">Send</button>
                </form>
                <div class="typing-indicator" id="typingIndicator">
                    Assistant is typing...
                </div>
            </div>
        </div>

        <script>
            const chatContainer = document.getElementById('chatContainer');
            const chatForm = document.getElementById('chatForm');
            const questionInput = document.getElementById('question');
            const typingIndicator = document.getElementById('typingIndicator');

            function addMessage(content, isUser = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
                messageDiv.innerHTML = `
                    <div class="message-content">
                        ${content}
                    </div>
                `;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            chatForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const question = questionInput.value.trim();
                if (!question) return;

                // Add user message
                addMessage(question, true);
                questionInput.value = '';

                // Show typing indicator
                typingIndicator.style.display = 'block';

                try {
                    const response = await fetch('/ask/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: `question=${encodeURIComponent(question)}`
                    });

                    const html = await response.text();
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    const answer = doc.querySelector('.mt-2').innerHTML;

                    // Hide typing indicator
                    typingIndicator.style.display = 'none';

                    // Add bot message
                    addMessage(answer);
                } catch (error) {
                    console.error('Error:', error);
                    typingIndicator.style.display = 'none';
                    addMessage('Sorry, there was an error processing your request.');
                }
            });

            // Auto-resize textarea
            questionInput.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = (this.scrollHeight) + 'px';
            });
        </script>
    </body>
    </html>
    """

@app.post("/upload/", response_class=HTMLResponse)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(PDF_REPORTS_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Rebuild vector store with new PDF
        global vector_db
        vector_db = build_vectorstore()
        
        return HTMLResponse(content=f"""
            <html>
            <head>
                <title>PDF RAG QA - Upload</title>
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
            </head>
            <body class="bg-light">
                <div class="container py-5">
                    <h2 class="mb-4">PDF Upload Successful</h2>
                    <p>File "{filename}" has been uploaded and processed.</p>
                    <a href="/" class="btn btn-primary">Back to Home</a>
                </div>
            </body>
            </html>
        """)
    except Exception as e:
        logging.exception("Failed to upload PDF")
        return HTMLResponse(content=f"<h2>Error:</h2><pre>{str(e)}</pre>")

@app.post("/ask/", response_class=HTMLResponse)
async def ask_question(question: str = Form(...)):
    try:
        state = {"messages": [], "user_input": question}
        result = graph.invoke(state)
        return HTMLResponse(content=result["messages"][-1])
    except Exception as e:
        logging.exception("Failed to process RAG")
        return HTMLResponse(content=f"<div class='text-danger'>Error: {str(e)}</div>")

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