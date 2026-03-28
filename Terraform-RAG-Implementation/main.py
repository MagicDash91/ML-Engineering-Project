"""
Simple RAG — FastAPI + LangChain + Ollama qwen3.5:cloud
Frontend served from the same file (HTML/CSS/JS/Bootstrap).

Usage:
    python main.py
    open http://localhost:8000
"""

import os
import uuid
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import uvicorn

# ── LLM & Embeddings ──────────────────────────────────────────────────────────
llm        = ChatOllama(model="qwen3.5:cloud")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# In-memory store: session_id → FAISS retriever
_retrievers: dict[str, object] = {}

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Simple RAG")


# ── Frontend ──────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Simple RAG</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
  <style>
    body { background: #f0f2f5; }
    .chat-box {
      height: 420px;
      overflow-y: auto;
      background: #fff;
      border: 1px solid #dee2e6;
      border-radius: 0.5rem;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .bubble {
      max-width: 80%;
      padding: 0.6rem 1rem;
      border-radius: 1rem;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .bubble.user   { background: #0d6efd; color: #fff; align-self: flex-end; border-bottom-right-radius: 0.2rem; }
    .bubble.bot    { background: #e9ecef; color: #212529; align-self: flex-start; border-bottom-left-radius: 0.2rem; white-space: normal; }
    .bubble.bot h1,.bubble.bot h2,.bubble.bot h3,.bubble.bot h4 { font-size: 1rem; font-weight: 600; margin: 0.5rem 0 0.25rem; }
    .bubble.bot ul,.bubble.bot ol { padding-left: 1.25rem; margin: 0.25rem 0; }
    .bubble.bot p { margin: 0.25rem 0; }
    .bubble.bot code { background: #d0d0d0; padding: 0.1rem 0.3rem; border-radius: 0.25rem; font-size: 0.85rem; }
    .bubble.bot pre { background: #d0d0d0; padding: 0.5rem; border-radius: 0.4rem; overflow-x: auto; }
    .bubble.bot strong { font-weight: 600; }
    .bubble.system { background: #fff3cd; color: #664d03; align-self: center; font-size: 0.85rem; border-radius: 0.5rem; }
    #file-status { font-size: 0.85rem; }
    .spinner-border { width: 1rem; height: 1rem; border-width: 0.15em; }
  </style>
</head>
<body>
<div class="container py-5" style="max-width:720px">
  <h3 class="mb-1 fw-bold">Simple RAG</h3>
  <p class="text-muted mb-4">Upload a file, then ask questions about it.</p>

  <!-- Upload -->
  <div class="card mb-4 shadow-sm">
    <div class="card-body">
      <label class="form-label fw-semibold">Upload document</label>
      <div class="input-group">
        <input type="file" id="file-input" class="form-control" accept=".pdf,.txt" />
        <button class="btn btn-primary" id="upload-btn" onclick="uploadFile()">Upload</button>
      </div>
      <div id="file-status" class="mt-2 text-muted">No file uploaded yet.</div>
    </div>
  </div>

  <!-- Chat -->
  <div class="card shadow-sm">
    <div class="card-body">
      <div class="chat-box mb-3" id="chat-box">
        <div class="bubble system">Upload a document to get started.</div>
      </div>
      <div class="input-group">
        <input type="text" id="question" class="form-control" placeholder="Ask a question…"
               onkeydown="if(event.key==='Enter') sendQuestion()" />
        <button class="btn btn-primary" id="ask-btn" onclick="sendQuestion()">Ask</button>
      </div>
    </div>
  </div>
</div>

<script>
  let sessionId = null;

  function addBubble(text, role) {
    const box = document.getElementById('chat-box');
    const div = document.createElement('div');
    div.className = 'bubble ' + role;
    setBubble(div, text, role);
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
    return div;
  }

  function setBubble(div, text, role) {
    if (role === 'bot') {
      div.innerHTML = marked.parse(text);
    } else {
      div.textContent = text;
    }
    const box = document.getElementById('chat-box');
    if (box) box.scrollTop = box.scrollHeight;
  }

  async function uploadFile() {
    const input = document.getElementById('file-input');
    const status = document.getElementById('file-status');
    if (!input.files.length) { status.textContent = 'Please select a file first.'; return; }

    const btn = document.getElementById('upload-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border me-1"></span>Uploading…';
    status.textContent = 'Processing…';

    const form = new FormData();
    form.append('file', input.files[0]);

    try {
      const res  = await fetch('/upload', { method: 'POST', body: form });
      const data = await res.json();
      if (data.session_id) {
        sessionId = data.session_id;
        status.innerHTML = '<span class="text-success fw-semibold">&#10003; ' + data.filename + ' uploaded (' + data.chunks + ' chunks)</span>';
        const box = document.getElementById('chat-box');
        box.innerHTML = '';
        addBubble('Document ready! Ask me anything about it.', 'system');
      } else {
        status.innerHTML = '<span class="text-danger">Upload failed: ' + (data.error || 'Unknown error') + '</span>';
      }
    } catch(e) {
      status.innerHTML = '<span class="text-danger">Upload error: ' + e.message + '</span>';
    }

    btn.disabled = false;
    btn.textContent = 'Upload';
  }

  async function sendQuestion() {
    const input = document.getElementById('question');
    const q = input.value.trim();
    if (!q) return;
    if (!sessionId) { addBubble('Please upload a document first.', 'system'); return; }

    input.value = '';
    addBubble(q, 'user');

    const btn     = document.getElementById('ask-btn');
    btn.disabled  = true;
    const loading = addBubble('Thinking…', 'bot');

    const form = new FormData();
    form.append('session_id', sessionId);
    form.append('question', q);

    try {
      const res  = await fetch('/ask', { method: 'POST', body: form });
      const data = await res.json();
      setBubble(loading, data.answer || data.error || 'No response.', 'bot');
    } catch(e) {
      setBubble(loading, 'Error: ' + e.message, 'bot');
    }

    btn.disabled = false;
  }
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


# ── Upload endpoint ───────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in (".pdf", ".txt"):
            return JSONResponse(status_code=400, content={"error": "Only PDF and TXT files are supported."})

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        shutil.copyfileobj(file.file, tmp)
        tmp.close()

        # Load document
        if suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            docs = PyPDFLoader(tmp.name).load()
        else:
            from langchain_community.document_loaders import TextLoader
            docs = TextLoader(tmp.name, encoding="utf-8").load()

        os.unlink(tmp.name)

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks   = splitter.split_documents(docs)

        if not chunks:
            return JSONResponse(status_code=400, content={"error": "Could not extract text from the file."})

        # Build FAISS vectorstore
        vectorstore = FAISS.from_documents(chunks, embeddings)
        session_id  = str(uuid.uuid4())
        _retrievers[session_id] = vectorstore.as_retriever(search_kwargs={"k": 4})

        return {"session_id": session_id, "filename": file.filename, "chunks": len(chunks)}

    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ── Ask endpoint ──────────────────────────────────────────────────────────────
@app.post("/ask")
async def ask(session_id: str = Form(...), question: str = Form(...)):
    retriever = _retrievers.get(session_id)
    if not retriever:
        return JSONResponse(status_code=404, content={"error": "Session not found. Please re-upload your document."})

    try:
        docs    = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        system = (
            "You are a helpful assistant that answers questions based strictly on the provided document context.\n\n"
            "Rules:\n"
            "- Answer only from the context below — do not make things up.\n"
            "- If the answer is not in the context, say: 'I could not find that information in the document.'\n"
            "- Be concise and clear.\n\n"
            f"CONTEXT:\n{context}"
        )

        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=question)])
        return {"answer": response.content}

    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
