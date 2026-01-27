# Super Agentic RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system with two implementations: **Traditional RAG with RAGAS Evaluation** and **Super Agentic RAG** with intelligent routing and multi-source retrieval.

## Overview

This project provides two powerful RAG implementations:

### 1. Traditional RAG (`main2.py`)
A straightforward RAG system focused on simplicity and quality evaluation using RAGAS metrics.

**Key Features:**
- Simple retrieve-and-generate pipeline
- RAGAS evaluation metrics (Faithfulness, Answer Relevancy, Context Precision, Answer Correctness)
- FAISS vector store for semantic search
- HuggingFace embeddings
- Google Gemini 2.5 Flash for generation
- Web-based interface

### 2. Super Agentic RAG (`main.py`)
An advanced RAG system with intelligent routing, hybrid search, and multi-source retrieval.

**Key Features:**
- **Agentic workflow** with LangGraph for complex multi-step reasoning
- **Hybrid search** (Dense FAISS + Sparse BM25) for better retrieval accuracy
- **Multi-source retrieval** (Documents + Wikipedia + arXiv + Web Search)
- **Intelligent file type routing** (documents vs tabular data)
- **Data visualization** with NVIDIA LLAMA for code generation
- **Adaptive search strategy** that improves with iterations
- **Query expansion** with synonym mapping
- Support for multiple file types (PDF, Word, PowerPoint, CSV, Excel, TXT)

## Architecture Comparison

| Feature | Traditional RAG (`main2.py`) | Super Agentic RAG (`main.py`) |
|---------|------------------------------|-------------------------------|
| **Search Method** | Dense (FAISS only) | Hybrid (FAISS + BM25) |
| **Workflow** | Linear (Retrieve → Generate) | Agentic (Multi-step with LangGraph) |
| **Data Sources** | Single document only | Multi-source (Docs + Wikipedia + arXiv + Web) |
| **Evaluation** | RAGAS metrics built-in | Optional external evaluation |
| **Chunk Size** | 500 characters | 800 characters |
| **Chunk Overlap** | 100 characters | 150 characters |
| **File Types** | PDF, Word, TXT, PowerPoint, CSV | Same + Excel with intelligent routing |
| **Tabular Data** | Basic text search | Advanced analysis with code generation |
| **LLM Models** | Gemini 2.5 Flash only | Gemini 2.5 Flash + NVIDIA LLAMA 3.3 |
| **Use Case** | Q&A with quality metrics | Complex research, data analysis, visualization |

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Langsmith-main/Super_Agentic_RAG
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
```txt
# Core
langchain
langchain-google-genai
langchain-huggingface
langchain-community
langgraph
faiss-cpu

# Document Processing
pypdf
docx2txt
unstructured
python-pptx

# Data Analysis
pandas
numpy
matplotlib
seaborn
rank-bm25
openpyxl

# Web Framework
fastapi
uvicorn
python-multipart

# External APIs
openai
tavily-python

# Evaluation (for main2.py)
ragas
datasets

# Utilities
python-dotenv
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Required for both systems
GOOGLE_API_KEY=your_google_api_key_here

# Optional for LangSmith tracing
LANGCHAIN_API_KEY=your_langchain_api_key_here

# Required only for Super Agentic RAG (main.py)
NVIDIA_API_KEY=your_nvidia_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

#### How to Get API Keys

**Google API Key (Required for both systems):**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy and paste into `.env`

**NVIDIA API Key (Optional - for main.py only):**
1. Go to [NVIDIA API Catalog](https://build.nvidia.com/)
2. Sign up and get API key
3. Used for LLAMA 3.3 model and code generation

**Tavily API Key (Optional - for main.py only):**
1. Go to [Tavily](https://tavily.com/)
2. Sign up for an account
3. Used for web search capabilities

**LangChain API Key (Optional):**
1. Go to [LangSmith](https://smith.langchain.com/)
2. Create account and get API key
3. Used for tracing and debugging

## Usage

### Running Traditional RAG with RAGAS Evaluation

```bash
python main2.py
```

Then open your browser to: `http://localhost:8001`

**Features:**
- Upload documents (PDF, DOCX, TXT, PPTX, CSV)
- Ask questions about the content
- Get evaluated answers with RAGAS metrics:
  - **Faithfulness**: How grounded the answer is in the context (0-100%)
  - **Answer Relevancy**: How well the answer addresses the question (0-100%)
  - **Context Precision**: How relevant the retrieved documents are (0-100%)
  - **Answer Correctness**: Accuracy compared to ground truth (0-100%, optional)

**Tips for Best Results:**
- Use chunk size 1000-1500 for longer documents
- Retrieve k=6-8 chunks for complex questions
- Provide ground truth as the complete expected answer (not the question!)
- Re-upload documents after changing chunk size

### Running Super Agentic RAG

```bash
python main.py
```

Then open your browser to: `http://localhost:8000`

**Features:**
- **Document Analysis**: Upload PDF, Word, PowerPoint, or text files
- **Data Analysis**: Upload CSV/Excel for intelligent data processing
- **Automatic Routing**: System automatically chooses the best strategy
- **Multi-Source Search**: Combines document content with Wikipedia, arXiv, and web search
- **Data Visualization**: Automatically generates plots for data analysis questions
- **Adaptive Retrieval**: Improves search with each iteration

**Example Use Cases:**

**1. Resume/CV Analysis:**
```
Upload: resume.pdf
Question: "What is Michael's teaching experience?"
Result: Agentic RAG with hybrid search retrieves relevant sections
```

**2. Data Analysis with Visualization:**
```
Upload: sales_data.csv
Question: "Create a bar chart showing sales by region"
Result: NVIDIA LLAMA generates Python code → Creates visualization
```

**3. Research Questions:**
```
Upload: research_paper.pdf
Question: "What are the latest developments in transformer models?"
Result: Combines document content + arXiv papers + web search
```

## API Endpoints

### Traditional RAG (`main2.py`) - Port 8001

#### Upload Document
```http
POST /api/upload
Content-Type: multipart/form-data

{
  "file": <file_data>
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "file_name": "document.pdf",
  "file_type": "pdf",
  "message": "File uploaded successfully. Processed 45 documents into 150 chunks",
  "num_chunks": 150
}
```

#### Query Document
```http
POST /api/query
Content-Type: application/json

{
  "session_id": "uuid",
  "query": "What is Michael's experience?",
  "k": 8,
  "evaluate": true,
  "ground_truth": "Michael has 5 years of experience..."
}
```

**Response:**
```json
{
  "answer": "Michael's experience includes...",
  "num_contexts": 8,
  "evaluation": {
    "faithfulness": 1.0,
    "answer_relevancy": 0.835,
    "context_precision": 0.667,
    "answer_correctness": 0.991
  },
  "error": false
}
```

#### Health Check
```http
GET /api/health
```

### Super Agentic RAG (`main.py`) - Port 8000

#### Upload File
```http
POST /api/upload
Content-Type: multipart/form-data

{
  "file": <file_data>
}
```

#### Query
```http
POST /api/query
Content-Type: application/json

{
  "session_id": "uuid",
  "query": "Your question here"
}
```

**Response (Document Analysis):**
```json
{
  "answer": "Detailed answer...",
  "file_type": "pdf",
  "strategy": "agentic_rag_hybrid_search",
  "tools_used": ["PDF_Documents", "Wikipedia"],
  "iterations": 2,
  "error": false
}
```

**Response (Data Visualization):**
```json
{
  "answer": "The chart shows...",
  "plot_url": "/static/plot_uuid.png",
  "code": "python code used",
  "data_type": "tabular",
  "strategy": "nvidia_llama_visualization",
  "visualization_created": true,
  "error": false
}
```

## Configuration

### Traditional RAG (`main2.py`)

**Adjust chunk size and overlap:**
```python
# Line 776
rag = TraditionalRAG(chunk_size=1000, chunk_overlap=200)
```

**Adjust number of retrieved documents:**
```python
# Line 813 - In query_with_evaluation
result = rag.query_with_evaluation(
    question=request.query,
    k=request.k,  # Default: 4, Recommended: 6-8
    ground_truth=request.ground_truth
)
```

### Super Agentic RAG (`main.py`)

**Adjust chunk size and overlap:**
```python
# Line 145-148
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
)
```

**Adjust hybrid search weights:**
```python
# Line 144
retriever = HybridRetriever(dense_weight=0.5, sparse_weight=0.5)
```

**Adjust rate limiting:**
```python
# Line 77
GEMINI_CALL_DELAY = 13  # seconds between API calls
```

## Technical Details

### Hybrid Search Algorithm (main.py)

The hybrid retriever combines two search methods:

1. **Dense Search (FAISS)**: Semantic similarity using embeddings
2. **Sparse Search (BM25)**: Keyword-based search with term frequency

**Score Calculation:**
```
hybrid_score = (dense_weight × dense_similarity) + (sparse_weight × sparse_similarity)
```

**Query Expansion:**
- Automatically expands queries with synonyms
- Example: "teaching" → ["teach", "teacher", "trainer", "tutor", "instructor"]

### Agentic Workflow (main.py)

```
┌─────────────────┐
│ Tool Selection  │  ← Analyze question, choose tools
└────────┬────────┘
         ↓
┌─────────────────┐
│ Hybrid Retrieve │  ← Dense + Sparse search
└────────┬────────┘
         ↓
┌─────────────────┐
│ Multi Retrieve  │  ← Wikipedia, arXiv, Web
└────────┬────────┘
         ↓
┌─────────────────┐
│ Grade Documents │  ← Evaluate relevance
└────────┬────────┘
         ↓
   ┌─────┴─────┐
   │  Relevant?│
   └─────┬─────┘
    Yes  │  No
         ↓     ↓
┌────────┴──┐  ┌──────────┐
│ Generate  │  │  Adapt   │ ← Modify strategy
└─────┬─────┘  └────┬─────┘
      ↓              ↓
┌─────────────┐      │
│  Evaluate   │      │
└──────┬──────┘      │
       │             │
       └─────────────┘
              ↓
         (Loop up to 3 times)
```

### RAGAS Evaluation Metrics (main2.py)

**1. Faithfulness (0-100%)**
- Measures if the answer is grounded in the retrieved context
- High score = No hallucinations
- Formula: Claims supported by context / Total claims

**2. Answer Relevancy (0-100%)**
- Measures if the answer addresses the question
- High score = Answer is focused and on-topic
- Uses cosine similarity between question and answer embeddings

**3. Context Precision (0-100%)**
- Measures if retrieved documents are relevant
- High score = Retrieved the right chunks
- Formula: Relevant chunks / Total retrieved chunks

**4. Answer Correctness (0-100%, requires ground truth)**
- Measures accuracy compared to expected answer
- High score = Answer matches ground truth
- Uses semantic similarity + factual overlap

## Troubleshooting

### RAGAS Showing 0% Context Precision

**Problem:** Context Precision is 0% even though the answer looks good.

**Solutions:**
1. **Check ground truth format**: Should be the complete expected answer, NOT the question
   - ❌ Wrong: "List Michael's experience"
   - ✅ Right: "Michael has experience at ZICY.COM as AI Engineer..."

2. **Re-upload document**: After changing chunk_size, must re-upload the file

3. **Try different embedding model**: Change in line 79-81 of main2.py

4. **Increase chunk size**: Try 1000-1500 instead of 500

### API Rate Limiting Errors

**Problem:** "429 Too Many Requests" from Google Gemini

**Solution:**
- Increase `GEMINI_CALL_DELAY` (line 95 in main2.py or line 77 in main.py)
- Free tier allows 5 requests/minute
- Recommended delay: 13-15 seconds

### File Upload Errors

**Problem:** "Unsupported file type"

**Solution:**
- Ensure file has correct extension (.pdf, .docx, .txt, .pptx, .csv, .xlsx)
- Check file is not corrupted
- For PowerPoint, may need to install additional dependencies:
  ```bash
  pip install python-pptx unstructured[local-inference]
  ```

### Visualization Not Working (main.py)

**Problem:** No plot generated for data analysis

**Solution:**
1. Ensure NVIDIA_API_KEY is set in `.env`
2. Check query contains visualization keywords ("plot", "chart", "graph", "visualize")
3. Verify CSV/Excel file is properly formatted
4. Check `static/` folder has write permissions

## Performance Tips

### For Better Retrieval

1. **Optimal chunk size**: 800-1500 characters for most documents
2. **Increase k**: Retrieve 6-8 chunks instead of default 4
3. **Use hybrid search**: Super Agentic RAG (main.py) has better retrieval
4. **Ground truth format**: Must be the full expected answer

### For Faster Responses

1. **Use Traditional RAG** (main2.py) for simple Q&A
2. **Disable evaluation**: Set `evaluate: false` in query
3. **Reduce k**: Retrieve fewer chunks
4. **Cache documents**: Keep sessions alive to avoid re-processing

### For Better Answers

1. **Provide context**: Ask specific questions
2. **Use ground truth**: Helps RAGAS calculate Answer Correctness
3. **Iterate queries**: Super Agentic RAG adapts with iterations
4. **Multi-source**: Enable Wikipedia/arXiv for research questions

## Project Structure

```
Super_Agentic_RAG/
├── main.py                 # Super Agentic RAG system
├── main2.py                # Traditional RAG with RAGAS
├── .env                    # API keys (create this)
├── README.md               # This file
├── uploaded_files/         # Uploaded documents (auto-created)
├── static/                 # Generated plots (auto-created)
└── requirements.txt        # Python dependencies
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

## Acknowledgments

- **LangChain** for RAG framework
- **LangGraph** for agentic workflows
- **RAGAS** for evaluation metrics
- **Google Gemini** for LLM capabilities
- **NVIDIA LLAMA** for code generation
- **HuggingFace** for embeddings
