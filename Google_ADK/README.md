# Google ADK File Analysis Agent

An agentic AI assistant built with **Google ADK**, **LangChain**, and **LangGraph** that can analyze files, explore directories, search Wikipedia, and run deep multi-step document analysis workflows — all powered by **Gemini 2.5 Flash**.

---

## Project Screenshots :

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Google_ADK/static/a1.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Google_ADK/static/a2.JPG)

## Features

- **File analysis** — Read and extract text from `.docx`, `.pdf`, `.xlsx`, `.xls`, `.csv`, and plain text files
- **File upload support** — Drag-and-drop `.docx` / `.xlsx` files directly in the web UI (handled via `before_model_callback` to bypass Gemini's MIME restrictions)
- **Directory exploration** — List files, browse folder structures, get metadata
- **Text search** — Recursively search for terms across files with glob filtering
- **Wikipedia search** — LangChain tool for factual lookups (no API key needed)
- **LangGraph document workflow** — A structured 3-step analysis pipeline:

```
read_file → analyze_document_workflow
                    │
                    ▼
             [ summarize ]
                    │
                    ▼
        [ extract_key_points ]
                    │
                    ▼
        [ analyze_sentiment ]
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | [Google ADK](https://google.github.io/adk-docs/) |
| LLM | Gemini 2.5 Flash (via Google GenAI API) |
| LLM framework | [LangChain](https://python.langchain.com/) (`langchain-google-genai`) |
| Workflow engine | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| File parsing | `python-docx`, `pypdf`, `openpyxl` |

---

## Project Structure

```
google-adk/
├── my_agent/
│   ├── agent.py        # Agent definition, tools, LangGraph workflow
│   ├── __init__.py
│   └── .env            # API keys (not committed)
└── README.md
```

---

## Prerequisites

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com/) API key

---

## Installation

**1. Clone the repo and create a virtual environment:**

```bash
git clone <your-repo-url>
cd google-adk
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**2. Install dependencies:**

```bash
pip install google-adk \
            python-docx pypdf openpyxl \
            langchain langchain-community langchain-google-genai \
            langgraph wikipedia
```

---

## Configuration

Create `my_agent/.env` with your Google API key:

```env
GOOGLE_GENAI_USE_VERTEXAI=0
GOOGLE_API_KEY=your_google_api_key_here
```

Get your key at [Google AI Studio](https://aistudio.google.com/apikey).

---

## Running the Agent

**Web UI (recommended):**

```bash
adk web
```

Then open `http://localhost:8000` in your browser, select `my_agent`, and start chatting.

**Terminal:**

```bash
adk run my_agent
```

---

## Available Tools

| Tool | Description |
|---|---|
| `read_file` | Read any file — `.docx`, `.pdf`, `.xlsx`, `.csv`, `.txt`, `.py`, etc. |
| `list_directory` | List files and subdirectories in a folder |
| `get_file_info` | Get metadata: size, type, created/modified timestamps |
| `search_in_files` | Recursively search for text across files (supports glob patterns) |
| `analyze_document_workflow` | Run the LangGraph 3-step analysis: summary + key points + sentiment |
| `wikipedia_search` | Search Wikipedia for factual information |

---

## Example Prompts

```
"Read and summarize the file at C:\Users\me\Documents\report.docx"

"List all files in D:\my_project and tell me which are Python files"

"Search for the word 'TODO' in all .py files under D:\my_project"

"Analyze this document: first read it, then run the full workflow on it"

"What is LangGraph according to Wikipedia?"

"What's the largest file in D:\Downloads?"
```

---

## How It Works

### File Upload (drag & drop in web UI)

When a file is uploaded in the `adk web` interface, ADK sends it as a raw binary blob to Gemini. Since Gemini doesn't support `.docx` or `.xlsx` MIME types natively, a `before_model_callback` intercepts the request, extracts the text using the appropriate parser, and replaces the blob with plain text before Gemini ever sees it.

### LangGraph Workflow

The `analyze_document_workflow` tool runs a compiled `StateGraph` with three sequential LangChain LLM calls:

1. **Summarize** — condenses the content into 2–3 sentences
2. **Extract key points** — produces a numbered list of main ideas
3. **Analyze sentiment** — identifies the overall tone and explains why

Each node uses `ChatGoogleGenerativeAI` (Gemini 2.5 Flash via LangChain) and passes its result forward through shared state.

### LangChain Tool

`WikipediaQueryRun` is wrapped with `LangchainTool` — ADK's bridge that lets any LangChain-compatible tool be called natively by the agent.

---

## License

MIT
