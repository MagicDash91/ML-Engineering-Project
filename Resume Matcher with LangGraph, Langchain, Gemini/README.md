# 🧠 Resume Matcher with LangGraph & Gemini AI

This project showcases how to use **LangGraph**, a powerful library for building stateful LLM workflows, to analyze and match resumes against a job description using **Google's Gemini AI**.

It’s built with **FastAPI** for the web interface and leverages **LangChain** tools for document parsing. The goal is to provide an intuitive and modular example of LangGraph in action.

---

## 💡 Why LangGraph?

LangGraph allows you to define workflows as graphs where nodes represent AI or data processing steps. In this app, LangGraph is used to:

- Orchestrate the document analysis pipeline
- Handle multi-step LLM interactions
- Ensure stateful transitions across resume uploads, LLM calls, and score rendering

This makes it easy to **modularize**, **extend**, and **visualize** AI workflows.

---

## 🧩 Workflow Overview

```text
[Start]
  |
  v
[Load Resumes] ---> [Job Description Input]
  |                        |
  v                        v
[Gemini Evaluation Node] --→ [Score + Feedback Node]
                               |
                               v
                           [Render HTML]
```

Built entirely with **LangGraph**!

---

## 🚀 Features

- Upload multiple resumes (PDF, DOCX)
- Input a job description
- Analyze resumes using Gemini AI
- Score each resume and generate detailed feedback
- Results rendered as HTML with FastAPI

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🕸️ LangGraph | Core workflow engine |
| 🧱 LangChain | File parsing & tools |
| 🌐 FastAPI | Web interface |
| 🧠 Gemini | LLM-based resume analysis |
| 🧾 Markdown | Format output |
| 🎨 Bootstrap | UI styling |

---

## 📁 Project Structure

```
.
├── main.py             # FastAPI app with LangGraph workflow
├── templates/          # HTML templates (optional)
├── static/             # Uploaded files + result storage
└── README.md
```

---

## 🧪 How It Works

1. User uploads resumes and a job description.
2. Files are loaded and parsed using LangChain's `UnstructuredFileLoader`.
3. LangGraph takes over:
   - Node 1: Load documents
   - Node 2: Use Gemini to score resumes
   - Node 3: Format feedback in HTML
4. Results are displayed on the browser.

---

## ⚙️ Requirements

```bash
pip install fastapi uvicorn langgraph langchain google-generativeai \
            langchain-google-genai unstructured markdown
```

---

## 🔐 Setup Your API Key

Set your Gemini API key using an environment variable:

```bash
export GOOGLE_API_KEY=your_api_key
```

---

## ▶️ Run the App

```bash
uvicorn main:app --reload --port 9000
```

Visit: [http://localhost:9000](http://localhost:9000)

---

## 📦 TODO & Ideas

- [ ] Add resume comparison dashboard
- [ ] Graph visualization of LangGraph flow
- [ ] Database logging of job-candidate match history
- [ ] Docker + cloud deployment
- [ ] Role-based access (admin/HR)

---

## 📜 License

MIT License © 2025 Michael

---

## 🙌 Contributions

Learning LangGraph too? Fork the repo and build with me!
