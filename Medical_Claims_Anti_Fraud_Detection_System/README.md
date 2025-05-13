
# 🧠 Medical Claims Anti-Fraud Detection System

A dual-pipeline AI system combining **LLMs with RAG (Retrieval-Augmented Generation)** and **Computer Vision + OCR** to analyze and detect potential fraud in medical claim submissions.

---

## 🚀 Project Overview

This project focuses on **detecting fraudulent activities in medical claims** by using both textual and image-based approaches. It is divided into two primary pipelines:

1. **LLM + RAG Pipeline (`main.py`)**: Analyze text documents (e.g. policy PDFs, claim descriptions) to determine fraud risk level (low-risk vs. high-risk).
2. **Computer Vision + OCR Pipeline (`main2.py`)**: Extract and analyze data from uploaded receipts and detect potential tampering or inconsistencies.

---

## 🔍 Core AI Engineering Tasks

### ✅ LLM & RAG Integration for Anti-Fraud (`main.py`)

**Goal:** Use Large Language Models to classify claims as `Low Risk` or `High Risk`.

- ✅ Fine-tune or prompt LLMs (via Google Generative AI) to recognize fraud-related patterns.
- ✅ Implement **Retrieval-Augmented Generation (RAG)** to enrich the model context with claim-specific documents.
- ✅ Use vector-based search with **FAISS** to retrieve relevant chunks from PDFs.
- ✅ Prompt engineering to refine fraud detection accuracy across diverse scenarios.

📦 Tech Stack:
- `LangChain`, `FAISS`, `LangGraph`, `Google Generative AI`
- `FastAPI` for API exposure
- `UnstructuredPDFLoader` & `RecursiveCharacterTextSplitter` for preprocessing

### 🛠️ Prompt Engineering Strategy

- Craft targeted prompts for various fraud scenarios (e.g. duplicate claims, excessive charges, mismatched items).
- Evaluate LLM outputs, optimize prompts iteratively.

---

## 📸 Computer Vision & OCR Work (`main2.py`)

**Goal:** Extract and analyze data from medical receipt images to detect inconsistencies and fraud.

### 🔍 Document Image Processing

- ✅ Use OCR (`PaddleOCR`) to extract structured information from receipts (e.g., name, date, amount).
- ✅ Apply image preprocessing with OpenCV (`denoising`, `grayscale`, `thresholding`) to improve OCR accuracy.
- ✅ Pattern matching for extracting name, date, total amount, email, and address.

### 🧠 Fraud Detection via Vision

- 🚩 Detect tampered documents (e.g., suspicious totals, missing fields).
- 🧾 Generate and process dummy medical receipts for testing.

📦 Tech Stack:
- `PaddleOCR`, `OpenCV`, `NumPy`, `FastAPI`

---

## 💡 Key Features

- 🧠 **Hybrid LLM + CV System**: Combine textual and image-based fraud detection.
- 🔍 **Similarity Search**: Identify similar historical claims using FAISS.
- 📤 **FastAPI UI**: Upload receipts, view extracted structured fields.
- 📊 **Low Risk / High Risk**: Output fraud risk classification based on LLM reasoning.

---

## 📁 Folder Structure

```
├── main.py                # LLM + RAG pipeline
├── main2.py               # OCR and CV pipeline
├── static/ 
├── .env                   # Environment variables (e.g., Gemini API key)
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_generative_ai_key
```

### 3. Start the FastAPI App

For LLM + RAG pipeline:

```bash
uvicorn main:app --reload --port 9000
```

For OCR pipeline:

```bash
uvicorn main2:app --reload --port 9000
```

---

## 📌 Example Use Cases

- 🧾 **Detect duplicate medical reimbursements**
- 📷 **Check for altered totals on receipts**
- 📚 **Validate claim consistency against policy documents**
- 🧠 **Assist fraud analysts in triaging risky claims**

---

## 🛡️ Future Enhancements

- [ ] Integrate tampering detection using deep learning models.
- [ ] Expand support for multilingual OCR (e.g., Indonesian).
- [ ] Store outputs in a vector database for historical tracking.
