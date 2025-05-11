# RAG Challenge: PDF Document QA with LangChain, LangGraph, and Gemini

Welcome to my submission for the **LangChain RAG Challenge**! This project demonstrates a modular, production-ready Retrieval-Augmented Generation (RAG) system designed to ingest and analyze multiple PDF documents. It utilizes **LangChain**, **LangGraph**, **LangSmith**, **FAISS**, **Google Gemini**, and a microservices architecture built with **FastAPI**.

## 🚀 Project Overview

This system allows users to:

* Upload multiple PDF files
* Automatically generate summaries
* Ask context-aware questions about the uploaded content

All of this is orchestrated via a clean, scalable microservices backend.

## 🧱 Architecture

The project is divided into 4 core services:

1. **Upload Service**

   * Ingests multiple PDF files
   * Uses LangChain document loaders to parse and split content

2. **Analyze Service**

   * Summarizes documents using **Google Gemini**
   * Tracks process with **LangSmith**

3. **Ask Service**

   * Embeds content with LangChain
   * Performs semantic search using **FAISS**
   * Answers questions with **LangGraph** + Gemini

4. **Orchestrator Service**

   * Manages flow between services
   * Provides RESTful APIs using **FastAPI**


## Project Screenshots :

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/RAG_Challenge/static/r1.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/RAG_Challenge/static/r2.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/RAG_Challenge/static/r3.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/RAG_Challenge/static/r4.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/RAG_Challenge/static/r5.JPG)


## 📂 Folder Structure

```
rag-challenge/
├── venv/
├── pdf_report/
│   ├── 0a61a353b1ea9fd9b8f63b60239634ca3007d58f.pdf
│   ├── 0c0faea14d108e1617f2d6d2a7c1aae04eb88fe0.pdf
│   └── and other PDF files
├── static/
├── main.py
```

## 🔧 Tech Stack

* [LangChain](https://github.com/langchain-ai/langchain)
* [LangGraph](https://github.com/langchain-ai/langgraph)
* [LangSmith](https://smith.langchain.com/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [FastAPI](https://fastapi.tiangolo.com/)
* [Google Gemini API](https://ai.google.dev/)
* [Docker Compose](https://docs.docker.com/compose/)

## 💡 Features

* ✅ Modular microservices design
* ✅ Handles multiple PDFs
* ✅ RAG pipeline with observability and monitoring
* ✅ Easy to scale and extend

## 📦 How to Run

1. Clone the repo:

```bash
git clone https://github.com/yourusername/rag-challenge.git
cd rag-challenge
```

2. Create `.env` files for API keys in each service (if needed).

3. Start all services:

```bash
python main.py
```

4. Access the orchestrator endpoints via `http://localhost:9000`.

---

⭐ If you like this project, give it a star!

#LangChain #LangGraph #LangSmith #RAG #FAISS #FastAPI #Gemini #Microservices #LLM #AI #PDFprocessing
