# 🧠 All of my Machine Learning Engineering Portfolio

Welcome to my GitHub portfolio! I'm a Machine Learning and Artificial Intelligence Engineer with experience building full-stack ML systems — from database integration and data pipelines to large language model applications and cloud deployment. This repo is a collection of my projects combining **ML, NLP, LLMs, and data engineering** to deliver intelligent, scalable solutions.

---

## 🚀 Featured Projects

### 💬 Chat With Your Database
**Tech Stack**: FastAPI, LangChain, SQL, PostgreSQL/MySQL/BigQuery, Docker

A FastAPI-based web app that lets users query databases using natural language.
- Converts user input into SQL queries via **LangChain SQL Agent and Google Gemini**
- Supports dynamic database backends (PostgreSQL, MySQL, BigQuery)
- Real-time query execution and result rendering
- Easily deployable using Docker

---

### 🔁 Automatic Churn Data Analysis & Prediction
**Tech Stack**: FastAPI, DynamoDB, Scikit-learn, LangChain, Docker, GCP

A churn analysis web app that connects to a **DynamoDB** table, enabling:
- Natural language-based churn summary and question-answering using **LangChain and Google Gemini**
- Churn prediction and feature importance via **Scikit-learn**
- Built with **FastAPI** for a smooth, API-ready interface
- Designed for deployment on **Google Cloud Run** or Docker

---

### 🔁 Resume Matcher with Google Gemini and FastAPI

A Resume Matcher with Google Gemini and FastAPI, enabling:
- Upload **PDF** or **DOCX** resumes and a **Job Description**.
- **Google Gemini** analyzes the resumes and compares them with the job description.
- Results are displayed with a **match score** and a breakdown of relevant skills, experience, and education.
- Uses **Markdown** for structured, clean output, which is then rendered as HTML.
- User-friendly **FastAPI** backend with **Bootstrap** frontend.

---

### 🔁 AgenticAI AI-Powered Indonesian Legal Document Analysis

This project is a FastAPI web application that leverages advanced AI technologies, including Google Gemini (Gemini 2.0), LangChain, and LangGraph, to analyze Indonesian legal documents. Users can upload multiple legal files (PDF, Word, Excel, CSV, etc.), ask detailed questions about the content, and receive accurate, structured answers powered by a custom language model prompt focused on Indonesian law.
- Upload multiple legal documents in various formats.
- Extract and combine content from uploaded files using LangChain's UnstructuredFileLoader.
- Use a custom language model prompt template specialized for interpreting Indonesian legal texts.
- Intelligent question classification to query Wikipedia, academic papers (arXiv), and current events (Tavily) tools if relevant.
- A multi-step LangGraph workflow that classifies questions, searches relevant sources, and summarizes results.
- User-friendly web interface built with FastAPI and Bootstrap.
- Supports detailed and structured responses, focusing on legal analysis, comparisons, summaries, and numeric data interpretation.
- Designed specifically to understand and analyze Indonesian laws and regulations.

---

### 🧠 Adaptive RAG System
An intelligent document analysis system that uses Retrieval-Augmented Generation (RAG) with self-reflection capabilities, built with FastAPI, LangChain, and Google's Gemini AI.
Features :
- PDF Document Processing: Upload and analyze PDF documents
- Intelligent Retrieval: Vector-based document search using FAISS
- Self-Reflection: Automatic quality checking and answer validation
- Web Search Fallback: Integrates web search when documents lack information
- Adaptive Iterations: Rewrites questions and retries when needed
- Hallucination Detection: Validates answers against source documents
- Relevance Grading: Grades retrieved documents for relevance
- Process Transparency: Shows workflow steps and decision points

---

### 🧠 Gemini Mixture of Experts Document Analyzer
An advanced AI-powered document analysis system that uses Mixture of Experts (MoE) with Google Gemini to provide comprehensive insights from multiple file types. The system features specialized AI experts that analyze documents from different perspectives and synthesizes their findings into actionable insights.
Features :
- Core Functionality
- Multi-file Upload: Support for PDF, DOC, DOCX, PPT, PPTX, XLS, XLSX, CSV, TXT files
- Mixture of Experts: 5 specialized AI personas for comprehensive analysis
- Interactive Chat: RAG-powered Q&A with your documents
- Session Management: Multi-user support with isolated sessions
- Real-time Processing: Asynchronous document processing and analysis

---

### 🧠 Medical Claims Anti-Fraud Detection System
A dual-pipeline AI system combining LLMs with RAG (Retrieval-Augmented Generation) and Computer Vision + OCR to analyze and detect potential fraud in medical claim submissions. This project focuses on detecting fraudulent activities in medical claims by using both textual and image-based approaches. It is divided into two primary pipelines:
- LLM + RAG Pipeline (main.py): Analyze text documents (e.g. policy PDFs, claim descriptions) to determine fraud risk level (low-risk vs. high-risk).
- Computer Vision + OCR Pipeline (main2.py): Extract and analyze data from uploaded receipts and detect potential tampering or inconsistencies.

---

### 📈 Sales Demand Forecasting App (SARIMA-powered)
This project provides a web-based interactive platform for forecasting sales demand using SARIMA time series models. Built with FastAPI, Jinja2, and Chart.js, it offers intuitive visual diagnostics, forecast outputs, and performance metrics from uploaded CSV/Excel datasets.
Features :
- Upload sales datasets in CSV or Excel format
- Select date and target value columns
- Automatic data cleaning & frequency inference
- Hyperparameter tuning for SARIMA with AIC optimization
- Diagnostics: ACF, PACF, and seasonal decomposition
- Forecasts: 30-day forecast with upper & lower confidence bounds
- Metrics: MAE, RMSE, and MAPE evaluation
- Visuals powered by Chart.js and Matplotlib
- Export forecast results as CSV

---

### 🧠 Fraud Prediction Web Application
This project implements a Fraud Detection system that predicts fraudulent transactions using a Random Forest machine learning model. It provides an interactive web interface built with FastAPI and Bootstrap, allowing users to select a date range, view transaction predictions, and visualize total transaction amounts over time. Additionally, the project integrates advanced LangChain components with Google Generative AI for enhanced fraud data analysis and natural language insights.
Features :
- Upload or fetch transaction data from PostgreSQL database
- Filter transactions by date range
- Predict fraud on transactions using a pre-trained Random Forest model (random_forest_model.pkl)
- Display fraud prediction results in a responsive HTML table
- Visualize total transaction amount over selected dates using Seaborn line charts
- Interactive UI built with FastAPI and Bootstrap for easy usability
- Rich fraud analysis and insights powered by LangChain with Google Generative AI and LangSmith
- Support for PDF, CSV document loading and embedding with FAISS vector stores for deep content analysis

--- 

### ☁️ AWS Cloud Infrastructure with CAG System
A comprehensive AWS cloud infrastructure project demonstrating enterprise-level networking and deployment practices. This project showcases the implementation of a custom AWS Virtual Private Cloud (VPC) with properly segmented public and private subnets, AWS Internet Gateway for public connectivity, and AWS NAT Gateway for secure outbound access from private resources. The infrastructure hosts a Context Augmented Generation (CAG) system powered by Google Gemini AI, deployed on an AWS EC2 instance running Ubuntu Server. The project emphasizes AWS best practices including proper subnet isolation, custom AWS route table configurations, AWS security group management, and scalable AWS network architecture design. This implementation demonstrates proficiency in AWS networking fundamentals, AWS resource provisioning, and deploying production-ready applications on AWS cloud infrastructure.
Tech Stack :
- Cloud: AWS (VPC, EC2, NAT Gateway, Internet Gateway)
- Backend: Python, FastAPI, Uvicorn
- AI: Google Gemini API
- OS: Ubuntu 24.04 LTS

--- 

### 🔍 Passport OCR Fine-Tuning with DeepSeek Vision Model
A specialized OCR (Optical Character Recognition) system fine-tuned on DeepSeek-OCR vision model to accurately extract and transcribe text from Canadian passport documents. This project focuses on improving baseline OCR performance through parameter-efficient fine-tuning using LoRA adapters, reducing Character Error Rate (CER) from 23% to optimized levels for passport data extraction. The system is divided into two primary components:

- Vision Model Fine-Tuning Pipeline: Utilizes Unsloth's FastVisionModel with DeepSeek-OCR as the base model, implementing LoRA (Low-Rank Adaptation) adapters to efficiently train only 1% of model parameters while maintaining high accuracy.
- Custom Dataset Processing Pipeline: Handles hierarchical folder structures (CAN/HK888152/L1/B1/A1/D1.jpg) with automated image path construction from CSV metadata, enabling efficient batch processing and lazy loading for memory optimization.

#### Project Objectives :
- Fine-tune DeepSeek-OCR vision model for Canadian passport text extraction
- Reduce Character Error Rate (CER) from 23% baseline to production-ready accuracy
- Implement parameter-efficient training using LoRA adapters
- Create scalable pipeline for processing passport datasets with complex folder hierarchies
- Enable accurate digitization of passport information for automated document processing

---

## 🛠️ Skill Set

### 🗃️ Databases
- MySQL, PostgreSQL, Google BigQuery, Snowflake, DynamoDB

### 📊 ML & Python Libraries
- Scikit-learn, TensorFlow, PyTorch, Pandas, NumPy, Matplotlib, Seaborn, Scipy

### 📚 NLP & Transformers
- NLTK, HuggingFace Transformers, Wordcloud, N-grams, Sentiment Analysis, Topic Modeling

### 🧠 LLM & RAG Apps
- LangChain, LlamaIndex, OpenAI, Google Gemini, Langgraph, Langsmith
- LLM Finetuning : LoRA, QLoRA, Deepseek OCR (Unsloth)
- Vector DBs: FAISS, ChromaDB

### ☁️ Cloud & Deployment
- Google Cloud (Cloud Run, Cloud SQL)
- AWS (DynamoDB, EC2, VPC, Lambda, ECR, Cloud Watch)
- Docker

### ⚡ Data Pipelines & Big Data
- PySpark, Prefect, Airflow

### 📈 Visualization & BI
- Tableau, Power BI

### 🌐 Web Frameworks
- Flask, FastAPI

### 🌐 Backend
- Celery, Redis

## 📬 Let’s Connect!

- 💼 [LinkedIn](https://www.linkedin.com/in/michael-wiryaseputra/)
- ✉️ Email: michwirja@gmail.com
If you find these projects interesting, feel free to ⭐ star the repo or reach out for collaboration opportunities!

---

