from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io, base64
import pickle
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from sklearn import preprocessing
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader, UnstructuredCSVLoader
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
# Load environment variables
load_dotenv()

app = FastAPI()

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load API keys
api_key = os.getenv("GOOGLE_API_KEY")
api_key="**********************************"
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key

# Load your model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

import psycopg2

def fetch_data_from_postgres(start_date=None, end_date=None):
    conn = psycopg2.connect(
        dbname="fraud",
        user="postgres",
        password="***************",
        host="localhost",
        port="5432"
    )

    if start_date and end_date:
        query = """
        SELECT * FROM simple_transactions 
        WHERE date BETWEEN %s AND %s;
        """
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
    else:
        query = "SELECT * FROM simple_transactions;"
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def prepare_data_for_prediction(df):
    # Create a copy of the dataframe to avoid modifying the original
    prediction_df = df.copy()
    
    # Store label encoders and original values for each object column
    encoders = {}
    
    # Label encoding for object datatype columns
    for col in prediction_df.select_dtypes(include=['object']).columns:
        # Skip the date column if it's still an object
        if col == 'date':
            continue
            
        # Create and fit a label encoder
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(prediction_df[col].unique())
        
        # Store the encoder and transform the data
        encoders[col] = label_encoder
        prediction_df[col] = label_encoder.transform(prediction_df[col])
        
    return prediction_df, encoders

def generate_summary(df, prompt="Provide a summary of the transaction data"):
    try:
        # Save DataFrame to CSV first
        csv_path = "static/temp_data.csv"
        df.to_csv(csv_path, index=False)

        # Initialize Google Gemini
        logging.info("Starting Analysis with LLM")
        genai.configure(api_key=api_key)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=api_key,
            temperature=0
        )

        # Load CSV file using UnstructuredCSVLoader
        loader = UnstructuredCSVLoader(csv_path, mode="elements")
        docs = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000000, chunk_overlap=5000
        )
        chunks = text_splitter.split_documents(docs)

        # Process each chunk
        all_summaries = []
        for idx, chunk in enumerate(chunks):
            logging.info(f"Processing chunk {idx + 1}/{len(chunks)}")
            
            template = """
            Based on the following dataset context:
            {text}

            As an expert Data Analyst specializing in fraud detection, analyze the data to deliver a comprehensive and actionable report focusing on fraud-related insights. Structure your analysis around these points:

            1. **Fraud Group Profiles**  
            Describe the main characteristics of fraudulent versus non-fraudulent transactions, highlighting differences in transaction amounts, categories, states, and customer demographics.

            2. **Fraud Patterns and Trends**  
            Identify notable patterns, trends, or anomalies that distinguish fraudulent activities over time or across groups, including seasonal, geographic, or categorical variations.

            3. **Risk Indicators**  
            Highlight key risk factors and predictive indicators associated with fraud, explaining which features most strongly correlate with fraudulent behavior.

            4. **Business Impact and Recommendations**  
            Discuss the potential business impact of fraud on revenue and customer trust, and suggest effective strategies for fraud prevention, detection, and mitigation.

            5. **Targeted Engagement Strategies**  
            Recommend how to segment customers or transactions for targeted interventions, such as tailored monitoring, alerts, or loyalty programs to minimize fraud risk while maintaining positive engagement.

            6. **Summary of Insights**  
            Provide a clear, concise summary that synthesizes the findings and prioritizes actions for fraud risk management.

            Avoid tabular data; present the analysis in a structured and easy-to-understand narrative format.

            {question}
            """
            
            prompt_template = PromptTemplate.from_template(template)
            llm_chain = LLMChain(llm=llm, prompt=prompt_template)
            stuff_chain = StuffDocumentsChain(
                llm_chain=llm_chain, document_variable_name="text"
            )
            
            response = stuff_chain.invoke(
                {
                    "input_documents": [chunk],
                    "question": prompt,
                }
            )
            
            all_summaries.append(response["output_text"])

        # Clean up temporary file
        if os.path.exists(csv_path):
            os.remove(csv_path)

        final_summary = "\n\n".join(all_summaries)
        return final_summary
    except Exception as e:
        logging.error(f"Error in summarization: {str(e)}")
        return f"Error generating summary: {str(e)}"

# Middleware (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML Template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Detection</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
    <style>
        body {
            background-color: #f8f9fa;
        }
        .card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .card-header {
            background-color: #fff;
            border-bottom: 1px solid #eee;
            border-radius: 15px 15px 0 0 !important;
            padding: 20px;
        }
        .form-control {
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #dee2e6;
        }
        .form-control:focus {
            box-shadow: 0 0 0 0.2rem rgba(0,123,255,0.25);
            border-color: #80bdff;
        }
        .btn-primary {
            border-radius: 10px;
            padding: 12px 25px;
            font-weight: 500;
            background-color: #0d6efd;
            border: none;
        }
        .btn-primary:hover {
            background-color: #0b5ed7;
            transform: translateY(-1px);
            transition: all 0.3s ease;
        }
        .table {
            margin-bottom: 0;
        }
        .table th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .chart-container {
            background-color: #fff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .section-title {
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 20px;
        }
        .badge {
            padding: 8px 12px;
            border-radius: 8px;
            font-weight: 500;
        }
        .badge-success {
            background-color: #28a745;
        }
        .badge-danger {
            background-color: #dc3545;
        }
        .table-responsive {
            max-height: 500px;
            overflow-y: auto;
        }
        .table th {
            position: sticky;
            top: 0;
            background-color: #f8f9fa;
            font-weight: 600;
            z-index: 1;
        }
        /* Markdown styling */
        .markdown-body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #24292e;
            padding: 10px;
        }
        .markdown-body h1, 
        .markdown-body h2, 
        .markdown-body h3, 
        .markdown-body h4 {
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
            color: #2c3e50;
        }
        .markdown-body h1 {
            font-size: 2em;
            border-bottom: 1px solid #eaecef;
            padding-bottom: .3em;
        }
        .markdown-body h2 {
            font-size: 1.5em;
            border-bottom: 1px solid #eaecef;
            padding-bottom: .3em;
        }
        .markdown-body h3 {
            font-size: 1.25em;
        }
        .markdown-body h4 {
            font-size: 1em;
        }
        .markdown-body ul, 
        .markdown-body ol {
            padding-left: 2em;
            margin-top: 0;
            margin-bottom: 16px;
        }
        .markdown-body li {
            margin-top: 0.25em;
        }
        .markdown-body p {
            margin-top: 0;
            margin-bottom: 16px;
        }
        .markdown-body blockquote {
            padding: 0 1em;
            color: #6a737d;
            border-left: 0.25em solid #dfe2e5;
            margin: 0 0 16px 0;
        }
        .markdown-body code {
            padding: 0.2em 0.4em;
            margin: 0;
            font-size: 85%;
            background-color: rgba(27, 31, 35, 0.05);
            border-radius: 3px;
            font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
        }
        .markdown-body pre {
            word-wrap: normal;
            padding: 16px;
            overflow: auto;
            font-size: 85%;
            line-height: 1.45;
            background-color: #f6f8fa;
            border-radius: 3px;
            margin-bottom: 16px;
        }
        .markdown-body pre code {
            display: inline;
            max-width: auto;
            padding: 0;
            margin: 0;
            overflow: visible;
            line-height: inherit;
            word-wrap: normal;
            background-color: transparent;
            border: 0;
        }
        .markdown-body strong {
            font-weight: 600;
        }
    </style>
</head>
<body>
<div class="container py-5">
    <div class="row justify-content-center">
        <div class="col-lg-10">
            <div class="card">
                <div class="card-header">
                    <h2 class="section-title mb-0">
                        <i class="fas fa-shield-alt me-2"></i>
                        Fraud Detection Analysis
                    </h2>
                </div>
                <div class="card-body">
                    <form method="post" class="mb-4">
                        <div class="row g-3">
                            <div class="col-md-5">
                                <label for="start_date" class="form-label">
                                    <i class="far fa-calendar-alt me-2"></i>Start Date
                                </label>
                                <input type="date" id="start_date" name="start_date" class="form-control" required>
                            </div>
                            <div class="col-md-5">
                                <label for="end_date" class="form-label">
                                    <i class="far fa-calendar-alt me-2"></i>End Date
                                </label>
                                <input type="date" id="end_date" name="end_date" class="form-control" required>
                            </div>
                            <div class="col-md-2 d-flex align-items-end">
                                <button type="submit" class="btn btn-primary w-100">
                                    <i class="fas fa-search me-2"></i>Analyze
                                </button>
                            </div>
                        </div>
                    </form>

                    {% if table %}
                    <div class="card mb-4">
                        <div class="card-header">
                            <h4 class="section-title mb-0">
                                <i class="fas fa-table me-2"></i>
                                Prediction Results
                            </h4>
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                {{ table | safe }}
                            </div>
                            <div class="mt-3">
                                <a href="/download-excel" class="btn btn-success">
                                    <i class="fas fa-file-excel me-2"></i>Download Excel
                                </a>
                            </div>
                        </div>
                    </div>

                    <div class="card mb-4">
                        <div class="card-header">
                            <h4 class="section-title mb-0">
                                <i class="fas fa-chart-pie me-2"></i>
                                Data Summary
                            </h4>
                        </div>
                        <div class="card-body">
                            <div class="summary-content markdown-body">
                                {{ summary | safe }}
                            </div>
                        </div>
                    </div>

                    <div class="chart-container">
                        <h4 class="section-title">
                            <i class="fas fa-chart-line me-2"></i>
                            Transaction Amount Trend
                        </h4>
                        <img src="/static/transaction_chart.png" class="img-fluid" alt="Transaction Chart"/>
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</div>
</body>
</html>
"""

# Setup templates with string template
templates = Jinja2Templates(directory=".")
template = templates.env.from_string(html_template)

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return template.render(request=request, table=None, chart=None)

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, start_date: str = Form(...), end_date: str = Form(...)):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Fetch the data
    original_df = fetch_data_from_postgres(start_date, end_date)
    
    # Create a copy for prediction and apply label encoding
    prediction_df, encoders = prepare_data_for_prediction(original_df)
    
    # Prepare data for model (drop date column)
    X = prediction_df.drop(columns=['date'])
    
    # Make prediction
    y_pred = model.predict(X)
    
    # Add prediction to the original dataframe (not the encoded one)
    original_df['is_fraud'] = y_pred
    
    # Convert fraud labels to text for display
    original_df['is_fraud'] = original_df['is_fraud'].map({1: 'Fraud', 0: 'Not Fraud'})
    
    # Generate summary
    summary = generate_summary(original_df)
    
    # Save to Excel - use the original dataframe with human-readable values
    excel_path = "static/transaction_data.xlsx"
    original_df.to_excel(excel_path, index=False)
    
    # Create HTML table from the original dataframe (not encoded)
    result_table = original_df.to_html(classes="table table-striped", index=False)
    
    # Create chart
    chart_df = original_df.groupby('date')['amt'].sum().reset_index()
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='date', y='amt', data=chart_df, marker="o", linewidth=2)
    plt.title('Transaction Amount Trend', pad=20, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Amount', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save chart to static folder
    plt.savefig("static/transaction_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return template.render(
        request=request,
        table=result_table,
        summary=markdown.markdown(summary)
    )

@app.get("/download-excel")
async def download_excel():
    return FileResponse(
        "static/transaction_data.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="transaction_data.xlsx"
    )

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