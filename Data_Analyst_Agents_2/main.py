# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import io
import re
import uuid
import base64
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import StuffDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_unstructured import UnstructuredLoader as UnstructuredFileLoader
except ImportError:
    from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
import google.generativeai.types as gtypes
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from fpdf import FPDF
from datetime import datetime
import os
from PIL import Image

# New imports for database functionality
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType

# New imports for Research (Agentic AI) functionality
from langchain.tools import WikipediaQueryRun, ArxivQueryRun, Tool
from langchain.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain.agents.agent_types import AgentType
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
import json
import uuid

# New imports for Google Drive functionality
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pickle

import tempfile

load_dotenv()

# === Configuration ===
api_key = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# FastAPI app
app = FastAPI(title="Data Analysis Agent", description="Powered by NVIDIA Llama-3.1-Nemotron-Ultra-253B-v1")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory storage for sessions (in production, use Redis or database)
sessions = {}

# Pydantic models
class QueryRequest(BaseModel):
    session_id: str
    query: str

class DatabaseQueryRequest(BaseModel):
    database_uri: str
    query: str

class ResearchQueryRequest(BaseModel):
    session_id: str
    query: str

class GoogleDriveLoadRequest(BaseModel):
    session_id: str
    drive_url: str
    recursive: bool = False

class GoogleDriveQueryRequest(BaseModel):
    session_id: str
    query: str

class ResearchState(TypedDict):
    question: str
    original_question: str
    docs: Optional[List[str]]
    external_docs: Optional[List[str]]
    answer: Optional[str]
    relevant: Optional[bool]
    answered: Optional[bool]
    selected_tools: Optional[List[str]]
    search_strategy: Optional[str]
    iteration_count: Optional[int]
    reasoning: Optional[str]
    visualization_needed: Optional[bool]
    chart_data: Optional[str]
    plot_url: Optional[str]

class ChatResponse(BaseModel):
    response: str
    plot_url: Optional[str] = None
    thinking: Optional[str] = None
    code: Optional[str] = None

class DatasetInfo(BaseModel):
    columns: List[str]
    rows: int
    insights: str
    preview: List[Dict[str, Any]]

# === Utility Functions ===

def extract_google_drive_folder_id(url: str) -> str:
    """
    Extract folder ID from Google Drive URL.

    Supports URLs like:
    - https://drive.google.com/drive/folders/1yucgL9WGgWZdM1TOuKkeghlPizuzMYb5
    - https://drive.google.com/drive/u/0/folders/1yucgL9WGgWZdM1TOuKkeghlPizuzMYb5
    - Or just the folder ID itself
    """
    import re

    # Pattern to match folder ID in various URL formats
    patterns = [
        r'/folders/([a-zA-Z0-9_-]+)',  # Standard folder URL
        r'id=([a-zA-Z0-9_-]+)',         # URL with id parameter
        r'^([a-zA-Z0-9_-]{25,})$'       # Direct folder ID (typically 25+ chars)
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # If no pattern matches, assume it's already a folder ID
    return url

def load_google_drive_documents(folder_id: str, credentials_path: str, token_path: str, recursive: bool = False):
    """
    Manually load documents from Google Drive using Google Drive API.

    Returns list of documents with metadata.
    """
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    creds = None

    # Try service account first (if credentials_path is a service account key)
    try:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES
        )
        print("Using service account credentials")
    except Exception as e:
        print(f"Not a service account, trying OAuth: {e}")

        # Check if token already exists
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)

        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)

            # Save credentials for future use
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)

    # Build Drive service
    service = build('drive', 'v3', credentials=creds)

    documents = []

    def get_files_from_folder(folder_id, recursive=False):
        """Recursively get all files from folder"""
        query = f"'{folder_id}' in parents and trashed=false"

        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType)",
            pageSize=100
        ).execute()

        items = results.get('files', [])

        for item in items:
            mime_type = item['mimeType']

            # Handle folders recursively if requested
            if mime_type == 'application/vnd.google-apps.folder' and recursive:
                get_files_from_folder(item['id'], recursive=True)

            # Process supported file types
            elif mime_type in [
                'application/vnd.google-apps.document',  # Google Docs
                'application/vnd.google-apps.spreadsheet',  # Google Sheets
                'application/pdf'  # PDF
            ]:
                try:
                    # Export Google Docs/Sheets as plain text
                    if mime_type == 'application/vnd.google-apps.document':
                        content = service.files().export(
                            fileId=item['id'],
                            mimeType='text/plain'
                        ).execute()
                        text_content = content.decode('utf-8')

                    elif mime_type == 'application/vnd.google-apps.spreadsheet':
                        content = service.files().export(
                            fileId=item['id'],
                            mimeType='text/csv'
                        ).execute()
                        text_content = content.decode('utf-8')

                    elif mime_type == 'application/pdf':
                        # For PDFs, download and extract text (basic)
                        request = service.files().get_media(fileId=item['id'])
                        text_content = f"[PDF File: {item['name']} - Content extraction requires additional processing]"

                    else:
                        text_content = ""

                    if text_content.strip():
                        documents.append({
                            'name': item['name'],
                            'content': text_content,
                            'mime_type': mime_type,
                            'id': item['id']
                        })

                except Exception as e:
                    print(f"Error processing {item['name']}: {e}")
                    continue

    # Start loading files
    get_files_from_folder(folder_id, recursive)

    return documents

# === Core Functions (from original code) ===

def QueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation based on keywords."""
    messages = [
        {"role": "system", "content": "detailed thinking off. You are an assistant that determines if a query is requesting a data visualization. Respond with only 'true' if the query is asking for a plot, chart, graph, or any visual representation of data. Otherwise, respond with 'false'."},
        {"role": "user", "content": query}
    ]
    
    response = client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        messages=messages,
        temperature=0.1,
        max_tokens=5
    )
    
    intent_response = response.choices[0].message.content.strip().lower()
    return intent_response == "true"

def PlotCodeGeneratorTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas+matplotlib code for a plot based on the query and columns."""
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using pandas **and matplotlib** (as plt) to answer:
    "{query}"

    Rules
    -----
    1. Use `pandas` for all data processing. For visualization, you may use either:
    - `matplotlib.pyplot as plt`
    - or `seaborn as sns` (recommended for enhanced aesthetics and ease of use).

    2. Perform data cleansing before any plotting or analysis:
    2.1. Drop unrelated identifier columns (e.g., 'id', 'user_id', or any unique identifiers).
    2.2. Handle missing values:
            - For numeric columns: fill missing values with the column mean.
            - For object (categorical) columns: fill missing values with the column mode.
    2.3. Label-encode object-type columns only if they are needed for computation or plotting:
            - Use `pd.factorize()` or `sklearn.preprocessing.LabelEncoder`.

    3. Create exactly one meaningful plot:
    - If using `matplotlib`, set `figsize=(8, 6)` using `plt.subplots()`.
    - If using `seaborn`, you may use its internal figure handling but ensure the plot size is visually appropriate.
    - Add a clear title, and label the x-axis and y-axis.
    - Use colors or grouping variables meaningfully (e.g., `hue`, `c`, `style`, etc.).

    4. If the user asks to use SpaCy:
    - First, ensure the model is available:
        ```python
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        ```

    - Then proceed with SpaCy-based operations (e.g., NER or POS tagging) using the `nlp` object.

    5. If the user asks to visualize the number of cases for each country or region:

    - First, try to import `geopandas`. If not available, attempt to install it. Then load world map data from the Natural Earth CDN.

    - Example:
        ```python
        import pandas as pd
        import matplotlib.pyplot as plt

        try:
            import geopandas as gpd
        except ImportError:
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "geopandas"])
                import geopandas as gpd
            except Exception:
                gpd = None

        if gpd:
            url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
            world = gpd.read_file(url)

            # df_cases = pd.DataFrame({{'country': [...], 'cases': [...]}})
            merged = world.merge(df_cases, how="left", left_on="NAME", right_on="country")
            merged["cases"] = merged["cases"].fillna(0)

            fig, ax = plt.subplots(figsize=(12, 8))
            merged.plot(column="cases", ax=ax, legend=True, cmap="OrRd", edgecolor="black")
            ax.set_title("Number of Cases by Country or Region")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            df_cases.sort_values("cases", ascending=False).plot.bar(x="country", y="cases", ax=ax)
            ax.set_title("Cases by Country or Region")
            ax.set_xlabel("Country")
            ax.set_ylabel("Number of Cases")
            plt.xticks(rotation=90)

        result = fig
        ```

    - Always assign the final output to a variable named `result`.

    6. If the user asks to explain model predictions or mentions "SHAP":

    - Ensure the `shap` library is installed:
        ```python
        try:
            import shap
        except ImportError:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
            import shap
        ```

    - Use SHAP `TreeExplainer` with your trained model:
        ```python
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        ```

    - Then, for **bar plot of global feature importance**:
        ```python
        # Handle both multi-class and binary/regression cases
        if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
            # Multi-class case: use the first class or specify class index
            shap.plots.bar(shap_values[:, :, 0])  # For class 0
        else:
            # Binary classification or regression
            shap.plots.bar(shap_values)
        ```

    - For **waterfall plot of one prediction**:
        ```python
        # For one specific prediction (e.g., first test sample)
        if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
            # Multi-class: show explanation for first sample, first class
            shap.plots.waterfall(shap_values[0, :, 0])
        else:
            # Binary classification or regression
            shap.plots.waterfall(shap_values[0])
        ```

    - Do **not** use `matplotlib.pyplot` for SHAP visualizations.
    - Note: SHAP plots display directly and don't return objects to assign to `result`.



    7. If the user asks for sentiment analysis or mentions "sentiment", "emotion", or "TextBlob":
       - IMPORTANT: Always put installation code FIRST, before any imports:
            ```python
            import subprocess
            import sys
            import importlib
            
            # Install TextBlob if not available
            try:
                import textblob
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob"])
                importlib.invalidate_caches()
                import textblob
            
            # Install and setup NLTK dependencies
            try:
                import nltk
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
                import nltk
            
            # Download required corpora
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/brown')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('brown', quiet=True)
            
            # Now import everything needed
            from textblob import TextBlob
            import pandas as pd
            import matplotlib.pyplot as plt
            ```
       - Then perform sentiment analysis on text columns. Look for columns containing text data (e.g., 'full_text', 'text', 'content', 'message', 'review', etc.):
            ```python
            # Identify text column (common names: full_text, text, content, message, review)
            text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'content', 'message', 'review', 'comment'])]
            text_col = text_columns[0] if text_columns else df.select_dtypes(include=['object']).columns[0]
            
            # Perform sentiment analysis
            text_data = df[text_col].dropna().astype(str)
            sentiments = []
            
            for text in text_data:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Determine sentiment label
                if polarity > 0.1:
                    sentiment_label = 'positive'
                elif polarity < -0.1:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'
                
                sentiments.append({{
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'sentiment_label': sentiment_label
                }})
            
            # Add sentiment scores to original dataframe
            df_sentiment = pd.DataFrame(sentiments)
            result = pd.concat([df.reset_index(drop=True), df_sentiment], axis=1)
            ```
       - NEVER import TextBlob at the top of the file. Always install first, then import.
    
    8. **Dashboard Mode (triggered when the query includes 'dashboard'):**
       - you can use piechart, barchart, boxplot and other charts to build a compact dashboard with multiple charts in a single figure using matplotlib/seaborn.

    9. **Time Series Analysis Mode (triggered when the query includes 'forecast' or 'forecasting'):**
        - Create **4 subplots in a single figure** (`fig, axes = plt.subplots(2, 2, figsize=(14, 10))`):
            a) **Original + 30-Step Forecast** — Overlay forecast in a contrasting color with a legend.
            b) **First Difference** — Compute `target_series.diff()` and plot, removing NaN from the start.
            c) **Moving Average Plot** — Rolling mean (window=7) overlaid on the original.
            d) **Rolling Standard Deviation** — Rolling std (window=7) to assess volatility changes.
        - **Column Selection Logic:**
            - If `"High"` exists (case-insensitive), use it.
            - Otherwise, pick the numeric column with highest variance.
            - If no numeric column exists, raise an error.
        - **Time Column Handling:**
            - Detect date/time column by name (`date`, `time`, `year`, `month`) and convert with `pd.to_datetime()`.
            - Sort by this column before plotting.
        - **Aesthetics:**
            - Add clear titles to each subplot (`Original + Forecast`, `First Difference`, `Moving Average`, `Rolling Std`).
            - Rotate x-axis labels by 30°.
            - Use `sns.set_theme(style="whitegrid")` if available.
            - Limit y-axis in all plots to min/max of the chosen series ± 10% for clarity.
        - **Output:**
            - Always call `plt.tight_layout()`.
            - Assign the final `matplotlib` Figure to `result`.

    10. Assign the final result (whether a DataFrame, Series, scalar value, or plot Figure) to a variable named `result`.

    11. Return only the Python code, wrapped inside a single markdown code block that begins with ```python and ends with ```.
    """

def CodeWritingTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas-only code for a data query (no plotting)."""
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code (pandas **only**, no plotting) to answer:
    "{query}"

    Rules
    -----
    1. Use **only pandas operations** on the DataFrame `df`. Do not use any external libraries like matplotlib or scikit-learn.

    2. Assign the final output (DataFrame, Series, or scalar value) to a variable named `result`.

    3. Perform **data cleansing** before any analysis or transformation:
    3.1. Drop unrelated identifier columns, such as 'id', 'user_id', or other unique IDs not needed for analysis.
    3.2. Handle missing values:
        - For numeric columns: fill missing values using the column's mean.
        - For object (categorical) columns: fill missing values using the column's mode.
    3.3. If any object-type columns are needed in the result, apply label encoding:
        - Use `df[col] = pd.factorize(df[col])[0]` or another suitable pandas-only approach.
    
    4. If the user asks to use SpaCy:
       - First, ensure the model is available:
            ```python
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
            ```
       - Then proceed with SpaCy-based operations (e.g., NER or POS tagging) using the `nlp` object.

    5. If the user asks to visualize the number of cases for each country or region:

    - First, try to import `geopandas`. If not available, attempt to install it. Then load world map data from the Natural Earth CDN.

    - Example:
        ```python
        import pandas as pd
        import matplotlib.pyplot as plt

        try:
            import geopandas as gpd
        except ImportError:
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "geopandas"])
                import geopandas as gpd
            except Exception:
                gpd = None

        if gpd:
            url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
            world = gpd.read_file(url)

            # df_cases = pd.DataFrame({{'country': [...], 'cases': [...]}})
            merged = world.merge(df_cases, how="left", left_on="NAME", right_on="country")
            merged["cases"] = merged["cases"].fillna(0)

            fig, ax = plt.subplots(figsize=(12, 8))
            merged.plot(column="cases", ax=ax, legend=True, cmap="OrRd", edgecolor="black")
            ax.set_title("Number of Cases by Country or Region")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            df_cases.sort_values("cases", ascending=False).plot.bar(x="country", y="cases", ax=ax)
            ax.set_title("Cases by Country or Region")
            ax.set_xlabel("Country")
            ax.set_ylabel("Number of Cases")
            plt.xticks(rotation=90)

        result = fig
        ```

    - Always assign the final output to a variable named `result`.

    6. If the user asks to explain model predictions or mentions "SHAP":

    - Ensure the `shap` library is installed:
        ```python
        try:
            import shap
        except ImportError:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
            import shap
        ```

    - Use SHAP `TreeExplainer` with your trained model:
        ```python
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        ```

    - Then, for **bar plot of global feature importance**:
        ```python
        # Handle both multi-class and binary/regression cases
        if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
            # Multi-class case: use the first class or specify class index
            shap.plots.bar(shap_values[:, :, 0])  # For class 0
        else:
            # Binary classification or regression
            shap.plots.bar(shap_values)
        ```

    - For **waterfall plot of one prediction**:
        ```python
        # For one specific prediction (e.g., first test sample)
        if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
            # Multi-class: show explanation for first sample, first class
            shap.plots.waterfall(shap_values[0, :, 0])
        else:
            # Binary classification or regression
            shap.plots.waterfall(shap_values[0])
        ```

    - Do **not** use `matplotlib.pyplot` for SHAP visualizations.
    - Note: SHAP plots display directly and don't return objects to assign to `result`.


    7. If the user asks for sentiment analysis or mentions "sentiment", "emotion", or "TextBlob":
       - IMPORTANT: Always put installation code FIRST, before any imports:
            ```python
            import subprocess
            import sys
            import importlib
            
            # Install TextBlob if not available
            try:
                import textblob
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob"])
                importlib.invalidate_caches()
                import textblob
            
            # Install and setup NLTK dependencies
            try:
                import nltk
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
                import nltk
            
            # Download required corpora
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/brown')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('brown', quiet=True)
            
            # Now import everything needed
            from textblob import TextBlob
            import pandas as pd
            import matplotlib.pyplot as plt
            ```
       - Then perform sentiment analysis on text columns. Look for columns containing text data (e.g., 'full_text', 'text', 'content', 'message', 'review', etc.):
            ```python
            # Identify text column (common names: full_text, text, content, message, review)
            text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'content', 'message', 'review', 'comment'])]
            text_col = text_columns[0] if text_columns else df.select_dtypes(include=['object']).columns[0]
            
            # Perform sentiment analysis
            text_data = df[text_col].dropna().astype(str)
            sentiments = []
            
            for text in text_data:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Determine sentiment label
                if polarity > 0.1:
                    sentiment_label = 'positive'
                elif polarity < -0.1:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'
                
                sentiments.append({{
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'sentiment_label': sentiment_label
                }})
            
            # Add sentiment scores to original dataframe
            df_sentiment = pd.DataFrame(sentiments)
            result = pd.concat([df.reset_index(drop=True), df_sentiment], axis=1)
            ```
       - NEVER import TextBlob at the top of the file. Always install first, then import.
    
    8. **Dashboard Mode (triggered when the query includes 'dashboard'):**
       - you can use piechart, barchart, boxplot and other charts to build a compact dashboard with multiple charts in a single figure using matplotlib/seaborn.
    
    9. **Time Series Analysis Mode (triggered when the query includes 'forecast' or 'forecasting'):**
        - Create **4 subplots in a single figure** (`fig, axes = plt.subplots(2, 2, figsize=(14, 10))`):
            a) **Original + 30-Step Forecast** — Overlay forecast in a contrasting color with a legend.
            b) **First Difference** — Compute `target_series.diff()` and plot, removing NaN from the start.
            c) **Moving Average Plot** — Rolling mean (window=7) overlaid on the original.
            d) **Rolling Standard Deviation** — Rolling std (window=7) to assess volatility changes.
        - **Column Selection Logic:**
            - If `"High"` exists (case-insensitive), use it.
            - Otherwise, pick the numeric column with highest variance.
            - If no numeric column exists, raise an error.
        - **Time Column Handling:**
            - Detect date/time column by name (`date`, `time`, `year`, `month`) and convert with `pd.to_datetime()`.
            - Sort by this column before plotting.
        - **Aesthetics:**
            - Add clear titles to each subplot (`Original + Forecast`, `First Difference`, `Moving Average`, `Rolling Std`).
            - Rotate x-axis labels by 30°.
            - Use `sns.set_theme(style="whitegrid")` if available.
            - Limit y-axis in all plots to min/max of the chosen series ± 10% for clarity.
        - **Output:**
            - Always call `plt.tight_layout()`.
            - Assign the final `matplotlib` Figure to `result`.

    10. Wrap the entire code snippet inside a single markdown code block that begins with ```python and ends with ```. Do not include any explanation, comments, or output — only valid executable Python code.  
    """

def CodeGenerationAgent(query: str, df: pd.DataFrame):
    """Selects the appropriate code generation tool and gets code from the LLM for the user's query."""
    should_plot = QueryUnderstandingTool(query)
    prompt = PlotCodeGeneratorTool(df.columns.tolist(), query) if should_plot else CodeWritingTool(df.columns.tolist(), query)

    messages = [
        {"role": "system", "content": "detailed thinking off. You are a Python data-analysis expert who writes clean, efficient code. Solve the given problem with optimal pandas operations. Be concise and focused. Your response must contain ONLY a properly-closed ```python code block with no explanations before or after. Ensure your solution is correct, handles edge cases, and follows best practices for data analysis."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        messages=messages,
        temperature=0.2,
        max_tokens=1024
    )

    full_response = response.choices[0].message.content
    code = extract_first_code_block(full_response)
    return code, should_plot, ""

def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    """Executes the generated code in a controlled environment and returns the result or error message."""
    env = {"pd": pd, "df": df}
    if should_plot:
        plt.rcParams["figure.dpi"] = 100
        env["plt"] = plt
        env["sns"] = sns
        env["io"] = io
    try:
        exec(code, {}, env)
        return env.get("result", None)
    except Exception as exc:
        return f"Error executing code: {exc}"

def ReasoningCurator(query: str, result: Any) -> str:
    """Builds and returns the LLM prompt for reasoning about the result."""
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    if is_error:
        desc = result
    elif is_plot:
        title = ""
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        elif isinstance(result, plt.Axes):
            title = result.get_title()
        desc = f"[Plot Object: {title or 'Chart'}]"
    else:
        desc = str(result)[:300]

    if is_plot:
        prompt = f'''
        The user asked: "{query}".
        Below is a description of the plot result:
        {desc}
        Explain in 2–3 concise sentences what the chart shows (no code talk).'''
    else:
        prompt = f'''
        The user asked: "{query}".
        The result value is: {desc}
        Explain in 2–3 concise sentences what this tells about the data (no mention of charts).'''
    return prompt

def ReasoningAgent(query: str, result: Any):
    """Gets the LLM's reasoning about the result (plot or value) and extracts model 'thinking' and final explanation."""
    prompt = ReasoningCurator(query, result)

    response = client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        messages=[
            {"role": "system", "content": "detailed thinking on. You are an insightful data analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1024
    )

    full_response = response.choices[0].message.content
    
    # Extract thinking content
    thinking_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
    thinking_content = thinking_match.group(1).strip() if thinking_match else ""
    
    # Remove thinking tags from final response
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    
    return thinking_content, cleaned

def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Generate a summary prompt string for the LLM based on the DataFrame."""
    prompt = f"""
        You are a data analyst tasked with summarizing and exploring a dataset for an upcoming report.

        ### Dataset Information
        - **Number of Rows:** {len(df)}
        - **Number of Columns:** {len(df.columns)}
        - **Columns:** {', '.join(df.columns)}
        - **Data Types:** {df.dtypes.to_dict()}
        - **Missing Values per Column:** {df.isnull().sum().to_dict()}

        ---

        ### Instructions
        Provide your output in **Markdown** format, clearly structured and easy to read. Your response should include the following sections:

        ---

        ## 🧾 Dataset Overview
        Write a concise paragraph (3–5 sentences) describing:
        - What the dataset likely represents based on the column names
        - Types of variables (e.g., categorical, numerical, datetime)
        - Any notable features or assumptions that can be made from the column names or data types

        ---

        ## 🔍 Key Observations
        Include a bulleted list of 3–5 insights such as:
        - Presence of missing values and potential impact
        - Unusual data types or suspicious columns
        - Any imbalances or data quality concerns
        - Early intuition about the kind of data (e.g., transaction data, customer info, sensor logs)

        ---

        ## ❓ Possible Analysis Questions
        Provide 3–4 bullet points on exploratory data analysis (EDA) or business questions that could be investigated. Examples:
        - What are the key drivers of a target variable (if present)?
        - Are there meaningful clusters or groups within the data?
        - How does a certain feature relate to others (e.g., correlation, group stats)?
        - Are there outliers or anomalies that need addressing?

        ---

        ### Formatting Guidelines
        - Use **bold** or *italic* for emphasis where needed
        - Structure content with appropriate markdown headings (`##`, `###`, `-`)
        - Avoid repeating column names unless relevant to insights

        Keep your writing concise, focused, and written as if it will be read by a stakeholder or teammate.
        """
    return prompt

def DataInsightAgent(df: pd.DataFrame) -> str:
    """Uses the LLM to generate a brief summary and possible questions for the uploaded dataset."""
    prompt = DataFrameSummaryTool(df)
    try:
        response = client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1",
            messages=[
                {"role": "system", "content": "You are a data analyst who generates clear, concise, and insightful markdown-formatted summaries of datasets. Do not show step-by-step reasoning. Use professional tone and focus on key insights only. Structure responses with: ## 🧾 Dataset Overview (describe what the dataset contains, its structure, any unique aspects), ## 📌 Key Observations (highlight missing data, outliers, data types, or quality issues), and ## ❓ Exploratory Questions (suggest 3–4 analysis questions, focusing on relationships, trends, or business relevance). Use markdown formatting elements like headers, bullet points, bold, and inline code where appropriate. Be brief but informative. Avoid technical jargon and unnecessary elaboration unless relevant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"Error generating dataset insights: {exc}"

def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

def save_plot_to_static(fig, session_id: str) -> str:
    """Save matplotlib figure to static folder and return URL."""
    plot_id = str(uuid.uuid4())
    filename = f"plot_{session_id}_{plot_id}.png"
    filepath = os.path.join("static", filename)
    
    fig.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return f"/static/{filename}"

# === Database Chat Functions ===

def DatabaseChatAgent(database_uri: str, query: str) -> str:
    """Handle database queries using LangChain SQL agent."""
    try:
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            google_api_key="AIzaSyCowWFDIENMNDTBtGm5HkvWzeXG9SpIboI"
        )

        # Create database connection
        db = SQLDatabase.from_uri(database_uri)

        # Create SQL toolkit
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        # Initialize agent
        agent_executor = initialize_agent(
            tools=toolkit.get_tools(),
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        # Execute query
        response = agent_executor.run({"input": query})
        return response

    except Exception as e:
        return f"Error connecting to database or executing query: {str(e)}"

# === Automatic Dashboard Generation ===

def AutomaticDashboardAgent(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive, interactive dashboard HTML using NVIDIA LLAMA Nemotron.
    Strategy: AI analyzes data and creates dashboard template, then we inject actual data.
    """

    # Prepare dataset analysis (lightweight - no raw data)
    dataset_analysis = {
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
        "numeric_cols": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_cols": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "datetime_cols": df.select_dtypes(include=['datetime64']).columns.tolist(),
        "sample_preview": df.head(3).to_dict('records'),  # Just 3 rows for structure
        "statistics": df.describe().to_dict() if not df.empty else {},
        "missing_values": df.isnull().sum().to_dict(),
        "unique_counts": {col: df[col].nunique() for col in df.columns}
    }

    # Get value ranges for filters
    numeric_ranges = {}
    for col in dataset_analysis['numeric_cols']:
        numeric_ranges[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }

    # Get unique values for categorical filters (limit to top 20 per column)
    categorical_values = {}
    for col in dataset_analysis['categorical_cols']:
        unique_vals = df[col].value_counts().head(20).index.tolist()
        categorical_values[col] = [str(v) for v in unique_vals]

    dataset_analysis['numeric_ranges'] = numeric_ranges
    dataset_analysis['categorical_values'] = categorical_values

    # Streamlined prompt for dashboard template generation
    prompt = f"""
You are an expert Data Visualization Engineer specializing in creating professional, interactive dashboards similar to Tableau and Power BI.

# DATASET INFORMATION
- **Rows**: {dataset_analysis['shape'][0]:,}
- **Columns**: {dataset_analysis['shape'][1]}
- **Column Names**: {', '.join(dataset_analysis['columns'])}
- **Numeric Columns**: {', '.join(dataset_analysis['numeric_cols']) if dataset_analysis['numeric_cols'] else 'None'}
- **Categorical Columns**: {', '.join(dataset_analysis['categorical_cols']) if dataset_analysis['categorical_cols'] else 'None'}
- **Data Types**: {json.dumps(dataset_analysis['dtypes'], indent=2)}
- **Statistics**: {json.dumps(dataset_analysis['statistics'], indent=2)[:500]}...

# YOUR MISSION
Create a **COMPLETE, SELF-CONTAINED HTML FILE** that displays a professional, interactive dashboard for this dataset.

# CRITICAL REQUIREMENTS

## 1. DASHBOARD STRUCTURE (Tableau/Power BI Style)
```
┌─────────────────────────────────────────────────┐
│ 📊 DASHBOARD TITLE & DESCRIPTION                │
├─────────────────────────────────────────────────┤
│  [KPI 1]   │  [KPI 2]   │  [KPI 3]   │  [KPI 4] │  ← Key Metrics Cards
├─────────────────────────────────────────────────┤
│  🔍 INTERACTIVE FILTERS SECTION                 │  ← Dropdowns, Date pickers, Sliders
├─────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐            │
│  │   Chart 1    │  │   Chart 2    │            │  ← Grid of visualizations
│  │  (Bar/Line)  │  │   (Pie/Donut)│            │
│  └──────────────┘  └──────────────┘            │
│  ┌──────────────┐  ┌──────────────┐            │
│  │   Chart 3    │  │   Chart 4    │            │
│  │  (Scatter)   │  │   (Heatmap)  │            │
│  └──────────────┘  └──────────────┘            │
├─────────────────────────────────────────────────┤
│  📋 DATA TABLE (Paginated, Sortable, Searchable)│
└─────────────────────────────────────────────────┘
```

## 2. MUST-HAVE COMPONENTS

### A. Header Section
- Dashboard title based on dataset content
- Brief description of what the dashboard shows
- Last updated timestamp
- Export/Download buttons (Export as PDF, PNG)

### B. KPI Cards (4-6 cards minimum)
- Display key metrics (totals, averages, counts, percentages)
- Use icons from Font Awesome
- Color-coded (green for positive, red for negative, blue for neutral)
- Animated counters (numbers count up on load)
- Trend indicators (↑ ↓ arrows with percentage change if applicable)

### C. Interactive Filters (Must implement)
- Dropdowns for categorical columns (top 3-5 most relevant)
- Date range picker if datetime columns exist
- Numeric range sliders if numeric columns exist
- "Reset Filters" button
- Filters MUST update all charts and KPIs in real-time

### D. Visualizations (Minimum 4-6 charts)
Choose appropriate chart types based on data:
- **Bar Chart**: For comparing categories
- **Line Chart**: For trends over time
- **Pie/Donut Chart**: For proportions/distributions
- **Scatter Plot**: For correlations between numeric variables
- **Heatmap**: For correlation matrix
- **Area Chart**: For cumulative trends
- **Stacked Bar**: For grouped comparisons

### E. Data Table
- Display all filtered data
- Sortable columns (click header to sort)
- Searchable (search box)
- Paginated (show 10/25/50/100 rows)
- Export to CSV button

## 3. TECHNICAL SPECIFICATIONS

### Libraries to Use:
```html
<!-- Use Chart.js for visualizations (lightweight, beautiful, interactive) -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

<!-- Bootstrap for responsive layout -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

<!-- Font Awesome for icons -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<!-- Optional: DataTables for advanced table features -->
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
```

### Data Handling:
**CRITICAL**: The data will be injected programmatically. Use this exact structure:
```javascript
// Data will be injected here - DO NOT MODIFY THIS LINE
const rawData = __DATA_PLACEHOLDER__;
const allColumns = {json.dumps(dataset_analysis['columns'])};
const numericColumns = {json.dumps(dataset_analysis['numeric_cols'])};
const categoricalColumns = {json.dumps(dataset_analysis['categorical_cols'])};
const numericRanges = {json.dumps(numeric_ranges)};
const categoricalOptions = {json.dumps(categorical_values)};
```

The `rawData` variable will contain all {dataset_analysis['shape'][0]:,} rows of data. Build all your charts, KPIs, filters, and tables to work with this variable.

### Color Scheme (Professional):
```css
:root {{
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --success-color: #11998e;
    --danger-color: #ff416c;
    --warning-color: #ffc107;
    --info-color: #00b4d8;
    --dark-bg: #1a1a2e;
    --card-bg: #f8f9fa;
    --text-primary: #2d3748;
    --text-secondary: #6c757d;
}}
```

## 4. INTERACTIVITY REQUIREMENTS

### Chart Interactions:
- Tooltips on hover showing exact values
- Click on legend to toggle data series
- Responsive to window resize
- Smooth animations on load and update

### Filter System:
```javascript
// When any filter changes:
function applyFilters() {{
    // 1. Filter the raw data based on selected criteria
    // 2. Recalculate KPIs
    // 3. Update all chart data
    // 4. Refresh data table
    // 5. Add smooth transition animations
}}
```

## 5. DESIGN PRINCIPLES

### Visual Hierarchy:
1. KPIs at top (most important metrics first)
2. Filters in prominent position
3. Most important charts in top-left
4. Data table at bottom

### Responsive Design:
- Desktop: 2-column chart grid
- Tablet: 1-column chart grid
- Mobile: Stack everything vertically
- Use Bootstrap grid system (row/col-md-6/col-12)

### Professional Styling:
- Subtle shadows for depth (`box-shadow: 0 4px 6px rgba(0,0,0,0.1)`)
- Rounded corners (`border-radius: 12px`)
- Smooth transitions (`transition: all 0.3s ease`)
- Consistent spacing (`padding: 20px`, `margin-bottom: 20px`)
- Clean typography (system fonts)

## 6. CODE STRUCTURE

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - [Dataset Name]</title>

    <!-- All CDN links here -->

    <style>
        /* All CSS here - make it beautiful! */
    </style>
</head>
<body>
    <!-- Header -->
    <header class="dashboard-header">
        <!-- Title, description, export buttons -->
    </header>

    <!-- KPI Cards -->
    <section class="kpi-section container">
        <div class="row">
            <!-- 4-6 KPI cards -->
        </div>
    </section>

    <!-- Filters -->
    <section class="filters-section container">
        <!-- Interactive filters -->
    </section>

    <!-- Charts -->
    <section class="charts-section container">
        <div class="row">
            <!-- 4-6 canvas elements for charts -->
        </div>
    </section>

    <!-- Data Table -->
    <section class="table-section container">
        <!-- Interactive data table -->
    </section>

    <script>
        // 1. Data variables
        // 2. Calculate KPIs function
        // 3. Create charts function
        // 4. Filter handling function
        // 5. Table initialization
        // 6. Initialize everything on page load
    </script>
</body>
</html>
```

## 7. QUALITY CHECKLIST

Before generating the final HTML, ensure:
- ✅ All data is embedded (no external files needed)
- ✅ Charts are interactive and responsive
- ✅ Filters actually work and update everything
- ✅ KPIs show meaningful metrics
- ✅ Color scheme is professional and consistent
- ✅ Layout works on mobile, tablet, desktop
- ✅ Code is clean, commented, and organized
- ✅ No console errors
- ✅ Smooth animations and transitions
- ✅ Looks as good as Tableau/Power BI

# SAMPLE DATA PREVIEW
Here's a sample to understand the data structure (full data will be injected):
```json
{json.dumps(dataset_analysis['sample_preview'], indent=2)}
```

**IMPORTANT**: The `rawData` variable will contain ALL {dataset_analysis['shape'][0]:,} rows. Design your dashboard to handle this complete dataset dynamically!

# FINAL INSTRUCTION
Generate the COMPLETE HTML file now. Return ONLY the HTML code, no explanations.
Start with `<!DOCTYPE html>` and end with `</html>`.
Make it absolutely stunning, professional, and fully functional!
"""

    # Call NVIDIA LLAMA Nemotron for generation
    try:
        messages = [
            {
                "role": "system",
                "content": "You are an expert data visualization engineer who creates stunning, interactive dashboards. You output ONLY valid HTML code, no explanations, no markdown blocks."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1",
            messages=messages,
            temperature=0.3,
            max_tokens=16000
        )

        dashboard_html = response.choices[0].message.content

        # Clean up response (remove markdown code blocks if present)
        if "```html" in dashboard_html:
            dashboard_html = dashboard_html.split("```html")[1].split("```")[0].strip()
        elif "```" in dashboard_html:
            dashboard_html = dashboard_html.split("```")[1].split("```")[0].strip()

        # INJECT ACTUAL DATA into the template
        # Convert dataframe to JSON (all rows) - handle NaN values
        df_clean = df.fillna('')  # Replace NaN with empty strings for JSON compatibility
        actual_data_json = json.dumps(df_clean.to_dict('records'))

        # Replace the placeholder with actual data
        dashboard_html = dashboard_html.replace('__DATA_PLACEHOLDER__', actual_data_json)

        # Also handle cases where AI might use quotes around placeholder
        dashboard_html = dashboard_html.replace('"__DATA_PLACEHOLDER__"', actual_data_json)
        dashboard_html = dashboard_html.replace("'__DATA_PLACEHOLDER__'", actual_data_json)

        return dashboard_html

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"<html><body><h1>Error generating dashboard</h1><p>{str(e)}</p><pre>{error_detail}</pre></body></html>"

# === Research (Agentic AI) Functions ===

# Initialize research tools
def initialize_research_tools():
    """Initialize external research tools"""
    try:
        # Setup tools
        wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
        
        # Note: You may need to set up Tavily API key
        tavily_api_key = os.getenv("TAVILY_API_KEY", "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")
        tavily_tool_instance = TavilySearchResults(k=3, tavily_api_key=tavily_api_key)
        
        tools = {
            "Wikipedia": Tool(
                name="Wikipedia",
                func=wikipedia_tool.run,
                description="Use for general concepts and historical information"
            ),
            "arXiv": Tool(
                name="arXiv",
                func=arxiv_tool.run,
                description="Use for academic research and scientific studies"
            ),
            "TavilySearch": Tool(
                name="TavilySearch",
                func=tavily_tool_instance.run,
                description="Use for current information, news, and web search"
            ),
        }
        return tools
    except Exception as e:
        print(f"Warning: Could not initialize some research tools: {e}")
        return {}

# Research workflow nodes
def tool_selection_node(state: ResearchState) -> ResearchState:
    """Agent decides which tools to use based on question analysis"""
    question = state["question"]
    
    prompt = f"""
    You are an intelligent research agent that selects the best tools for answering questions.
    
    Question: {question}
    
    Available tools:
    1. Wikipedia - For general concepts and historical information
    2. arXiv - For academic research and scientific studies  
    3. TavilySearch - For current information, news, and web search
    4. DocumentAnalysis - For specific document content (already loaded)
    
    Analysis:
    - Is this about current events or news? -> Use TavilySearch
    - Is this about academic/scientific research? -> Use arXiv
    - Is this about general concepts? -> Use Wikipedia
    - Is this about specific documents? -> Use DocumentAnalysis
    
    Select 1-3 tools that would be most helpful. Return as comma-separated list.
    
    Format:
    TOOLS: tool1,tool2,tool3
    REASONING: why these tools were selected
    """
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
            google_api_key="AIzaSyCowWFDIENMNDTBtGm5HkvWzeXG9SpIboI"
        )
        result = llm.invoke(prompt)
        content = result.content.strip()
        
        # Parse response for tool selection
        lines = content.split('\n')
        selected_tools = []
        reasoning = ""
        
        for line in lines:
            if line.startswith('TOOLS:'):
                selected_tools = [tool.strip() for tool in line.replace('TOOLS:', '').split(',')]
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
        
        # Use NVIDIA's QueryUnderstandingTool for visualization detection (like CSV Data feature)
        visualization_needed = False
        try:
            visualization_needed = QueryUnderstandingTool(question)
        except Exception as viz_error:
            print(f"Visualization classification error: {viz_error}")
        
        return {
            **state,
            "selected_tools": selected_tools,
            "reasoning": reasoning,
            "visualization_needed": visualization_needed,
            "search_strategy": "multi_tool_search"
        }
    except Exception as e:
        return {
            **state,
            "selected_tools": ["DocumentAnalysis"],
            "reasoning": f"Error in tool selection: {e}",
            "visualization_needed": False,
            "search_strategy": "document_only"
        }

def multi_source_retrieve_node(state: ResearchState) -> ResearchState:
    """Retrieve from multiple sources based on selected tools"""
    question = state["question"]
    selected_tools = state.get("selected_tools", [])
    
    # Internal document retrieval (from uploaded documents)
    internal_docs = state.get("docs", [])
    if not internal_docs:
        internal_docs = [f"Document content related to: {question}"]
    
    # External tool retrieval
    external_docs = []
    tools = initialize_research_tools()
    
    for tool_name in selected_tools:
        if tool_name in tools:
            try:
                tool_result = tools[tool_name].run(question)
                external_docs.append(f"{tool_name}: {tool_result}")
            except Exception as e:
                external_docs.append(f"{tool_name}: Error - {str(e)}")
    
    return {
        **state,
        "docs": internal_docs,
        "external_docs": external_docs
    }

def enhanced_grade_node(state: ResearchState) -> ResearchState:
    """Grade relevance of documents from multiple sources"""
    question = state["question"]
    docs = state.get("docs", [])
    external_docs = state.get("external_docs", [])
    
    all_docs = docs + external_docs
    
    # Make grading more lenient to avoid infinite loops
    iteration_count = state.get("iteration_count", 0)
    
    # If we've already tried once, be more lenient
    if iteration_count >= 1 or not all_docs:
        return {**state, "relevant": True}
    
    prompt = f"""
    Evaluate the relevance of retrieved documents:
    
    Question: {question}
    
    Documents:
    {all_docs}
    
    Are these documents sufficient to provide at least a basic answer to the question?
    Be lenient - even partial information should be considered sufficient.
    
    Reply 'yes' if ANY useful information is available, 'no' only if completely irrelevant.
    """
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            google_api_key="AIzaSyCowWFDIENMNDTBtGm5HkvWzeXG9SpIboI"
        )
        result = llm.invoke(prompt)
        is_relevant = "yes" in result.content.lower()
        
        return {**state, "relevant": is_relevant}
    except Exception as e:
        return {**state, "relevant": True}  # Default to relevant to avoid infinite loops

def enhanced_generation_node(state: ResearchState) -> ResearchState:
    """Generate answer using multiple sources"""
    question = state["question"]
    docs = state.get("docs", [])
    external_docs = state.get("external_docs", [])
    selected_tools = state.get("selected_tools", [])
    
    all_docs = docs + external_docs
    context = "\n".join(all_docs)
    
    # Visualization will be handled by separate visualization node
    plot_url = state.get("plot_url")
    chart_data = state.get("chart_data")
    
    prompt = f"""
    You are an expert research assistant. Synthesize information from multiple sources to provide a comprehensive answer.
    
    Question: {question}
    
    Sources used: {', '.join(selected_tools)}
    
    Context from multiple sources:
    {context}
    
    Instructions:
    1. Provide a clear, comprehensive answer
    2. Mention which sources contributed key information
    3. If sources conflict, note the differences
    4. Ensure accuracy and completeness
    5. Structure the response with clear sections
    
    Answer:
    """
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key="AIzaSyCowWFDIENMNDTBtGm5HkvWzeXG9SpIboI"
        )
        response = llm.invoke(prompt)
        return {
            **state, 
            "answer": response.content.strip(),
            "plot_url": plot_url,
            "chart_data": chart_data
        }
    except Exception as e:
        return {
            **state, 
            "answer": f"Error generating response: {e}",
            "plot_url": plot_url,
            "chart_data": chart_data
        }

def answer_check_node(state: ResearchState) -> ResearchState:
    """Check if answer is sufficient"""
    question = state["question"]
    answer = state.get("answer", "")
    iteration_count = state.get("iteration_count", 0)
    
    # Be more lenient if we've tried once already or if we have any answer
    if iteration_count >= 1 or (answer and len(answer.strip()) > 50):
        return {**state, "answered": True}
    
    prompt = f"Does this answer provide at least some useful information about the question?\nQuestion: {question}\nAnswer: {answer}\nReply yes if it provides ANY relevant information, no only if completely useless."
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            google_api_key="AIzaSyCowWFDIENMNDTBtGm5HkvWzeXG9SpIboI"
        )
        result = llm.invoke(prompt)
        answered = "yes" in result.content.lower()
        return {**state, "answered": answered}
    except Exception as e:
        return {**state, "answered": True}  # Default to answered to avoid infinite loops

def strategy_adaptation_node(state: ResearchState) -> ResearchState:
    """Adapt search strategy if answer is insufficient"""
    question = state["question"]
    current_tools = state.get("selected_tools", [])
    iteration_count = state.get("iteration_count", 0)
    
    # Limit iterations to prevent infinite loops - be more aggressive
    if iteration_count >= 1:  # Reduced from 2 to 1
        # Force completion with existing answer or a basic response
        existing_answer = state.get("answer", "")
        if not existing_answer or len(existing_answer.strip()) < 10:
            existing_answer = "Based on the uploaded documents, I was unable to provide a complete analysis after multiple attempts. The documents may not contain sufficient information to fully answer your question."
        
        return {
            **state,
            "answered": True,  # Force completion
            "relevant": True,  # Force relevance
            "answer": existing_answer
        }
    
    prompt = f"""
    The current answer was insufficient. Adapt the search strategy:
    
    Original question: {question}
    Previously used tools: {current_tools}
    Iteration: {iteration_count}
    
    Suggest:
    1. Different tools to try
    2. Modified search query
    3. Alternative approach
    
    Format:
    NEW_TOOLS: tool1,tool2
    NEW_QUERY: modified query
    """
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.4,
            google_api_key="AIzaSyCowWFDIENMNDTBtGm5HkvWzeXG9SpIboI"
        )
        result = llm.invoke(prompt)
        content = result.content.strip()
        
        # Parse adaptation
        new_tools = current_tools  # default
        new_query = question       # default
        
        for line in content.split('\n'):
            if line.startswith('NEW_TOOLS:'):
                new_tools = [tool.strip() for tool in line.replace('NEW_TOOLS:', '').split(',')]
            elif line.startswith('NEW_QUERY:'):
                new_query = line.replace('NEW_QUERY:', '').strip()
        
        return {
            **state,
            "selected_tools": new_tools,
            "question": new_query,
            "iteration_count": iteration_count + 1
        }
    except Exception as e:
        return {
            **state,
            "answered": True,  # Force completion on error
            "iteration_count": iteration_count + 1
        }

def visualization_node(state: ResearchState) -> ResearchState:
    """Generate visualization using LangChain Python Agent (like reference langchain_python_agent_for_ds.py)"""
    question = state.get("question", "")
    docs = state.get("docs", [])
    external_docs = state.get("external_docs", [])
    visualization_needed = state.get("visualization_needed", False)
    
    plot_url = None
    chart_data = None
    
    print(f"Visualization node: visualization_needed = {visualization_needed}")
    print(f"Visualization node: question = {question}")
    print(f"Visualization node: docs count = {len(docs) if docs else 0}")
    print(f"Visualization node: external_docs count = {len(external_docs) if external_docs else 0}")
    
    if visualization_needed:
        try:
            # Prepare research data for visualization
            all_docs = docs + external_docs
            context = "\n".join(all_docs) if all_docs else ""
            
            print(f"Context length: {len(context)}")
            
            if not context or len(context) < 50:
                print("Insufficient context for visualization")
                chart_data = "Insufficient context for visualization"
                return {**state, "plot_url": plot_url, "chart_data": chart_data}
            
            # Smart visualization detection - analyze what the user actually wants
            import pandas as pd
            import re
            
            # Detect if user wants specific data visualization vs general text analysis
            question_lower = question.lower()
            performance_keywords = ['performance', 'accuracy', 'precision', 'recall', 'f1', 'score', 'comparison', 'compare', 'model', 'algorithm', 'result']
            chart_keywords = ['chart', 'plot', 'graph', 'visualization', 'bar', 'line', 'scatter', 'histogram']
            model_keywords = ['adaboost', 'decision tree', 'random forest', 'svm', 'neural network', 'logistic regression', 'naive bayes']
            
            is_performance_request = any(keyword in question_lower for keyword in performance_keywords)
            is_chart_request = any(keyword in question_lower for keyword in chart_keywords)
            is_model_request = any(keyword in question_lower for keyword in model_keywords)
            
            print(f"Visualization analysis: performance={is_performance_request}, chart={is_chart_request}, model={is_model_request}")
            
            # Create LangChain Python Agent (using reference approach)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.3,
                google_api_key="AIzaSyCowWFDIENMNDTBtGm5HkvWzeXG9SpIboI"
            )
            
            # Create Python agent with manual prompt to avoid parsing errors
            from langchain.agents import create_react_agent, AgentExecutor
            from langchain.prompts import PromptTemplate
            
            prompt = PromptTemplate.from_template("""
You are an expert Python programmer. You must write and execute Python code to create visualizations.

CRITICAL: You MUST follow this exact format for every response:

Question: the input question you must answer
Thought: you should always think about what to do
Action: Python_REPL
Action Input: <put your python code here - NO markdown, NO backticks, just plain python code>
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT RULES:
- NEVER use markdown code blocks (```python or ```tool_code)
- Put code directly after "Action Input:" with no formatting
- You have access to: matplotlib.pyplot, pandas, numpy, uuid, os, re
- Save plots to "static/research_plot_<uuid>.png"
- Print the final file path

You have access to the following tools:
{tools}

To use a tool, use the following format:
Action: the action to take, should be one of [{tool_names}]

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
            
            tools = [PythonREPLTool()]
            agent = create_react_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,
                return_intermediate_steps=True
            )
            
            # Save research context to text file for the Python agent
            temp_text_path = os.path.join("static", "research_context.txt")
            os.makedirs("static", exist_ok=True)
            with open(temp_text_path, 'w', encoding='utf-8') as f:
                f.write(f"USER QUESTION: {question}\n\n")
                f.write("RESEARCH CONTEXT:\n")
                f.write(context)
            print(f"Saved research context to {temp_text_path}")
            
            # Intelligent visualization prompt based on request type
            if is_performance_request and (is_model_request or is_chart_request):
                # User wants specific model performance visualization
                viz_prompt = f"""
I have research data about '{question}' in a text file at '{temp_text_path}'.

The user is asking for: "{question}"

Please:
1. Read the text file to understand the research context
2. Extract any performance metrics, model results, or numerical data mentioned in the text
3. Create an appropriate visualization based on what the user requested
4. If the text contains model performance data (accuracy, precision, recall, etc.), create a comparison chart
5. If no specific metrics are found, create a chart showing key findings from the research
6. Use matplotlib with appropriate figure size
7. Add proper title and axis labels based on the user's request
8. Use good colors and styling
9. Save the plot as PNG to 'static/research_plot_{{unique_id}}.png' where unique_id is a random 8-character string
10. Print the saved file path

Make sure to import all required libraries (pandas, matplotlib.pyplot, numpy, uuid, os).
Be creative and intelligent about extracting and visualizing the data based on the user's specific request.
Return the saved file path at the end.
"""
            else:
                # Fall back to word frequency analysis for general text analysis requests
                from collections import Counter
                
                words = context.lower().split()
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those', 'using', 'used', 'use', 'study', 'research', 'data', 'results', 'model', 'models', 'method', 'methods', 'from', 'also', 'can', 'may', 'one', 'two', 'three', 'new', 'more', 'time', 'first', 'last', 'other', 'each', 'many', 'some', 'most', 'all'}
                
                meaningful_words = []
                for word in words:
                    clean_word = re.sub(r'[^a-zA-Z]', '', word)
                    if len(clean_word) > 3 and clean_word.lower() not in stop_words:
                        meaningful_words.append(clean_word.lower())
                
                word_freq = Counter(meaningful_words)
                top_words = word_freq.most_common(15)
                
                if top_words and len(top_words) >= 3:
                    research_df = pd.DataFrame(top_words, columns=['Term', 'Frequency'])
                    temp_csv_path = os.path.join("static", "research_temp_data.csv")
                    research_df.to_csv(temp_csv_path, index=False)
                    
                    viz_prompt = f"""
I have research data about '{question}' in a CSV file at '{temp_csv_path}'.

Please:
1. Load the CSV file using pandas: pd.read_csv('{temp_csv_path}')
2. Create a horizontal bar chart showing the top 10 most frequent terms
3. Use matplotlib with figure size (10, 6)
4. Add proper title: 'Top Terms from Research: {question[:50]}...'
5. Add axis labels: x-axis='Frequency', y-axis='Terms'
6. Use color='skyblue' for bars
7. Add grid with alpha=0.3
8. Save the plot as PNG to 'static/research_plot_{{unique_id}}.png' where unique_id is a random 8-character string
9. Use plt.tight_layout() before saving
10. Print the saved file path

Make sure to import all required libraries (pandas, matplotlib.pyplot, uuid, os).
Return the saved file path at the end.
"""
                else:
                    print("Insufficient data for any visualization")
                    chart_data = "Insufficient data for visualization"
                    return {**state, "plot_url": plot_url, "chart_data": chart_data}
            
            print("Executing visualization with LangChain Python Agent...")
            
            # Run the agent (this will execute Python code automatically)
            agent_result = agent_executor.run({"input": viz_prompt})
            print(f"Agent result: {agent_result}")
            
            # Extract the saved file path from agent result
            import re
            file_pattern = r"static/research_plot_[a-zA-Z0-9]{8}\.png"
            file_match = re.search(file_pattern, str(agent_result))
            
            if file_match:
                saved_file = file_match.group(0)
                plot_url = f"/{saved_file}"
                print(f"Research visualization created successfully using LangChain Python Agent: {plot_url}")
                chart_data = {
                    "visualization_type": "research_analysis",
                    "generated_by": "langchain_python_agent",
                    "data_source": "intelligent_detection" if is_performance_request else "research_terms"
                }
            else:
                # Fallback: look for any PNG files created in static directory
                import glob
                png_files = glob.glob("static/research_plot_*.png")
                if png_files:
                    # Get the most recent file
                    latest_file = max(png_files, key=os.path.getctime)
                    plot_url = f"/{latest_file}"
                    print(f"Found visualization file: {plot_url}")
                    chart_data = {
                        "visualization_type": "research_analysis",
                        "generated_by": "langchain_python_agent_fallback",
                        "data_source": "intelligent_detection" if is_performance_request else "research_terms"
                    }
                else:
                    print("No plot file was generated by agent")
                    chart_data = "No plot file was generated by Python agent"
            
            # Clean up temporary files
            try:
                if os.path.exists(temp_text_path):
                    os.remove(temp_text_path)
                    print(f"Cleaned up temporary file: {temp_text_path}")
                
                temp_csv_path = os.path.join("static", "research_temp_data.csv")
                if os.path.exists(temp_csv_path):
                    os.remove(temp_csv_path)
                    print(f"Cleaned up temporary file: {temp_csv_path}")
            except Exception as cleanup_error:
                print(f"Error cleaning up temp files: {cleanup_error}")
                
            else:
                print(f"Insufficient meaningful terms found: {len(top_words) if top_words else 0}")
                chart_data = f"Insufficient meaningful terms found for visualization (found {len(top_words) if top_words else 0} terms)"
                
        except Exception as e:
            print(f"Research visualization error: {e}")
            import traceback
            traceback.print_exc()
            chart_data = f"Research visualization error: {e}"
    else:
        print("Visualization not needed for this query")
        chart_data = "Visualization not requested"
    
    return {**state, "plot_url": plot_url, "chart_data": chart_data}

# Build research workflow
def build_research_workflow():
    """Build the agentic research workflow using LangGraph"""
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("ToolSelection", tool_selection_node)
    workflow.add_node("MultiRetrieve", multi_source_retrieve_node)
    workflow.add_node("Grade", enhanced_grade_node)
    workflow.add_node("Visualization", visualization_node)
    workflow.add_node("Generate", enhanced_generation_node)
    workflow.add_node("Evaluate", answer_check_node)
    workflow.add_node("Adapt", strategy_adaptation_node)
    
    # Set entry point
    workflow.set_entry_point("ToolSelection")
    
    # Define workflow edges
    workflow.add_edge("ToolSelection", "MultiRetrieve")
    workflow.add_edge("MultiRetrieve", "Grade")
    
    workflow.add_conditional_edges(
        "Grade",
        lambda state: "Yes" if state["relevant"] else "No",
        {
            "Yes": "Visualization",
            "No": "Adapt"
        }
    )
    
    workflow.add_edge("Visualization", "Generate")
    workflow.add_edge("Generate", "Evaluate")
    
    workflow.add_conditional_edges(
        "Evaluate",
        lambda state: "Yes" if state["answered"] else "No",
        {
            "Yes": END,
            "No": "Adapt"
        }
    )
    
    workflow.add_edge("Adapt", "MultiRetrieve")
    
    # Compile workflow
    return workflow.compile()

def process_research_query(session_id: str, query: str) -> ChatResponse:
    """Process research query using agentic workflow"""
    try:
        # Get session data
        if session_id not in sessions:
            raise HTTPException(status_code=400, detail="Session not found. Please upload documents first.")
        
        session = sessions[session_id]
        uploaded_docs = session.get('research_docs', [])
        
        if not uploaded_docs:
            raise HTTPException(status_code=400, detail="No documents found in session. Please upload documents first.")
        
        # Build workflow
        workflow = build_research_workflow()
        
        # Prepare initial state
        initial_state = {
            "question": query,
            "original_question": query,
            "docs": uploaded_docs,
            "external_docs": None,
            "answer": None,
            "relevant": None,
            "answered": None,
            "selected_tools": None,
            "search_strategy": None,
            "iteration_count": 0,
            "reasoning": None,
            "visualization_needed": None,
            "chart_data": None,
            "plot_url": None
        }
        
        # Execute workflow with increased recursion limit
        result = workflow.invoke(initial_state, config={"recursion_limit": 50})
        
        # Store in session history
        if 'research_messages' not in session:
            session['research_messages'] = []
        
        session['research_messages'].append({
            'query': query,
            'response': result.get("answer", "No answer generated"),
            'plot_url': result.get("plot_url"),
            'reasoning': result.get("reasoning", ""),
            'selected_tools': result.get("selected_tools", []),
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        return ChatResponse(
            response=result.get("answer", "No answer generated"),
            plot_url=result.get("plot_url"),
            thinking=f"Research workflow completed. Tools used: {', '.join(result.get('selected_tools', []))}. Reasoning: {result.get('reasoning', '')}",
            code=None
        )
        
    except Exception as e:
        error_msg = f"Research query failed: {str(e)}"
        return ChatResponse(
            response=error_msg,
            plot_url=None,
            thinking=f"Error occurred during research workflow: {str(e)}",
            code=None
        )

# === API Routes ===

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Galvatron AI 8.0</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <link href="https://cdn.jsdelivr.net/npm/marked/marked.min.js">
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            :root {
                --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                --danger-gradient: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
                --dark-bg: #1a1a2e;
                --card-bg: #16213e;
                --text-muted: #6c757d;
            }
            
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .main-container {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                margin: 20px;
                overflow: hidden;
            }
            
            .sidebar {
                background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
                color: white;
                min-height: calc(100vh - 40px);
            }
            
            .nav-tabs .nav-link {
                background: rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.2);
                margin-right: 5px;
                border-radius: 10px 10px 0 0;
                transition: all 0.3s ease;
            }
            
            .nav-tabs .nav-link:hover {
                background: rgba(255, 255, 255, 0.2);
                color: white;
            }
            
            .nav-tabs .nav-link.active {
                background: rgba(255, 255, 255, 0.9);
                color: #2c3e50;
                border-color: rgba(255, 255, 255, 0.3);
            }
            
            .tab-content {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 0 10px 10px 10px;
                padding: 20px;
                margin-top: -1px;
            }
            
            .chat-section {
                min-height: calc(100vh - 40px);
                display: flex;
                flex-direction: column;
            }
            
            .chat-container {
                flex-grow: 1;
                height: 60vh;
                overflow-y: auto;
                background: #f8f9fa;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;
                scrollbar-width: thin;
                scrollbar-color: #667eea #f8f9fa;
            }
            
            .chat-container::-webkit-scrollbar {
                width: 6px;
            }
            
            .chat-container::-webkit-scrollbar-track {
                background: #f8f9fa;
            }
            
            .chat-container::-webkit-scrollbar-thumb {
                background: #667eea;
                border-radius: 3px;
            }
            
            .message {
                margin-bottom: 20px;
                animation: fadeInUp 0.3s ease-out;
            }
            
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .user-message {
                background: var(--primary-gradient);
                color: white;
                border-radius: 20px 20px 5px 20px;
                padding: 15px 20px;
                margin-left: 20%;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            }
            
            .assistant-message {
                background: white;
                border: 1px solid #e9ecef;
                border-radius: 20px 20px 20px 5px;
                padding: 20px;
                margin-right: 20%;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            }
            
            .thinking-box, .code-box {
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 4px solid #667eea;
                margin-top: 15px;
            }
            
            .thinking-box summary, .code-box summary {
                padding: 10px 15px;
                cursor: pointer;
                font-weight: 500;
                color: #667eea;
                transition: all 0.3s ease;
            }
            
            .thinking-box summary:hover, .code-box summary:hover {
                background: rgba(102, 126, 234, 0.1);
            }
            
            .upload-area {
                border: 2px dashed #667eea;
                border-radius: 15px;
                padding: 30px;
                text-align: center;
                transition: all 0.3s ease;
                cursor: pointer;
                background: rgba(102, 126, 234, 0.05);
            }
            
            .upload-area:hover {
                border-color: #764ba2;
                background: rgba(102, 126, 234, 0.1);
                transform: translateY(-2px);
            }
            
            .upload-area.dragover {
                border-color: #38ef7d;
                background: rgba(56, 239, 125, 0.1);
            }
            
            .dataset-preview {
                max-height: 300px;
                overflow-y: auto;
                border-radius: 10px;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(5px);
            }
            
            .dataset-preview table {
                color: white;
            }
            
            .dataset-preview th {
                background: rgba(255, 255, 255, 0.2);
                border: none;
            }
            
            .dataset-preview td {
                border-color: rgba(255, 255, 255, 0.1);
            }
            
            .insights-container {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(5px);
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
            }
            
            .insights-container h2, .insights-container h3 {
                color: #fff;
                margin-bottom: 15px;
            }
            
            .insights-container ul {
                color: rgba(255, 255, 255, 0.9);
            }
            
            .insights-container li {
                margin-bottom: 8px;
                padding-left: 10px;
                position: relative;
            }
            
            .insights-container li::before {
                content: "▸";
                color: #38ef7d;
                position: absolute;
                left: 0;
            }
            
            .database-input-group {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(5px);
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
            }
            
            .database-input-group label {
                color: white;
                font-weight: 500;
                margin-bottom: 8px;
                display: block;
            }
            
            .database-input-group input, .database-input-group textarea {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                color: #333;
            }
            
            .database-input-group input:focus, .database-input-group textarea:focus {
                background: white;
                border-color: #38ef7d;
                box-shadow: 0 0 0 0.2rem rgba(56, 239, 125, 0.25);
            }
            
            .btn-primary {
                background: var(--primary-gradient);
                border: none;
                border-radius: 25px;
                padding: 10px 25px;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            
            .form-control {
                border-radius: 25px;
                border: 2px solid #e9ecef;
                padding: 12px 20px;
                transition: all 0.3s ease;
            }
            
            .form-control:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
            }
            
            .loading-spinner {
                text-align: center;
                padding: 20px;
            }
            
            .spinner-border {
                width: 3rem;
                height: 3rem;
            }
            
            .alert-warning {
                background: var(--success-gradient);
                border: none;
                color: white;
                border-radius: 15px;
            }
            
            .empty-chat {
                text-align: center;
                color: #6c757d;
                padding: 40px 20px;
            }
            
            .empty-chat i {
                font-size: 3rem;
                margin-bottom: 20px;
                opacity: 0.5;
            }
            
            .brand-title {
                background: linear-gradient(45deg, #fff, #f0f0f0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-weight: bold;
                margin-bottom: 10px;
            }
            
            .nvidia-link {
                color: rgba(255, 255, 255, 0.7);
                text-decoration: none;
                transition: color 0.3s ease;
            }
            
            .nvidia-link:hover {
                color: #38ef7d;
            }
            
            .plot-image {
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                transition: transform 0.3s ease;
            }
            
            .plot-image:hover {
                transform: scale(1.02);
            }
            
            @media (max-width: 768px) {
                .main-container {
                    margin: 10px;
                    border-radius: 15px;
                }
                
                .user-message {
                    margin-left: 10%;
                }
                
                .assistant-message {
                    margin-right: 10%;
                }
            }
        </style>
    </head>
    <body>
        <div class="container-fluid main-container">
            <div class="row g-0">
                <!-- Left Panel -->
                <div class="col-lg-4 sidebar">
                    <div class="p-4">
                        <h3 class="brand-title"><i class="fas fa-chart-line"></i> Galvatron AI 8.0</h3>
                        <p class="small mb-4">Powered by <a href="https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1" target="_blank" class="nvidia-link">Galvatron AI & Co</a></p>
                        
                        <!-- Tabs Navigation -->
                        <ul class="nav nav-tabs mb-3" id="mainTabs" role="tablist">
                            <li class="nav-item" role="presentation">
                                <button class="nav-link active" id="csv-tab" data-bs-toggle="tab" data-bs-target="#csv-panel" type="button" role="tab">
                                    <i class="fas fa-file-csv"></i> CSV Data
                                </button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="database-tab" data-bs-toggle="tab" data-bs-target="#database-panel" type="button" role="tab">
                                    <i class="fas fa-database"></i> Database
                                </button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="research-tab" data-bs-toggle="tab" data-bs-target="#research-panel" type="button" role="tab">
                                    <i class="fas fa-search"></i> Research
                                </button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="google-drive-tab" data-bs-toggle="tab" data-bs-target="#google-drive-panel" type="button" role="tab">
                                    <i class="fab fa-google-drive"></i> Google Drive
                                </button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="dashboard-tab" data-bs-toggle="tab" data-bs-target="#dashboard-panel" type="button" role="tab">
                                    <i class="fas fa-chart-line"></i> Auto Dashboard
                                </button>
                            </li>
                        </ul>
                        
                        <!-- Tab Content -->
                        <div class="tab-content" id="mainTabContent">
                            <!-- CSV Upload Tab -->
                            <div class="tab-pane fade show active" id="csv-panel" role="tabpanel">
                                <!-- File Upload -->
                                <div class="mb-4">
                                    <div class="upload-area" id="uploadArea">
                                        <i class="fas fa-cloud-upload-alt fa-3x mb-3" style="color: #667eea;"></i>
                                        <h5>Upload Your Dataset</h5>
                                        <p class="mb-0">Click or drag CSV file here</p>
                                        <input type="file" id="csvFile" accept=".csv" style="display: none;">
                                    </div>
                                </div>
                                
                                <!-- Dataset Info -->
                                <div id="datasetInfo" style="display: none;">
                                    <h5 class="mb-3"><i class="fas fa-table"></i> Dataset Preview</h5>
                                    <div id="datasetPreview" class="dataset-preview border rounded mb-3"></div>
                                    <div id="datasetInsights" class="insights-container"></div>
                                </div>
                                
                                <div id="uploadPrompt" class="alert alert-warning">
                                    <i class="fas fa-info-circle me-2"></i>
                                    <strong>Get Started!</strong><br>
                                    Upload a CSV file to begin analyzing your data with AI.
                                </div>
                            </div>
                            
                            <!-- Database Chat Tab -->
                            <div class="tab-pane fade" id="database-panel" role="tabpanel">
                                <div class="database-input-group">
                                    <label for="databaseUri">
                                        <i class="fas fa-database me-2"></i>Database URI
                                    </label>
                                    <input type="text" id="databaseUri" class="form-control mb-3" 
                                           placeholder="e.g., sqlite:///database.db or postgresql://user:pass@host:port/db">
                                    <small class="text-light">
                                        Supported: SQLite, PostgreSQL, MySQL, SQL Server
                                    </small>
                                </div>
                                
                                <div class="alert alert-info" style="background: rgba(56, 239, 125, 0.2); border: 1px solid rgba(56, 239, 125, 0.3); color: white;">
                                    <i class="fas fa-lightbulb me-2"></i>
                                    <strong>Examples:</strong><br>
                                    • <code>sqlite:///Chinook_Sqlite.sqlite</code><br>
                                    • <code>postgresql://user:pass@localhost:5432/mydb</code><br>
                                    • <code>mysql://user:pass@localhost:3306/mydb</code>
                                </div>
                            </div>
                            
                            <!-- Research Tab -->
                            <div class="tab-pane fade" id="research-panel" role="tabpanel">
                                <!-- Multiple File Upload -->
                                <div class="mb-4">
                                    <div class="upload-area" id="researchUploadArea">
                                        <i class="fas fa-file-upload fa-3x mb-3" style="color: #667eea;"></i>
                                        <h5>Upload Research Documents</h5>
                                        <p class="mb-0">Click or drag multiple files here</p>
                                        <p class="small text-muted">Supports: PDF, TXT, DOCX, XLSX, PPTX</p>
                                        <input type="file" id="researchFiles" multiple accept=".pdf,.txt,.docx,.xlsx,.pptx" style="display: none;">
                                    </div>
                                </div>
                                
                                <!-- Uploaded Files List -->
                                <div id="uploadedFilesList" style="display: none;">
                                    <h6 class="mb-3"><i class="fas fa-files"></i> Uploaded Files</h6>
                                    <div id="filesContainer" class="mb-3"></div>
                                </div>
                                
                                <!-- Research Tools Configuration -->
                                <div class="database-input-group">
                                    <label>
                                        <i class="fas fa-tools me-2"></i>Research Tools
                                    </label>
                                    <div class="row">
                                        <div class="col-md-4">
                                            <div class="form-check">
                                                <input class="form-check-input" type="checkbox" id="enableWikipedia" checked>
                                                <label class="form-check-label text-light" for="enableWikipedia">
                                                    Wikipedia
                                                </label>
                                            </div>
                                        </div>
                                        <div class="col-md-4">
                                            <div class="form-check">
                                                <input class="form-check-input" type="checkbox" id="enableArxiv" checked>
                                                <label class="form-check-label text-light" for="enableArxiv">
                                                    ArXiv
                                                </label>
                                            </div>
                                        </div>
                                        <div class="col-md-4">
                                            <div class="form-check">
                                                <input class="form-check-input" type="checkbox" id="enableTavily" checked>
                                                <label class="form-check-label text-light" for="enableTavily">
                                                    Web Search
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="alert alert-info" style="background: rgba(56, 239, 125, 0.2); border: 1px solid rgba(56, 239, 125, 0.3); color: white;">
                                    <i class="fas fa-robot me-2"></i>
                                    <strong>Agentic AI Research:</strong><br>
                                    Upload documents and ask research questions. The AI will intelligently select tools, search multiple sources, and provide comprehensive answers with visualizations when needed.
                                </div>
                            </div>

                            <!-- Google Drive Tab -->
                            <div class="tab-pane fade" id="google-drive-panel" role="tabpanel">
                                <!-- Google Drive URL Input -->
                                <div class="mb-4">
                                    <div class="database-input-group">
                                        <label for="googleDriveUrl">
                                            <i class="fab fa-google-drive me-2"></i>Google Drive Folder URL
                                        </label>
                                        <input
                                            type="text"
                                            class="form-control"
                                            id="googleDriveUrl"
                                            placeholder="https://drive.google.com/drive/folders/YOUR_FOLDER_ID"
                                            style="background: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); color: white;">
                                        <div class="mt-2">
                                            <div class="form-check">
                                                <input class="form-check-input" type="checkbox" id="recursiveLoad">
                                                <label class="form-check-label text-light" for="recursiveLoad">
                                                    Load files from subfolders (recursive)
                                                </label>
                                            </div>
                                        </div>
                                        <button class="btn btn-primary mt-3 w-100" id="loadGoogleDriveBtn">
                                            <i class="fab fa-google-drive me-2"></i>Load from Google Drive
                                        </button>
                                    </div>
                                </div>

                                <!-- Loaded Files List -->
                                <div id="googleDriveFilesList" style="display: none;">
                                    <h6 class="mb-3 text-light"><i class="fas fa-files"></i> Loaded Files</h6>
                                    <div id="googleDriveFilesContainer" class="mb-3"></div>
                                </div>

                                <div class="alert alert-info" style="background: rgba(56, 239, 125, 0.2); border: 1px solid rgba(56, 239, 125, 0.3); color: white;">
                                    <i class="fab fa-google-drive me-2"></i>
                                    <strong>Google Drive Integration:</strong><br>
                                    Paste your Google Drive folder URL and load documents directly from your drive. Supports Google Docs, Sheets, and PDFs. The first time you use this feature, you'll need to authorize access to your Google Drive.
                                </div>
                            </div>

                            <!-- Automatic Dashboard Tab -->
                            <div class="tab-pane fade" id="dashboard-panel" role="tabpanel">
                                <div class="alert alert-info" style="background: rgba(102, 126, 234, 0.2); border: 1px solid rgba(102, 126, 234, 0.3); color: white;">
                                    <i class="fas fa-chart-line me-2"></i>
                                    <strong>Dashboard Generator</strong><br>
                                    The dashboard interface is displayed on the right side →<br>
                                    Upload a CSV file and use the dashboard section to generate visualizations.
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Right Panel -->
                <div class="col-lg-8 chat-section">
                    <div class="p-4 h-100 d-flex flex-column">
                        <div class="d-flex align-items-center mb-4">
                            <h3 class="mb-0">
                                <i class="fas fa-comments text-primary"></i> 
                                <span id="chatTitle">Chat with your data</span>
                            </h3>
                            <div class="ms-auto">
                                <button id="clearChat" class="btn btn-outline-secondary btn-sm" style="display: none;">
                                    <i class="fas fa-trash"></i> Clear Chat
                                </button>
                            </div>
                        </div>
                        
                        <!-- Chat Container -->
                        <div id="chatContainer" class="chat-container">
                            <div id="messages">
                                <div class="empty-chat" id="emptyChatMessage">
                                    <i class="fas fa-robot"></i>
                                    <h5>Ready to analyze your data!</h5>
                                    <p>Upload a CSV file or connect to a database to start asking questions.</p>
                                </div>
                            </div>
                        </div>

                        <!-- Chat Input -->
                        <div class="input-group">
                            <input type="text" id="chatInput" class="form-control" placeholder="Ask me anything about your data..." disabled>
                            <button id="sendButton" class="btn btn-primary" disabled>
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>

                        <!-- Loading Indicator -->
                        <div id="loading" class="loading-spinner" style="display: none;">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Analyzing...</span>
                            </div>
                            <p class="text-muted mt-3">🤖 AI is analyzing your query...</p>
                        </div>

                        <!-- Dashboard Section (shown when dashboard tab is active) -->
                        <div id="dashboardSection" class="h-100 flex-column d-none">
                            <div class="text-center mb-4">
                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 15px;">
                                    <i class="fas fa-chart-line fa-4x mb-3" style="color: white;"></i>
                                    <h3 style="color: white; margin-bottom: 10px;">Automatic Dashboard Generator</h3>
                                    <p style="color: rgba(255,255,255,0.9); margin-bottom: 25px;">
                                        Transform your CSV data into a beautiful, interactive dashboard like Tableau or Power BI
                                    </p>
                                    <button class="btn btn-light btn-lg" id="generateDashboardBtn" style="padding: 15px 40px; font-size: 18px;">
                                        <i class="fas fa-magic me-2"></i>Generate Dashboard
                                    </button>
                                </div>
                            </div>

                            <!-- Dashboard Features -->
                            <div class="row mb-4">
                                <div class="col-md-3 text-center mb-3">
                                    <div style="background: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 10px; height: 100%;">
                                        <i class="fas fa-tachometer-alt fa-3x mb-3" style="color: #667eea;"></i>
                                        <h5 style="color: white;">KPI Cards</h5>
                                        <p class="small text-light">Key metrics at a glance</p>
                                    </div>
                                </div>
                                <div class="col-md-3 text-center mb-3">
                                    <div style="background: rgba(56, 239, 125, 0.1); padding: 20px; border-radius: 10px; height: 100%;">
                                        <i class="fas fa-chart-bar fa-3x mb-3" style="color: #38ef7d;"></i>
                                        <h5 style="color: white;">Interactive Charts</h5>
                                        <p class="small text-light">Multiple visualizations</p>
                                    </div>
                                </div>
                                <div class="col-md-3 text-center mb-3">
                                    <div style="background: rgba(255, 193, 7, 0.1); padding: 20px; border-radius: 10px; height: 100%;">
                                        <i class="fas fa-filter fa-3x mb-3" style="color: #ffc107;"></i>
                                        <h5 style="color: white;">Smart Filters</h5>
                                        <p class="small text-light">Dynamic data filtering</p>
                                    </div>
                                </div>
                                <div class="col-md-3 text-center mb-3">
                                    <div style="background: rgba(255, 65, 108, 0.1); padding: 20px; border-radius: 10px; height: 100%;">
                                        <i class="fas fa-table fa-3x mb-3" style="color: #ff416c;"></i>
                                        <h5 style="color: white;">Data Tables</h5>
                                        <p class="small text-light">Sortable & searchable</p>
                                    </div>
                                </div>
                            </div>

                            <!-- Generated Dashboard Preview -->
                            <div id="dashboardPreview" style="display: none; flex: 1; min-height: 800px;">
                                <div class="mb-3 d-flex justify-content-between align-items-center">
                                    <h5 style="color: white;"><i class="fas fa-eye me-2"></i>Dashboard Preview</h5>
                                    <div>
                                        <button class="btn btn-success btn-sm" id="openDashboardBtn">
                                            <i class="fas fa-external-link-alt me-2"></i>Open in New Tab
                                        </button>
                                        <button class="btn btn-primary btn-sm ms-2" id="regenerateDashboardBtn">
                                            <i class="fas fa-sync-alt me-2"></i>Regenerate
                                        </button>
                                    </div>
                                </div>
                                <div style="background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 750px;">
                                    <iframe id="dashboardFrame" style="width: 100%; height: 100%; border: none;"></iframe>
                                </div>
                            </div>

                            <!-- Loading State -->
                            <div id="dashboardLoading" style="display: none; text-align: center; padding: 60px;">
                                <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
                                    <span class="visually-hidden">Generating...</span>
                                </div>
                                <h5 class="mt-4" style="color: white;">🤖 AI is creating your dashboard...</h5>
                                <p class="text-light">This may take 30-60 seconds. We're analyzing your data and generating interactive visualizations.</p>
                            </div>

                            <div class="alert alert-info mt-3" style="background: rgba(56, 239, 125, 0.2); border: 1px solid rgba(56, 239, 125, 0.3); color: white;">
                                <i class="fas fa-info-circle me-2"></i>
                                <strong>How it works:</strong><br>
                                1. Upload a CSV file in the CSV Upload tab<br>
                                2. Click "Generate Dashboard" to create an interactive dashboard<br>
                                3. Get professional visualizations with KPIs, charts, filters, and data tables<br>
                                4. Powered by NVIDIA LLAMA Nemotron AI
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            let sessionId = null;
            let messageCount = 0;
            let currentMode = 'csv'; // 'csv', 'database', 'research', or 'google-drive'
            let researchSessionId = null;
            let googleDriveSessionId = null;
            let uploadedFiles = [];
            
            // Generate session ID
            function generateSessionId() {
                return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            }
            
            // Tab switching functionality
            document.addEventListener('DOMContentLoaded', function() {
                const tabs = document.querySelectorAll('#mainTabs button[data-bs-toggle="tab"]');
                tabs.forEach(tab => {
                    tab.addEventListener('shown.bs.tab', function(e) {
                        const targetId = e.target.getAttribute('data-bs-target');
                        if (targetId === '#csv-panel') {
                            currentMode = 'csv';
                            document.getElementById('chatTitle').textContent = 'Chat with your data';
                            updateChatState();
                            showChatSection();
                        } else if (targetId === '#database-panel') {
                            currentMode = 'database';
                            document.getElementById('chatTitle').textContent = 'Chat with your database';
                            updateChatState();
                            showChatSection();
                        } else if (targetId === '#research-panel') {
                            currentMode = 'research';
                            document.getElementById('chatTitle').textContent = 'Agentic AI Research';
                            updateChatState();
                            showChatSection();
                        } else if (targetId === '#google-drive-panel') {
                            currentMode = 'google-drive';
                            document.getElementById('chatTitle').textContent = 'Chat with Google Drive';
                            updateChatState();
                            showChatSection();
                        } else if (targetId === '#dashboard-panel') {
                            currentMode = 'dashboard';
                            document.getElementById('chatTitle').textContent = 'Automatic Dashboard Generator';
                            updateChatState();
                            showDashboardSection();
                        }
                    });
                });

                // Initialize with chat section visible
                showChatSection();
            });

            // Function to show chat section and hide dashboard
            function showChatSection() {
                const chatContainer = document.getElementById('chatContainer');
                const chatInput = document.querySelector('.input-group');
                const loading = document.getElementById('loading');
                const dashboardSection = document.getElementById('dashboardSection');
                const clearChatBtn = document.getElementById('clearChat');

                // Show chat elements
                if (chatContainer) {
                    chatContainer.classList.remove('d-none');
                    chatContainer.style.display = 'block';
                }
                if (chatInput) {
                    chatInput.classList.remove('d-none');
                    chatInput.style.display = 'flex';
                }

                // Hide dashboard section
                if (dashboardSection) {
                    dashboardSection.classList.remove('d-flex');
                    dashboardSection.classList.add('d-none');
                }

                if (clearChatBtn && messageCount > 0) clearChatBtn.style.display = 'block';
            }

            // Function to show dashboard section and hide chat
            function showDashboardSection() {
                const chatContainer = document.getElementById('chatContainer');
                const chatInput = document.querySelector('.input-group');
                const loading = document.getElementById('loading');
                const dashboardSection = document.getElementById('dashboardSection');
                const clearChatBtn = document.getElementById('clearChat');

                // Hide chat elements
                if (chatContainer) {
                    chatContainer.classList.add('d-none');
                    chatContainer.style.display = 'none';
                }
                if (chatInput) {
                    chatInput.classList.add('d-none');
                    chatInput.style.display = 'none';
                }
                if (loading) {
                    loading.style.display = 'none';
                }

                // Show dashboard section
                if (dashboardSection) {
                    dashboardSection.classList.remove('d-none');
                    dashboardSection.classList.add('d-flex');
                }

                if (clearChatBtn) clearChatBtn.style.display = 'none';
            }
            
            // Update chat input state based on current mode
            function updateChatState() {
                const chatInput = document.getElementById('chatInput');
                const sendButton = document.getElementById('sendButton');
                
                // Check if elements exist before trying to access them
                if (!chatInput || !sendButton) {
                    console.warn('Chat input elements not found');
                    return;
                }
                
                if (currentMode === 'csv') {
                    const hasSession = sessionId !== null;
                    chatInput.disabled = !hasSession;
                    sendButton.disabled = !hasSession;
                    chatInput.placeholder = hasSession ? 
                        "Ask me anything about your data..." : 
                        "Upload a CSV file first...";
                } else if (currentMode === 'database') {
                    const databaseUriElement = document.getElementById('databaseUri');
                    const hasUri = databaseUriElement ? databaseUriElement.value.trim() : false;
                    chatInput.disabled = !hasUri;
                    sendButton.disabled = !hasUri;
                    chatInput.placeholder = hasUri ? 
                        "Ask me anything about your database..." : 
                        "Enter database URI first...";
                } else if (currentMode === 'research') {
                    const hasFiles = uploadedFiles.length > 0;
                    chatInput.disabled = !hasFiles;
                    sendButton.disabled = !hasFiles;
                    chatInput.placeholder = hasFiles ?
                        "Ask research questions about your documents..." :
                        "Upload documents first...";
                } else if (currentMode === 'google-drive') {
                    const hasSession = googleDriveSessionId !== null;
                    chatInput.disabled = !hasSession;
                    sendButton.disabled = !hasSession;
                    chatInput.placeholder = hasSession ?
                        "Ask questions about your Google Drive documents..." :
                        "Load Google Drive folder first...";
                }
            }
            
            // Database URI input handler
            const databaseUriInput = document.getElementById('databaseUri');
            if (databaseUriInput) {
                databaseUriInput.addEventListener('input', function() {
                    if (currentMode === 'database') {
                        updateChatState();
                    }
                });
            }
            
            // Initialize research file upload
            function initializeResearchUpload() {
                const uploadArea = document.getElementById('researchUploadArea');
                const fileInput = document.getElementById('researchFiles');
                
                // Check if elements exist before adding event listeners
                if (!uploadArea || !fileInput) {
                    console.warn('Research upload elements not found');
                    return;
                }
                
                uploadArea.addEventListener('click', () => fileInput.click());
                
                ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                    uploadArea.addEventListener(eventName, preventDefaults, false);
                });
                
                function preventDefaults(e) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                
                ['dragenter', 'dragover'].forEach(eventName => {
                    uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
                });
                
                ['dragleave', 'drop'].forEach(eventName => {
                    uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
                });
                
                uploadArea.addEventListener('drop', handleResearchDrop, false);
                
                function handleResearchDrop(e) {
                    const dt = e.dataTransfer;
                    const files = dt.files;
                    if (files.length > 0) {
                        handleResearchFiles(Array.from(files));
                    }
                }
            }
            
            // Handle research file upload
            async function handleResearchFiles(files) {
                if (!researchSessionId) {
                    researchSessionId = generateSessionId();
                }
                
                const validExtensions = ['.pdf', '.txt', '.docx', '.xlsx', '.pptx'];
                const validFiles = files.filter(file => {
                    const ext = '.' + file.name.split('.').pop().toLowerCase();
                    return validExtensions.includes(ext);
                });
                
                if (validFiles.length === 0) {
                    showErrorMessage('Please upload valid documents (PDF, TXT, DOCX, XLSX, PPTX)');
                    return;
                }
                
                try {
                    showLoading();
                    
                    for (const file of validFiles) {
                        const formData = new FormData();
                        formData.append('file', file);
                        formData.append('session_id', researchSessionId);
                        
                        const response = await fetch('/research-upload', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const result = await response.json();
                        
                        if (response.ok) {
                            uploadedFiles.push({
                                name: file.name,
                                size: file.size,
                                uploadTime: new Date().toISOString()
                            });
                        } else {
                            throw new Error(result.detail || `Failed to upload ${file.name}`);
                        }
                    }
                    
                    displayUploadedFiles();
                    updateChatState();
                    clearMessages();
                    showSuccessMessage(`Successfully uploaded ${validFiles.length} file(s) for research.`);
                    
                } catch (error) {
                    showErrorMessage('Error uploading files: ' + error.message);
                } finally {
                    hideLoading();
                }
            }
            
            // Display uploaded files
            function displayUploadedFiles() {
                const container = document.getElementById('filesContainer');
                const listElement = document.getElementById('uploadedFilesList');
                
                // Check if elements exist before trying to access them
                if (!container || !listElement) {
                    console.warn('Files container elements not found');
                    return;
                }
                
                if (uploadedFiles.length > 0) {
                    listElement.style.display = 'block';
                    
                    let html = '';
                    uploadedFiles.forEach((file, index) => {
                        const sizeKB = Math.round(file.size / 1024);
                        html += `
                            <div class="d-flex justify-content-between align-items-center p-2 mb-2" style="background: rgba(255,255,255,0.1); border-radius: 8px;">
                                <div>
                                    <i class="fas fa-file me-2"></i>
                                    <span class="text-light">${file.name}</span>
                                    <small class="text-muted ms-2">(${sizeKB} KB)</small>
                                </div>
                                <button class="btn btn-sm btn-outline-danger" onclick="removeFile(${index})">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        `;
                    });
                    
                    container.innerHTML = html;
                } else {
                    listElement.style.display = 'none';
                }
            }
            
            // Remove uploaded file
            window.removeFile = function(index) {
                uploadedFiles.splice(index, 1);
                displayUploadedFiles();
                updateChatState();
            };
            
            // Research file input change handler
            const researchFilesInput = document.getElementById('researchFiles');
            if (researchFilesInput) {
                researchFilesInput.addEventListener('change', function(e) {
                    const files = Array.from(e.target.files);
                    if (files.length > 0) {
                        handleResearchFiles(files);
                    }
                });
            }

            // Google Drive load button handler
            const loadGoogleDriveBtn = document.getElementById('loadGoogleDriveBtn');
            if (loadGoogleDriveBtn) {
                loadGoogleDriveBtn.addEventListener('click', async function() {
                    const urlInput = document.getElementById('googleDriveUrl');
                    const recursiveCheckbox = document.getElementById('recursiveLoad');
                    const driveUrl = urlInput ? urlInput.value.trim() : '';

                    if (!driveUrl) {
                        showErrorMessage('Please enter a Google Drive folder URL');
                        return;
                    }

                    // Generate session ID if not exists
                    if (!googleDriveSessionId) {
                        googleDriveSessionId = generateSessionId();
                    }

                    try {
                        loadGoogleDriveBtn.disabled = true;
                        loadGoogleDriveBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Loading...';

                        const response = await fetch('/google-drive-load', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                session_id: googleDriveSessionId,
                                drive_url: driveUrl,
                                recursive: recursiveCheckbox ? recursiveCheckbox.checked : false
                            })
                        });

                        const data = await response.json();

                        if (response.ok && data.status === 'success') {
                            // Display loaded files
                            const filesContainer = document.getElementById('googleDriveFilesContainer');
                            const filesList = document.getElementById('googleDriveFilesList');

                            if (filesContainer && filesList) {
                                filesContainer.innerHTML = data.files.map(file =>
                                    `<div class="badge bg-success me-2 mb-2"><i class="fas fa-file me-1"></i>${file}</div>`
                                ).join('');
                                filesList.style.display = 'block';
                            }

                            showSuccessMessage(data.message);
                            updateChatState();
                        } else {
                            showErrorMessage(data.detail || 'Failed to load Google Drive folder');
                        }
                    } catch (error) {
                        showErrorMessage('Error loading Google Drive: ' + error.message);
                    } finally {
                        loadGoogleDriveBtn.disabled = false;
                        loadGoogleDriveBtn.innerHTML = '<i class="fab fa-google-drive me-2"></i>Load from Google Drive';
                    }
                });
            }

            // Dashboard generation handlers
            let currentDashboardUrl = null;

            const generateDashboardBtn = document.getElementById('generateDashboardBtn');
            if (generateDashboardBtn) {
                generateDashboardBtn.addEventListener('click', async function() {
                    if (!sessionId) {
                        alert('Please upload a CSV file first in the CSV Upload tab.');
                        return;
                    }

                    try {
                        // Show loading state
                        document.getElementById('dashboardLoading').style.display = 'block';
                        document.getElementById('dashboardPreview').style.display = 'none';
                        generateDashboardBtn.disabled = true;

                        const formData = new FormData();
                        formData.append('session_id', sessionId);

                        const response = await fetch('/dashboard-generate', {
                            method: 'POST',
                            body: formData
                        });

                        const data = await response.json();

                        if (response.ok && data.success) {
                            currentDashboardUrl = data.dashboard_url;

                            // Show preview
                            document.getElementById('dashboardLoading').style.display = 'none';
                            document.getElementById('dashboardPreview').style.display = 'block';

                            // Load dashboard in iframe
                            document.getElementById('dashboardFrame').src = currentDashboardUrl;

                            showSuccessMessage('Dashboard generated successfully!');
                        } else {
                            document.getElementById('dashboardLoading').style.display = 'none';
                            showErrorMessage(data.message || 'Failed to generate dashboard');
                        }
                    } catch (error) {
                        document.getElementById('dashboardLoading').style.display = 'none';
                        showErrorMessage('Error generating dashboard: ' + error.message);
                    } finally {
                        generateDashboardBtn.disabled = false;
                    }
                });
            }

            const openDashboardBtn = document.getElementById('openDashboardBtn');
            if (openDashboardBtn) {
                openDashboardBtn.addEventListener('click', function() {
                    if (currentDashboardUrl) {
                        window.open(currentDashboardUrl, '_blank');
                    }
                });
            }

            const regenerateDashboardBtn = document.getElementById('regenerateDashboardBtn');
            if (regenerateDashboardBtn) {
                regenerateDashboardBtn.addEventListener('click', function() {
                    generateDashboardBtn.click();
                });
            }

            // Initialize drag and drop
            function initializeDragDrop() {
                const uploadArea = document.getElementById('uploadArea');
                const fileInput = document.getElementById('csvFile');
                
                uploadArea.addEventListener('click', () => fileInput.click());
                
                ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                    uploadArea.addEventListener(eventName, preventDefaults, false);
                });
                
                function preventDefaults(e) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                
                ['dragenter', 'dragover'].forEach(eventName => {
                    uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
                });
                
                ['dragleave', 'drop'].forEach(eventName => {
                    uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
                });
                
                uploadArea.addEventListener('drop', handleDrop, false);
                
                function handleDrop(e) {
                    const dt = e.dataTransfer;
                    const files = dt.files;
                    if (files.length > 0) {
                        handleFile(files[0]);
                    }
                }
            }
            
            // Handle file upload
            async function handleFile(file) {
                if (!file.name.toLowerCase().endsWith('.csv')) {
                    alert('Please upload a CSV file.');
                    return;
                }
                
                sessionId = generateSessionId();
                const formData = new FormData();
                formData.append('file', file);
                formData.append('session_id', sessionId);
                
                try {
                    showLoading();
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        displayDatasetInfo(result);
                        updateChatState();
                        clearMessages();
                        showSuccessMessage(`Successfully loaded ${file.name} with ${result.rows} rows and ${result.columns.length} columns.`);
                    } else {
                        throw new Error(result.detail || 'Upload failed');
                    }
                } catch (error) {
                    showErrorMessage('Error uploading file: ' + error.message);
                } finally {
                    hideLoading();
                }
            }
            
            // File input change handler
            document.getElementById('csvFile').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    handleFile(file);
                }
            });
            
            // Display dataset information
            function displayDatasetInfo(data) {
                const uploadPrompt = document.getElementById('uploadPrompt');
                const datasetInfo = document.getElementById('datasetInfo');
                
                if (uploadPrompt) uploadPrompt.style.display = 'none';
                if (datasetInfo) datasetInfo.style.display = 'block';
                
                // Create table preview
                if (data.preview && data.preview.length > 0) {
                    let tableHtml = '<table class="table table-sm table-hover"><thead><tr>';
                    Object.keys(data.preview[0]).forEach(col => {
                        tableHtml += `<th>${col}</th>`;
                    });
                    tableHtml += '</tr></thead><tbody>';
                    
                    data.preview.slice(0, 5).forEach(row => {
                        tableHtml += '<tr>';
                        Object.values(row).forEach(val => {
                            const displayVal = val !== null && val !== undefined ? String(val).substring(0, 50) : '';
                            tableHtml += `<td>${displayVal}</td>`;
                        });
                        tableHtml += '</tr>';
                    });
                    tableHtml += '</tbody></table>';
                    
                    document.getElementById('datasetPreview').innerHTML = tableHtml;
                }
                
                // Display insights with markdown rendering
                const insightsHtml = marked.parse(data.insights);
                document.getElementById('datasetInsights').innerHTML = insightsHtml;
            }
            
            // Clear messages
            function clearMessages() {
                const messagesElement = document.getElementById('messages');
                const emptyChatElement = document.getElementById('emptyChatMessage');
                
                if (messagesElement) {
                    messagesElement.innerHTML = '';
                }
                if (emptyChatElement) {
                    emptyChatElement.style.display = 'block';
                }
                messageCount = 0;
            }
            
            // Show success message
            function showSuccessMessage(message) {
                const emptyChatElement = document.getElementById('emptyChatMessage');
                if (emptyChatElement) {
                    emptyChatElement.style.display = 'none';
                }
                addMessage('system', `✅ ${message}`, null, null, null, 'success');
            }
            
            // Show error message
            function showErrorMessage(message) {
                addMessage('system', `❌ ${message}`, null, null, null, 'error');
            }
            
            // Send message
            async function sendMessage() {
                const input = document.getElementById('chatInput');
                const query = input.value.trim();
                if (!query) return;
                
                let endpoint, requestBody;
                
                if (currentMode === 'csv') {
                    if (!sessionId) {
                        showErrorMessage('Please upload a CSV file first.');
                        return;
                    }
                    endpoint = '/chat';
                    requestBody = {
                        session_id: sessionId,
                        query: query
                    };
                } else if (currentMode === 'database') {
                    const databaseUri = document.getElementById('databaseUri').value.trim();
                    if (!databaseUri) {
                        showErrorMessage('Please enter a database URI first.');
                        return;
                    }
                    endpoint = '/database-chat';
                    requestBody = {
                        database_uri: databaseUri,
                        query: query
                    };
                } else if (currentMode === 'research') {
                    if (uploadedFiles.length === 0) {
                        showErrorMessage('Please upload documents first.');
                        return;
                    }
                    endpoint = '/research-chat';
                    requestBody = {
                        session_id: researchSessionId,
                        query: query
                    };
                } else if (currentMode === 'google-drive') {
                    if (!googleDriveSessionId) {
                        showErrorMessage('Please load Google Drive folder first.');
                        return;
                    }
                    endpoint = '/google-drive-chat';
                    requestBody = {
                        session_id: googleDriveSessionId,
                        query: query
                    };
                }

                // Add user message
                const emptyChatElement = document.getElementById('emptyChatMessage');
                if (emptyChatElement) {
                    emptyChatElement.style.display = 'none';
                }
                addMessage('user', query);
                input.value = '';
                
                try {
                    showLoading();
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(requestBody)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        addMessage('assistant', result.response, result.plot_url, result.thinking, result.code);
                    } else {
                        showErrorMessage(result.detail || 'Failed to process query');
                    }
                } catch (error) {
                    showErrorMessage('Network error: ' + error.message);
                } finally {
                    hideLoading();
                }
            }
            
            // Add message to chat
            function addMessage(role, content, plotUrl = null, thinking = null, code = null, type = 'normal') {
                const messagesDiv = document.getElementById('messages');
                if (!messagesDiv) {
                    console.warn('Messages container not found');
                    return;
                }
                
                const messageDiv = document.createElement('div');
                messageDiv.className = `message`;
                messageCount++;
                
                let messageClass = '';
                let roleDisplay = '';
                let iconClass = '';
                
                switch (role) {
                    case 'user':
                        messageClass = 'user-message';
                        roleDisplay = 'You';
                        iconClass = 'fas fa-user';
                        break;
                    case 'assistant':
                        messageClass = 'assistant-message';
                        roleDisplay = 'AI Assistant';
                        iconClass = 'fas fa-robot';
                        break;
                    case 'system':
                        messageClass = type === 'success' ? 'alert alert-success' : type === 'error' ? 'alert alert-danger' : 'alert alert-info';
                        roleDisplay = 'System';
                        iconClass = 'fas fa-info-circle';
                        break;
                }
                
                messageDiv.className += ` ${messageClass}`;
                
                let html = '';
                if (role !== 'system') {
                    html = `
                        <div class="d-flex align-items-center mb-2">
                            <i class="${iconClass} me-2"></i>
                            <strong>${roleDisplay}</strong>
                            <small class="text-muted ms-auto">${new Date().toLocaleTimeString()}</small>
                        </div>
                    `;
                }
                
                // Process content with markdown for assistant messages
                if (role === 'assistant') {
                    html += marked.parse(content);
                } else {
                    html += content;
                }
                
                if (thinking) {
                    html += `
                        <div class="mt-3">
                            <details class="thinking-box">
                                <summary><i class="fas fa-brain me-2"></i>Model Thinking Process</summary>
                                <div class="p-3">
                                    <pre class="small mb-0" style="white-space: pre-wrap;">${thinking}</pre>
                                </div>
                            </details>
                        </div>
                    `;
                }
                
                if (plotUrl) {
                    html += `
                        <div class="mt-3 text-center">
                            <img src="${plotUrl}" class="img-fluid plot-image" alt="Generated Plot" style="max-width: 100%; height: auto;">
                        </div>
                    `;
                }
                
                if (code) {
                    html += `
                        <div class="mt-3">
                            <details class="code-box">
                                <summary><i class="fas fa-code me-2"></i>View Generated Code</summary>
                                <div class="p-3">
                                    <pre class="small mb-0"><code class="language-python">${code}</code></pre>
                                </div>
                            </details>
                        </div>
                    `;
                }
                
                messageDiv.innerHTML = html;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                
                // Update clear button visibility
                if (messageCount > 0) {
                    const clearChatBtn = document.getElementById('clearChat');
                    if (clearChatBtn) {
                        clearChatBtn.style.display = 'inline-block';
                    }
                }
            }
            
            // Show loading
            function showLoading() {
                const loadingElement = document.getElementById('loading');
                const sendButton = document.getElementById('sendButton');
                const chatInput = document.getElementById('chatInput');
                
                if (loadingElement) loadingElement.style.display = 'block';
                if (sendButton) sendButton.disabled = true;
                if (chatInput) chatInput.disabled = true;
            }
            
            // Hide loading
            function hideLoading() {
                const loadingElement = document.getElementById('loading');
                const chatInput = document.getElementById('chatInput');
                
                if (loadingElement) loadingElement.style.display = 'none';
                updateChatState(); // Re-enable based on current mode
                if (chatInput) chatInput.focus();
            }
            
            // Clear chat handler
            const clearChatButton = document.getElementById('clearChat');
            if (clearChatButton) {
                clearChatButton.addEventListener('click', function() {
                    if (confirm('Are you sure you want to clear the chat history?')) {
                        clearMessages();
                        clearChatButton.style.display = 'none';
                    }
                });
            }
            
            // Initialize everything when DOM is ready
            function initializeApp() {
                // Event listeners
                const sendButton = document.getElementById('sendButton');
                const chatInput = document.getElementById('chatInput');
                
                if (sendButton) {
                    sendButton.addEventListener('click', sendMessage);
                }
                
                if (chatInput) {
                    chatInput.addEventListener('keypress', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            sendMessage();
                        }
                    });
                }
                
                // Initialize drag and drop functionality
                initializeDragDrop();
                initializeResearchUpload();
                updateChatState();
            }
            
            // Initialize when DOM is ready
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', initializeApp);
            } else {
                initializeApp();
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form(...)):
    """Handle CSV file upload and return dataset information."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate dataframe
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded CSV file is empty")
        
        # Store dataframe in session
        if session_id not in sessions:
            sessions[session_id] = {}
        sessions[session_id]['df'] = df
        sessions[session_id]['messages'] = []
        sessions[session_id]['filename'] = file.filename
        
        # Generate insights
        insights = DataInsightAgent(df)
        
        # Prepare response with better preview
        preview_data = df.head(10).fillna('').to_dict('records')
        
        return DatasetInfo(
            columns=df.columns.tolist(),
            rows=len(df),
            insights=insights,
            preview=preview_data
        )
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Unable to decode file. Please ensure it's a valid CSV file with UTF-8 encoding.")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The CSV file appears to be empty or invalid.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/database-chat", response_model=ChatResponse)
async def database_chat(request: DatabaseQueryRequest):
    """Handle database chat queries and return responses."""
    try:
        # Process the database query using the DatabaseChatAgent
        response = DatabaseChatAgent(request.database_uri, request.query)
        
        return ChatResponse(
            response=response,
            plot_url=None,
            thinking="Processed query using SQL Database Agent with Gemini LLM.",
            code=None
        )
        
    except Exception as e:
        error_msg = f"Database query failed: {str(e)}"
        return ChatResponse(
            response=error_msg,
            plot_url=None,
            thinking=f"Error occurred while processing database query: {str(e)}",
            code=None
        )

@app.post("/research-upload")
async def research_upload(file: UploadFile = File(...), session_id: str = Form(...)):
    """Handle research document upload using UnstructuredFileLoader."""
    try:
        # Validate file type
        valid_extensions = ['.pdf', '.txt', '.docx', '.xlsx', '.pptx']
        file_extension = '.' + file.filename.split('.')[-1].lower()
        
        if file_extension not in valid_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported: {', '.join(valid_extensions)}")
        
        # Create session if it doesn't exist
        if session_id not in sessions:
            sessions[session_id] = {}
        
        session = sessions[session_id]
        if 'research_docs' not in session:
            session['research_docs'] = []
        
        # Save file temporarily
        file_content = await file.read()
        temp_file_path = os.path.join(tempfile.gettempdir(), f"{session_id}_{file.filename}")
        
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file_content)
        
        # Load document using UnstructuredFileLoader
        try:
            loader = UnstructuredFileLoader(temp_file_path)
            documents = loader.load()
            
            # Extract text content
            doc_content = "\n".join([doc.page_content for doc in documents if doc.page_content.strip()])
            
            if doc_content.strip():
                session['research_docs'].append(f"File: {file.filename}\nContent: {doc_content}")
                
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                
                return JSONResponse({
                    "status": "success",
                    "filename": file.filename,
                    "content_length": len(doc_content),
                    "message": f"Successfully processed {file.filename}"
                })
            else:
                raise HTTPException(status_code=400, detail=f"No readable content found in {file.filename}")
                
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/research-chat", response_model=ChatResponse)
async def research_chat(request: ResearchQueryRequest):
    """Handle research chat queries using agentic AI workflow."""
    try:
        # Process query using agentic workflow
        response = process_research_query(request.session_id, request.query)
        return response

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Research chat failed: {str(e)}"
        return ChatResponse(
            response=error_msg,
            plot_url=None,
            thinking=f"Error occurred while processing research query: {str(e)}",
            code=None
        )

@app.post("/google-drive-load")
async def google_drive_load(request: GoogleDriveLoadRequest):
    """Load documents from Google Drive folder."""
    try:
        # Extract folder ID from URL
        folder_id = extract_google_drive_folder_id(request.drive_url)

        # Create session if it doesn't exist
        if request.session_id not in sessions:
            sessions[request.session_id] = {}

        session = sessions[request.session_id]
        if 'google_drive_docs' not in session:
            session['google_drive_docs'] = []

        # Path to client secret JSON
        client_secret_path = r"d:\Langsmith-main\data_analyst_agent\GenerativeAIExamples\community\data-analysis-agent\client_secret_5752229595-3ohesd691r9q4td6cst1fqqq2c3q59pf.apps.googleusercontent.com.json"

        # Path for token (will be created in same directory as client secret)
        token_path = os.path.join(
            os.path.dirname(client_secret_path),
            f"google_token_{request.session_id}.pkl"
        )

        # Load documents using custom function
        documents = load_google_drive_documents(
            folder_id=folder_id,
            credentials_path=client_secret_path,
            token_path=token_path,
            recursive=request.recursive
        )

        if not documents:
            raise HTTPException(status_code=400, detail="No documents found in the specified Google Drive folder")

        # Extract and store document content
        loaded_files = []
        for doc in documents:
            if doc['content'].strip():
                filename = doc['name']
                session['google_drive_docs'].append(f"File: {filename}\nContent: {doc['content']}")
                loaded_files.append(filename)

        if not loaded_files:
            raise HTTPException(status_code=400, detail="No readable content found in Google Drive documents")

        return JSONResponse({
            "status": "success",
            "folder_id": folder_id,
            "files_loaded": len(loaded_files),
            "files": loaded_files,
            "message": f"Successfully loaded {len(loaded_files)} documents from Google Drive"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Drive load failed: {str(e)}")

@app.post("/dashboard-generate")
async def generate_dashboard(session_id: str = Form(...)):
    """Generate an automatic interactive dashboard from uploaded CSV data."""
    try:
        # Check session and get DataFrame
        if session_id not in sessions:
            raise HTTPException(status_code=400, detail="Session not found. Please upload a CSV file first.")

        session = sessions[session_id]
        df = session.get('df')

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No valid data found in session. Please upload a CSV file.")

        # Generate dashboard HTML using AutomaticDashboardAgent
        dashboard_html = AutomaticDashboardAgent(df)

        # Save dashboard HTML to static folder
        dashboard_id = str(uuid.uuid4())
        filename = f"dashboard_{session_id}_{dashboard_id}.html"
        filepath = os.path.join("static", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)

        dashboard_url = f"/static/{filename}"

        return JSONResponse({
            "success": True,
            "dashboard_url": dashboard_url,
            "message": "Dashboard generated successfully!"
        })

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"Failed to generate dashboard: {str(e)}"
        }, status_code=500)

@app.post("/google-drive-chat", response_model=ChatResponse)
async def google_drive_chat(request: GoogleDriveQueryRequest):
    """Handle chat queries against Google Drive documents using agentic AI workflow."""
    try:
        # Get session data
        if request.session_id not in sessions:
            raise HTTPException(status_code=400, detail="Session not found. Please load Google Drive documents first.")

        session = sessions[request.session_id]
        google_drive_docs = session.get('google_drive_docs', [])

        if not google_drive_docs:
            raise HTTPException(status_code=400, detail="No Google Drive documents found in session. Please load documents first.")

        # Build workflow (reusing the research workflow)
        workflow = build_research_workflow()

        # Prepare initial state
        initial_state = {
            "question": request.query,
            "original_question": request.query,
            "docs": google_drive_docs,
            "external_docs": None,
            "answer": None,
            "relevant": None,
            "answered": None,
            "selected_tools": None,
            "search_strategy": None,
            "iteration_count": 0,
            "reasoning": None,
            "visualization_needed": None,
            "chart_data": None,
            "plot_url": None
        }

        # Execute workflow with increased recursion limit
        result = workflow.invoke(initial_state, config={"recursion_limit": 50})

        # Store in session history
        if 'google_drive_messages' not in session:
            session['google_drive_messages'] = []

        session['google_drive_messages'].append({
            'query': request.query,
            'response': result.get("answer", "No answer generated"),
            'plot_url': result.get("plot_url"),
            'reasoning': result.get("reasoning", ""),
            'selected_tools': result.get("selected_tools", []),
            'timestamp': pd.Timestamp.now().isoformat()
        })

        return ChatResponse(
            response=result.get("answer", "No answer generated"),
            plot_url=result.get("plot_url"),
            thinking=f"Google Drive analysis completed. Tools used: {', '.join(result.get('selected_tools', []))}. Reasoning: {result.get('reasoning', '')}",
            code=None
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Google Drive chat failed: {str(e)}"
        return ChatResponse(
            response=error_msg,
            plot_url=None,
            thinking=f"Error occurred during Google Drive chat: {str(e)}",
            code=None
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    """Handle chat queries and return responses."""
    import os
    import google.generativeai as genai
    from PIL import Image
    import tempfile

    user_query_lower = request.query.lower()

    # Check for PDF generation command first
    if "pdf" in user_query_lower:
        return await generate_pdf_report(request.session_id)

    # Define keywords that trigger Gemini-based insight generation
    insight_keywords = ["actionable insight", "conclusion", "insight", "insights", "analysis", "analyze", "summary", "summarize"]
    
    # Check if user query contains any of the insight keywords (case-insensitive)
    should_use_gemini = any(keyword in user_query_lower for keyword in insight_keywords)

    if should_use_gemini:
        # Check session and get DataFrame
        if request.session_id not in sessions:
            raise HTTPException(status_code=400, detail="Session not found. Please upload a CSV file first.")
        
        session = sessions[request.session_id]
        df = session['df']
        
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No valid data found in session. Please upload a CSV file.")

        llm_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="AIzaSyCowWFDIENMNDTBtGm5HkvWzeXG9SpIboI")

        # Convert DataFrame to Excel file temporarily for UnstructuredFileLoader
        import tempfile
        import os
        
        temp_excel_path = os.path.join(tempfile.gettempdir(), f"temp_data_{request.session_id}.xlsx")
        df.to_excel(temp_excel_path, index=False)
        
        # Load the Excel file using UnstructuredFileLoader
        loader = UnstructuredFileLoader(temp_excel_path)
        docs = loader.load()

        if not docs:
            # Clean up temp file
            if os.path.exists(temp_excel_path):
                os.remove(temp_excel_path)
            return ChatResponse(
                response="No data available to generate insights.",
                plot_url=None,
                thinking="No documents were loaded for insight generation.",
                code=None
            )

        # Split documents into chunks - necessary for both StuffDocumentsChain and vector stores
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000000, chunk_overlap=5000
        )
        chunks = text_splitter.split_documents(docs)

        # Process each chunk separately with LLM to generate insights (retaining original prompt logic)
        summaries = []

        for chunk in chunks:
            llm_chain = LLMChain(
                llm=llm_model,
                prompt = PromptTemplate.from_template("""
                    You are a highly skilled data analyst and interpreter. Below is the provided content:

                    {text}

                    Your task is to answer the user's question in a **detailed, structured, and insight-driven** manner using only the information from the provided content.

                    ## Instructions:

                    ### 1. Understand the Content
                    - Carefully read and comprehend all numerical and contextual information.
                    - Identify key entities, variables, and their relationships.

                    ### 2. If the question involves comparisons (e.g., highest, lowest, most, least):
                    - Extract all relevant numerical values.
                    - Compare and rank them clearly.
                    - Explain what these comparisons reveal and why they matter in context.

                    ### 3. If the question asks for totals or aggregates (e.g., total sales, average scores):
                    - Calculate and clearly report sums, averages, or other relevant metrics.
                    - If needed, include intermediate breakdowns to support clarity.

                    ### 4. If the question requests a summary:
                    - Provide a concise, well-organized overview of the most important findings.
                    - Use bullet points if appropriate.
                    - Highlight information that has strategic relevance.

                    ### 5. If the question asks for an analysis:
                    - Go beyond summarizing — interpret and explain the *significance* of the findings.
                    - Identify patterns, trends, correlations, and anomalies.
                    - Discuss the possible causes, implications, and what it suggests about user behavior, performance, or outcomes.

                    ### 6. Most Important — Provide Actionable Insights:
                    - Translate your findings into **clear, practical recommendations**.
                    - Each key insight should be paired with a **strategic suggestion** (e.g., improve feature X to address Y, prioritize product Z for scaling).
                    - Focus on insights that can inform decision-making in marketing, product, operations, or strategy.

                    ### Output Guidelines:
                    - Justify all conclusions with specific data references (e.g., "Product A had 1,200 units sold, which is the highest among all listed.").
                    - Use markdown-style formatting (e.g., headings, bullet points) to enhance readability.
                    - Do **not** invent or assume information that is not in the provided content.
                    - Be thoughtful, clear, and actionable — not just descriptive.

                    User Question: **{user_question}**
                    """),
            )
            stuff_chain = StuffDocumentsChain(
                llm_chain=llm_chain, document_variable_name="text"
            )
            response = stuff_chain.invoke({"input_documents": [chunk], "user_question": request.query})
            summaries.append(response["output_text"])

        # Combine all chunk summaries for the main insights
        insights = "\n\n".join(summaries)
        
        # Clean up temporary Excel file
        if os.path.exists(temp_excel_path):
            os.remove(temp_excel_path)

        # Create FAISS vector store and retriever for chat functionality
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedder)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # Save the FAISS index
        faiss_index_path = os.path.join(tempfile.gettempdir(), f"faiss_index_{request.session_id}")
        vectorstore.save_local(faiss_index_path)
        
        # Store FAISS index path in session for future use
        session['faiss_index_path'] = faiss_index_path

        # Store message in history (GEMINI RESPONSE)
        session['messages'].append({
            'query': request.query,
            'response': insights,  # This is the full Gemini response
            'plot_url': None,
            'code': None,
            'thinking': "Generated insights using Gemini LLM based on document analysis.",
            'timestamp': pd.Timestamp.now().isoformat()
        })

        return ChatResponse(
            response=insights,
            plot_url=None,
            thinking="Generated insights using Gemini LLM based on document analysis.",
            code=None
        )
    
    else:
        if request.session_id not in sessions:
            raise HTTPException(status_code=400, detail="Session not found. Please upload a CSV file first.")
        
        session = sessions[request.session_id]
        df = session['df']
        
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No valid data found in session. Please upload a CSV file.")
        
        try:
            # Generate and execute code
            code, should_plot, _ = CodeGenerationAgent(request.query, df)
            
            if not code:
                raise HTTPException(status_code=500, detail="Failed to generate code for your query. Please try rephrasing your question.")
            
            result = ExecutionAgent(code, df, should_plot)
            
            # Check for execution errors
            if isinstance(result, str) and result.startswith("Error executing code"):
                # Try to provide a helpful error message
                error_msg = f"I encountered an issue while processing your query: {result}"
                return ChatResponse(
                    response=error_msg,
                    plot_url=None,
                    thinking="The generated code failed to execute properly.",
                    code=code
                )
            
            # Generate reasoning
            thinking, reasoning = ReasoningAgent(request.query, result)
            
            # Handle plot saving and analysis
            plot_url = None
            enhanced_reasoning = reasoning  # Default to original reasoning
            
            if isinstance(result, (plt.Figure, plt.Axes)):
                fig = result.figure if isinstance(result, plt.Axes) else result
                plot_url = save_plot_to_static(fig, request.session_id)
                
                # Add visualization analysis with Google Gemini
                if plot_url:
                    try:
                        # Get the full path to the saved plot
                        plot_path = os.path.join("static", plot_url.split("/static/")[-1])
                        
                        if os.path.exists(plot_path):
                            # Open the image for Gemini analysis
                            img_plot = Image.open(plot_path)
                            
                            # Create analysis prompt
                            analysis_prompt = """
                            As a data analyst, provide a comprehensive analysis of this visualization. Your analysis should include:

                            - A detailed examination of the patterns, trends, and distributions shown in the plot
                            - Identification of key insights, outliers, correlations, or notable features
                            - Interpretation of what these patterns suggest about the underlying data
                            - Discussion of any significant relationships or anomalies visible in the visualization
                            - Strategic insights and actionable recommendations based on the visual findings
                            - Context about what these results might mean for decision-making

                            Focus on providing deep, meaningful insights that go beyond surface-level observations. 
                            Transform this visualization into clear, data-informed conclusions and recommendations.

                            Avoid mentioning the specific tools or platforms used for this analysis.
                            """
                            
                            # Generate enhanced analysis using Gemini
                            genai.configure(api_key="AIzaSyCowWFDIENMNDTBtGm5HkvWzeXG9SpIboI")
                            model = genai.GenerativeModel("gemini-2.0-flash")
                            gemini_response = model.generate_content(
                                [analysis_prompt, img_plot],
                                generation_config={"temperature": 0},
                            ).text
                            
                            # Combine original reasoning with Gemini analysis
                            enhanced_reasoning = f"{reasoning}\n\n## Visualization Analysis\n\n{gemini_response}"
                            
                    except Exception as e:
                        print(f"Error in Gemini analysis: {str(e)}")
                        # Continue with original reasoning if Gemini analysis fails
                        pass
            
            # Store message in history (NVIDIA LLAMA RESPONSE + Enhanced Analysis)
            session['messages'].append({
                'query': request.query,
                'response': enhanced_reasoning,  # This now includes both original reasoning and Gemini analysis
                'plot_url': plot_url,   # This is the plot URL
                'code': code,
                'thinking': thinking,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            return ChatResponse(
                response=enhanced_reasoning,
                plot_url=plot_url,
                thinking=thinking,
                code=code
            )
            
        except Exception as e:
            error_msg = f"I encountered an unexpected error while processing your query: {str(e)}"
            return ChatResponse(
                response=error_msg,
                plot_url=None,
                thinking=f"Error occurred: {str(e)}",
                code=code if 'code' in locals() else None
            )

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Data Analysis Chat Report', 0, 1, 'C')
        self.set_font('Arial', '', 8)
        self.cell(0, 5, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def add_user_message(self, text):
        self.set_font('Arial', 'B', 11)
        self.set_fill_color(220, 220, 220)
        text = f"You: {text}".encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 8, text, 1, 'L', 1)
        self.ln(2)
    
    def add_ai_message(self, text, code=None, plot_url=None):
        self.set_font('Arial', '', 11)
        self.set_fill_color(240, 240, 240)
        text = f"AI Assistant: {text}".encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 8, text, 1, 'L', 1)

        if code:
            self.ln(2)
            self.set_font('Courier', '', 10)
            self.set_fill_color(250, 250, 250)
            code_text = code.encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 5, code_text, border=1, align='L', fill=1)
        
        if plot_url:
            self.ln(2)
            plot_path = os.path.join(os.getcwd(), plot_url.lstrip('/\\'))
            if os.path.exists(plot_path):
                page_width = self.w - 2 * self.l_margin
                self.image(plot_path, w=page_width * 0.8)
            else:
                self.set_font('Arial', 'I', 10)
                self.set_text_color(255, 0, 0)
                self.cell(0, 10, f"[Plot image not found: {plot_path}]")
                self.set_text_color(0, 0, 0)
        
        self.ln(5)

async def generate_pdf_report(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not found. Please upload a CSV file first.")
        
    session = sessions.get(session_id, {})
    messages = session.get('messages', [])
        
    if not messages:
        return ChatResponse(
            response="There is no conversation history to generate a PDF from.",
            plot_url=None,
        )

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
        
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f"Chat Report for Session: {session_id}", 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Dataset: {session.get('filename', 'N/A')}", 0, 1, 'C')
    pdf.ln(5)

    for message in messages:
        # Add user query first
        if message.get('query'):
            pdf.add_user_message(message['query'])
        
        # Then add AI Assistant response
        # Check if message has both response text and visualization
        if message.get('response') and message.get('plot_url'):
            # This is a visualization response with description (NVIDIA LLAMA)
            pdf.add_ai_message(message['response'], code=None, plot_url=message['plot_url'])
        
        # Check if message has only text response (GEMINI)
        elif message.get('response'):
            # This is a text-only response from Gemini
            pdf.add_ai_message(message['response'], code=None, plot_url=None)
        
        # Check if message has only visualization (unlikely, but just in case)
        elif message.get('plot_url'):
            # This is a visualization without description
            pdf.add_ai_message("[Visualization]", code=None, plot_url=message['plot_url'])

    pdf_file_name = f"report_{session_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf_path = os.path.join("static", pdf_file_name)
        
    try:
        pdf.output(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {e}")
        
    pdf_url = f"/static/{pdf_file_name}"
        
    response_text = (
        f'I have generated a PDF report of our conversation. '
        f'<a href="{pdf_url}" class="btn btn-success" target="_blank" style="margin-left:10px;">'
        f'<i class="fas fa-file-pdf"></i> Download PDF Report'
        f'</a>'
    )
        
    return ChatResponse(
        response=response_text,
        plot_url=None,
        thinking=f"Generated a PDF report with {len(messages)} message pairs.",
        code=None
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        timeout_keep_alive=600,
        log_level="info",
        access_log=True,
    )
