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
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
import google.generativeai.types as gtypes
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fpdf import FPDF
from datetime import datetime
import os
from PIL import Image

# New imports for database functionality
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType

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
            google_api_key="AIzaSyAMAYxkjP49QZRCg21zImWWAu7c3YHJ0a8"
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
        response = agent_executor.run(query)
        return response
        
    except Exception as e:
        return f"Error connecting to database or executing query: {str(e)}"

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
        <title>Data Analysis Agent</title>
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
                        <h3 class="brand-title"><i class="fas fa-chart-line"></i> Data Analysis Agent</h3>
                        <p class="small mb-4">Powered by <a href="https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1" target="_blank" class="nvidia-link">NVIDIA Llama-3.1-Nemotron-Ultra-253B-v1</a></p>
                        
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
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            let sessionId = null;
            let messageCount = 0;
            let currentMode = 'csv'; // 'csv' or 'database'
            
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
                        } else if (targetId === '#database-panel') {
                            currentMode = 'database';
                            document.getElementById('chatTitle').textContent = 'Chat with your database';
                            updateChatState();
                        }
                    });
                });
            });
            
            // Update chat input state based on current mode
            function updateChatState() {
                const chatInput = document.getElementById('chatInput');
                const sendButton = document.getElementById('sendButton');
                
                if (currentMode === 'csv') {
                    const hasSession = sessionId && sessions && sessions[sessionId];
                    chatInput.disabled = !hasSession;
                    sendButton.disabled = !hasSession;
                    chatInput.placeholder = hasSession ? 
                        "Ask me anything about your data..." : 
                        "Upload a CSV file first...";
                } else if (currentMode === 'database') {
                    const hasUri = document.getElementById('databaseUri').value.trim();
                    chatInput.disabled = !hasUri;
                    sendButton.disabled = !hasUri;
                    chatInput.placeholder = hasUri ? 
                        "Ask me anything about your database..." : 
                        "Enter database URI first...";
                }
            }
            
            // Database URI input handler
            document.getElementById('databaseUri').addEventListener('input', function() {
                if (currentMode === 'database') {
                    updateChatState();
                }
            });
            
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
                document.getElementById('uploadPrompt').style.display = 'none';
                document.getElementById('datasetInfo').style.display = 'block';
                
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
                document.getElementById('messages').innerHTML = '';
                document.getElementById('emptyChatMessage').style.display = 'block';
                messageCount = 0;
            }
            
            // Show success message
            function showSuccessMessage(message) {
                document.getElementById('emptyChatMessage').style.display = 'none';
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
                }
                
                // Add user message
                document.getElementById('emptyChatMessage').style.display = 'none';
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
                    document.getElementById('clearChat').style.display = 'inline-block';
                }
            }
            
            // Show loading
            function showLoading() {
                document.getElementById('loading').style.display = 'block';
                document.getElementById('sendButton').disabled = true;
                document.getElementById('chatInput').disabled = true;
            }
            
            // Hide loading
            function hideLoading() {
                document.getElementById('loading').style.display = 'none';
                updateChatState(); // Re-enable based on current mode
                document.getElementById('chatInput').focus();
            }
            
            // Clear chat handler
            document.getElementById('clearChat').addEventListener('click', function() {
                if (confirm('Are you sure you want to clear the chat history?')) {
                    clearMessages();
                    document.getElementById('clearChat').style.display = 'none';
                }
            });
            
            // Event listeners
            document.getElementById('sendButton').addEventListener('click', sendMessage);
            document.getElementById('chatInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            
            // Initialize
            initializeDragDrop();
            updateChatState();
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

        llm_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="AIzaSyAMAYxkjP49QZRCg21zImWWAu7c3YHJ0a8")

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
        embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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
                            genai.configure(api_key="AIzaSyAMAYxkjP49QZRCg21zImWWAu7c3YHJ0a8")
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
