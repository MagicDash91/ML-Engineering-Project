"""
Super Agentic RAG System
Combines Agentic RAG with LangChain, LangGraph, and NVIDIA LLAMA
Supports intelligent routing based on file type and query context
"""

import os
import io
import re
import base64
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import WikipediaQueryRun, ArxivQueryRun, Tool
from langchain.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph imports
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# ===========================
# Configuration
# ===========================

# LangSmith tracing
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# NVIDIA API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
) if NVIDIA_API_KEY else None

# Google API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Initialize LLMs
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    max_retries=2,  # Add retry logic for rate limits
) if GOOGLE_API_KEY else None

# Rate limiting configuration for Gemini (5 requests per minute free tier)
GEMINI_CALL_DELAY = 13  # Wait 13 seconds between calls (5 calls/60s = 12s, add buffer)

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Web Search Tool
web_search_tool = TavilySearchResults(k=3) if TAVILY_API_KEY else None

# Initialize Research Tools
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())

research_tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia_tool.run,
        description="Use for general concepts and historical information"
    ),
    Tool(
        name="arXiv",
        func=arxiv_tool.run,
        description="Use for academic research and scientific studies"
    ),
    Tool(
        name="TavilySearch",
        func=web_search_tool.run if web_search_tool else lambda x: "Web search not available",
        description="Use for current events, updates, and real-time information"
    ),
]

# ===========================
# State Definitions
# ===========================

class AgentState(TypedDict):
    """State for Agentic RAG workflow"""
    question: str
    original_question: str
    file_type: str
    file_path: str
    docs: Optional[List[str]]
    external_docs: Optional[List[str]]
    answer: Optional[str]
    relevant: Optional[bool]
    answered: Optional[bool]
    selected_tools: Optional[List[str]]
    search_strategy: Optional[str]
    iteration_count: Optional[int]
    reasoning: Optional[str]
    data_type: Optional[str]  # "informational" or "tabular"
    visualization_needed: Optional[bool]
    plot_url: Optional[str]
    code: Optional[str]
    thinking: Optional[str]

# ===========================
# Hybrid Retriever Class
# ===========================

class HybridRetriever:
    """
    Hybrid retriever combining dense (FAISS) and sparse (BM25) search
    for optimal document retrieval
    """

    def __init__(self, dense_weight: float = 0.5, sparse_weight: float = 0.5):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for better precision
            chunk_overlap=150,
            length_function=len,
        )
        self.vectorstore = None
        self.bm25 = None
        self.documents = []
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 with preprocessing"""
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split and remove empty tokens
        tokens = [token for token in text.split() if token]
        return tokens

    def _expand_query_keywords(self, tokens: List[str]) -> List[str]:
        """Expand query keywords with synonyms for better matching"""
        # Synonym mapping for common CV/resume queries
        synonyms = {
            'teaching': ['teach', 'teacher', 'trainer', 'tutor', 'instructor', 'lecture', 'lecturer', 'training', 'educate', 'education'],
            'experience': ['work', 'role', 'position', 'job', 'employment', 'responsibility', 'responsibilities'],
            'education': ['study', 'degree', 'qualification', 'certification', 'school', 'university', 'college', 'bootcamp'],
            'skill': ['skills', 'expertise', 'competency', 'competencies', 'ability', 'abilities', 'proficiency'],
            'project': ['projects', 'work', 'development', 'implementation'],
        }

        expanded = []
        for token in tokens:
            expanded.append(token)
            # Add synonyms if the token matches
            for key, syn_list in synonyms.items():
                if token in key or key in token:
                    expanded.extend(syn_list)

        return list(set(expanded))  # Remove duplicates

    def process_documents(self, documents: List) -> str:
        """Process documents and create both vector store and BM25 index"""
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        self.documents = splits

        # Create dense vector store (FAISS)
        self.vectorstore = FAISS.from_documents(splits, embeddings)

        # Create sparse BM25 index with proper tokenization
        tokenized_docs = [self._tokenize(doc.page_content) for doc in splits]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Combine all text for context
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        return combined_text

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores or max(scores) == min(scores):
            return [0.0] * len(scores)

        min_score = min(scores)
        max_score = max(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def hybrid_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Perform hybrid search combining dense and sparse retrieval"""
        if not self.vectorstore or not self.bm25:
            return []

        # Dense search (FAISS)
        dense_results = self.vectorstore.similarity_search_with_score(
            query, k=len(self.documents)
        )

        # Convert FAISS distances to similarities
        dense_similarities = [1 / (1 + score) for _, score in dense_results]
        dense_similarities = self._normalize_scores(dense_similarities)

        # Sparse search (BM25) with proper tokenization and keyword expansion
        query_tokens = self._tokenize(query)
        expanded_tokens = self._expand_query_keywords(query_tokens)
        bm25_scores = self.bm25.get_scores(expanded_tokens)
        sparse_similarities = self._normalize_scores(bm25_scores.tolist())

        # Combine scores using weighted average
        hybrid_scores = []
        for i in range(len(self.documents)):
            dense_sim = dense_similarities[i] if i < len(dense_similarities) else 0
            sparse_sim = sparse_similarities[i] if i < len(sparse_similarities) else 0
            hybrid_score = (self.dense_weight * dense_sim) + (self.sparse_weight * sparse_sim)
            hybrid_scores.append((self.documents[i].page_content, hybrid_score))

        # Sort by hybrid score and return top k
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:k]

# ===========================
# File Processing Functions
# ===========================

def detect_file_type(file_path: str) -> str:
    """Detect file type based on extension"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return 'pdf'
    elif ext in ['.doc', '.docx']:
        return 'word'
    elif ext in ['.ppt', '.pptx']:
        return 'powerpoint'
    elif ext == '.txt':
        return 'text'
    elif ext == '.csv':
        return 'csv'
    elif ext in ['.xls', '.xlsx']:
        return 'excel'
    else:
        return 'unknown'

def load_document(file_path: str) -> List:
    """Load document based on file type"""
    file_type = detect_file_type(file_path)

    if file_type == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_type == 'word':
        loader = Docx2txtLoader(file_path)
    elif file_type == 'powerpoint':
        loader = UnstructuredPowerPointLoader(file_path)
    elif file_type == 'text':
        loader = TextLoader(file_path)
    elif file_type == 'csv':
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return loader.load()

def classify_data_type(df: pd.DataFrame, question: str) -> str:
    """
    Classify if the dataset is informational (Q&A/Glossary) or tabular data
    """
    if not gemini_llm:
        # Fallback to simple heuristic
        if df.shape[1] <= 3 and any(col.lower() in ['question', 'answer', 'term', 'definition', 'glossary']
                                      for col in df.columns):
            return "informational"
        return "tabular"

    prompt = f"""
    Analyze this dataset structure and question to determine the data type:

    Dataset Info:
    - Columns: {list(df.columns)}
    - Shape: {df.shape}
    - First few rows:
    {df.head(3).to_string()}

    Question: {question}

    Is this dataset:
    1. "informational" - Contains Q&A pairs, glossary terms, definitions, or simple lookup information
    2. "tabular" - Contains structured data for analysis, predictions, visualizations (like churn prediction,
       healthcare data, telecom data, sales data, etc.)

    Reply with only one word: "informational" or "tabular"
    """

    result = gemini_llm.invoke(prompt)
    data_type = result.content.strip().lower()

    # Add delay to prevent rate limiting
    time.sleep(GEMINI_CALL_DELAY)

    return "informational" if "informational" in data_type else "tabular"

# ===========================
# NVIDIA LLAMA Functions
# ===========================

def query_nvidia_llama(messages: List[Dict], temperature: float = 0.2,
                       max_tokens: int = 1024) -> str:
    """Query NVIDIA LLAMA model"""
    if not nvidia_client:
        raise ValueError("NVIDIA API client not initialized. Please set NVIDIA_API_KEY")

    response = nvidia_client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content

def check_visualization_intent(query: str) -> bool:
    """Check if query requests visualization using NVIDIA LLAMA"""
    if not nvidia_client:
        return False

    messages = [
        {
            "role": "system",
            "content": "You are an assistant that determines if a query is requesting a data visualization. Respond with only 'true' if the query is asking for a plot, chart, graph, or any visual representation of data. Otherwise, respond with 'false'."
        },
        {"role": "user", "content": query}
    ]

    response = query_nvidia_llama(messages, temperature=0.1, max_tokens=5)
    return response.strip().lower() == "true"

def generate_analysis_code(df: pd.DataFrame, query: str,
                          should_plot: bool) -> str:
    """Generate Python code for data analysis using NVIDIA LLAMA"""
    if not nvidia_client:
        raise ValueError("NVIDIA API client not initialized")

    cols = list(df.columns)

    if should_plot:
        # PlotCodeGeneratorTool prompt - exact copy from data_analyst_agent
        prompt = f"""
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
    else:
        # CodeWritingTool prompt - exact copy from data_analyst_agent
        prompt = f"""
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

    messages = [
        {
            "role": "system",
            "content": "detailed thinking off. You are a Python data-analysis expert who writes clean, efficient code. Solve the given problem with optimal pandas operations. Be concise and focused. Your response must contain ONLY a properly-closed ```python code block with no explanations before or after. Ensure your solution is correct, handles edge cases, and follows best practices for data analysis."
        },
        {"role": "user", "content": prompt}
    ]

    response = query_nvidia_llama(messages, temperature=0.2, max_tokens=1024)

    # Extract code block
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0].strip()
    elif "```" in response:
        code = response.split("```")[1].split("```")[0].strip()
    else:
        code = response.strip()

    return code

def execute_code(code: str, df: pd.DataFrame) -> Tuple[Any, str, bool]:
    """Execute generated code safely"""
    import matplotlib
    matplotlib.use('Agg')

    local_vars = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}
    plot_created = False
    result = None

    try:
        exec(code, {"__builtins__": __builtins__}, local_vars)

        # Get result if available
        if 'result' in local_vars:
            result = local_vars['result']

        # Check if result is a matplotlib figure or axis
        if isinstance(result, (plt.Figure, plt.Axes)):
            plot_created = True

        return result, None, plot_created

    except Exception as e:
        return None, str(e), False

def generate_insight_with_gemini(df: pd.DataFrame, query: str, result: Any) -> str:
    """Generate insights using Gemini for non-visualization queries"""
    if not gemini_llm:
        return f"Result: {result}"

    # Create summary of the data
    summary = f"""
    Dataset shape: {df.shape}
    Columns: {list(df.columns)}

    Query: {query}
    Result: {result}
    """

    prompt = f"""
    You are a data analyst. Provide clear, concise insights based on this analysis.

    {summary}

    Provide 2-3 sentences explaining what this result tells us about the data.
    Focus on actionable insights and key findings.
    """

    response = gemini_llm.invoke(prompt)

    # Add delay to prevent rate limiting
    time.sleep(GEMINI_CALL_DELAY)

    return response.content.strip()

def generate_insight_with_nvidia_gemini(df: pd.DataFrame, query: str,
                                        result: Any, plot_created: bool) -> str:
    """Generate insights using NVIDIA LLAMA with thinking mode"""
    if not nvidia_client:
        return f"Result: {result}"

    if plot_created:
        prompt = f"""
        The user asked: "{query}".
        A visualization has been created showing the data.
        Explain in 2-3 concise sentences what insights can be drawn from this analysis.
        """
    else:
        prompt = f"""
        The user asked: "{query}".
        The result value is: {result}
        Explain in 2-3 concise sentences what this tells about the data.
        """

    messages = [
        {
            "role": "system",
            "content": "detailed thinking off. You are an insightful data analyst. Provide clear, concise insights without showing your internal thought process. Just give the final answer."
        },
        {"role": "user", "content": prompt}
    ]

    response = query_nvidia_llama(messages, temperature=0.2, max_tokens=1024)
    return response

# ===========================
# Agentic RAG Nodes
# ===========================

def tool_selection_node(state: AgentState) -> AgentState:
    """Agent decides which tools to use based on question analysis"""
    question = state["question"]

    if not gemini_llm:
        return {
            **state,
            "selected_tools": ["PDF_Documents"],
            "reasoning": "Using local documents only",
            "search_strategy": "document_only"
        }

    prompt = f"""
    Select the best tools for answering this question:

    Question: {question}

    Available tools:
    1. Wikipedia - For general concepts and historical information
    2. arXiv - For academic research and scientific studies
    3. TavilySearch - For current events and real-time information
    4. PDF_Documents - For specific document content (already loaded)

    Select 1-3 tools that would be most helpful. Return as comma-separated list.

    Format:
    TOOLS: tool1,tool2,tool3
    REASONING: why these tools were selected
    """

    result = gemini_llm.invoke(prompt)
    content = result.content.strip()

    # Add delay to prevent rate limiting (5 requests/minute = 1 request every 12s)
    time.sleep(GEMINI_CALL_DELAY)

    # Parse response
    lines = content.split('\n')
    selected_tools = ["PDF_Documents"]
    reasoning = ""

    for line in lines:
        if 'TOOLS:' in line or 'Tools:' in line:
            tools_str = re.split(r'TOOLS:|Tools:', line, flags=re.IGNORECASE)[-1]
            selected_tools = [tool.strip() for tool in tools_str.split(',') if tool.strip()]
        elif 'REASONING:' in line or 'Reasoning:' in line:
            reasoning = re.split(r'REASONING:|Reasoning:', line, flags=re.IGNORECASE)[-1].strip()

    return {
        **state,
        "selected_tools": selected_tools,
        "reasoning": reasoning,
        "search_strategy": "multi_tool_search"
    }

def hybrid_retrieve_node(state: AgentState) -> AgentState:
    """Retrieve using hybrid search from documents"""
    question = state["question"]
    file_path = state.get("file_path")
    iteration_count = state.get("iteration_count", 0)

    # Load and process documents
    documents = load_document(file_path)

    # Create hybrid retriever
    retriever = HybridRetriever()
    retriever.process_documents(documents)

    # Increase k on subsequent iterations to get more diverse results
    # First iteration: top 10, second: top 15, third: top 20 (all chunks)
    k = min(10 + (iteration_count * 5), len(retriever.documents))

    # Perform hybrid search
    hybrid_results = retriever.hybrid_search(question, k=k)
    internal_docs = [doc for doc, score in hybrid_results]

    return {
        **state,
        "docs": internal_docs
    }

def multi_source_retrieve_node(state: AgentState) -> AgentState:
    """Retrieve from multiple sources based on selected tools"""
    question = state["question"]
    selected_tools = state.get("selected_tools", [])
    internal_docs = state.get("docs", [])

    # External tool retrieval
    external_docs = []

    for tool_name in selected_tools:
        if tool_name == "PDF_Documents":
            continue

        # Find matching tool
        matching_tool = None
        for tool in research_tools:
            if tool.name == tool_name or tool_name in tool.name:
                matching_tool = tool
                break

        if matching_tool:
            try:
                tool_result = matching_tool.func(question)
                external_docs.append(f"{tool_name}: {tool_result}")
            except Exception as e:
                external_docs.append(f"{tool_name}: Error - {str(e)}")

    return {
        **state,
        "docs": internal_docs,
        "external_docs": external_docs
    }

def grade_documents_node(state: AgentState) -> AgentState:
    """Grade relevance of documents from multiple sources"""
    question = state["question"]
    docs = state.get("docs", [])
    external_docs = state.get("external_docs", [])
    iteration_count = state.get("iteration_count", 0)

    all_docs = docs + external_docs

    # After 2 iterations, just accept what we have to avoid infinite loops
    if iteration_count >= 2:
        return {**state, "relevant": True}

    if not gemini_llm:
        return {**state, "relevant": True}

    # Show more documents to Gemini for better evaluation
    docs_preview = "\n\n---\n\n".join(all_docs[:5])

    prompt = f"""
    Evaluate the relevance of retrieved documents:

    Question: {question}

    Documents (showing 5 chunks):
    {docs_preview}

    Are these documents sufficient to answer the question?
    - Reply 'yes' if you can find relevant information to answer the question
    - Reply 'no' ONLY if the documents are completely irrelevant or missing critical information
    """

    result = gemini_llm.invoke(prompt)
    is_relevant = "yes" in result.content.lower()

    # Add delay to prevent rate limiting
    time.sleep(GEMINI_CALL_DELAY)

    return {**state, "relevant": is_relevant}

def generate_answer_node(state: AgentState) -> AgentState:
    """Generate answer using multiple sources"""
    question = state["question"]
    docs = state.get("docs", [])
    external_docs = state.get("external_docs", [])
    selected_tools = state.get("selected_tools", [])

    all_docs = docs + external_docs
    context = "\n".join(all_docs[:5])

    if not gemini_llm:
        return {**state, "answer": f"Documents: {context[:500]}..."}

    prompt = f"""
    You are an expert assistant. Answer the user's question directly and conversationally using the provided context.

    Question: {question}

    Context:
    {context}

    Instructions:
    - Answer the question DIRECTLY without explaining your process
    - Be concise and to-the-point
    - Use markdown formatting (headers, lists, bold) to make your answer clear
    - DO NOT use phrases like "based on the CV" or "Document_AI_Resume_Parser shows"
    - Just answer as if you're having a conversation
    - If the question asks for a list, provide a bullet list
    - If the question asks for a recommendation, give your recommendation directly

    Provide a clear, well-formatted answer.
    """

    response = gemini_llm.invoke(prompt)

    # Add delay to prevent rate limiting
    time.sleep(GEMINI_CALL_DELAY)

    return {**state, "answer": response.content.strip()}

def answer_check_node(state: AgentState) -> AgentState:
    """Check if answer is satisfactory"""
    question = state["question"]
    answer = state.get("answer", "")

    if not gemini_llm:
        return {**state, "answered": True}

    prompt = f"""
    Does this answer sufficiently respond to the question?
    Question: {question}
    Answer: {answer}
    Reply yes or no.
    """

    result = gemini_llm.invoke(prompt)
    answered = "yes" in result.content.lower()

    # Add delay to prevent rate limiting
    time.sleep(GEMINI_CALL_DELAY)

    return {**state, "answered": answered}

def strategy_adaptation_node(state: AgentState) -> AgentState:
    """Adapt search strategy if answer is insufficient"""
    question = state["question"]
    current_tools = state.get("selected_tools", [])
    iteration_count = state.get("iteration_count", 0)

    if not gemini_llm:
        return {
            **state,
            "selected_tools": current_tools,
            "question": question,
            "iteration_count": iteration_count + 1
        }

    prompt = f"""
    The current answer was insufficient. Adapt the search strategy:

    Original question: {question}
    Previously used tools: {current_tools}

    Suggest different tools or modified search query.

    Format:
    NEW_TOOLS: tool1,tool2
    NEW_QUERY: modified query
    """

    result = gemini_llm.invoke(prompt)
    content = result.content.strip()

    # Add delay to prevent rate limiting
    time.sleep(GEMINI_CALL_DELAY)

    new_tools = current_tools
    new_query = question

    for line in content.split('\n'):
        if 'NEW_TOOLS:' in line:
            new_tools = [tool.strip() for tool in line.replace('NEW_TOOLS:', '').split(',')]
        elif 'NEW_QUERY:' in line:
            new_query = line.replace('NEW_QUERY:', '').strip()

    return {
        **state,
        "selected_tools": new_tools,
        "question": new_query,
        "iteration_count": iteration_count + 1
    }

# ===========================
# Workflow Construction
# ===========================

def create_agentic_rag_workflow() -> StateGraph:
    """Create the Agentic RAG workflow with LangGraph"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("ToolSelection", tool_selection_node)
    workflow.add_node("HybridRetrieve", hybrid_retrieve_node)
    workflow.add_node("MultiRetrieve", multi_source_retrieve_node)
    workflow.add_node("Grade", grade_documents_node)
    workflow.add_node("Generate", generate_answer_node)
    workflow.add_node("Evaluate", answer_check_node)
    workflow.add_node("Adapt", strategy_adaptation_node)

    # Set entry point
    workflow.set_entry_point("ToolSelection")

    # Define edges
    workflow.add_edge("ToolSelection", "HybridRetrieve")
    workflow.add_edge("HybridRetrieve", "MultiRetrieve")
    workflow.add_edge("MultiRetrieve", "Grade")

    workflow.add_conditional_edges(
        "Grade",
        lambda state: "generate" if state.get("relevant") else "adapt",
        {
            "generate": "Generate",
            "adapt": "Adapt"
        }
    )

    workflow.add_edge("Generate", "Evaluate")

    workflow.add_conditional_edges(
        "Evaluate",
        lambda state: "end" if state.get("answered") or state.get("iteration_count", 0) >= 3 else "adapt",
        {
            "end": END,
            "adapt": "Adapt"
        }
    )

    workflow.add_edge("Adapt", "HybridRetrieve")

    return workflow.compile()

# ===========================
# Main Super Agentic RAG Function
# ===========================

def super_agentic_rag(file_path: str, question: str) -> Dict[str, Any]:
    """
    Main function that routes to appropriate RAG strategy based on file type

    Args:
        file_path: Path to the file to analyze
        question: User's question

    Returns:
        Dictionary containing answer, plot_url (if applicable), and other metadata
    """
    # Detect file type
    file_type = detect_file_type(file_path)

    # Route based on file type
    if file_type in ['pdf', 'word', 'powerpoint', 'text']:
        # Use Agentic RAG with Hybrid Search
        return process_document_file(file_path, question, file_type)

    elif file_type in ['csv', 'excel']:
        # Load CSV/Excel and classify data type
        return process_tabular_file(file_path, question, file_type)

    else:
        return {
            "answer": f"Unsupported file type: {file_type}",
            "error": True
        }

def process_document_file(file_path: str, question: str, file_type: str) -> Dict[str, Any]:
    """
    Process PDF, Word, PPT, TXT files using Agentic RAG with Hybrid Search
    """
    print(f"\n{'='*60}")
    print(f"Processing {file_type.upper()} file with Agentic RAG + Hybrid Search")
    print(f"{'='*60}\n")

    # Create workflow
    workflow = create_agentic_rag_workflow()

    # Initialize state
    initial_state = {
        "question": question,
        "original_question": question,
        "file_type": file_type,
        "file_path": file_path,
        "iteration_count": 0
    }

    # Run workflow
    result = workflow.invoke(initial_state)

    return {
        "answer": result.get("answer", "No answer generated"),
        "file_type": file_type,
        "strategy": "agentic_rag_hybrid_search",
        "tools_used": result.get("selected_tools", []),
        "iterations": result.get("iteration_count", 0),
        "reasoning": result.get("reasoning", "")
    }

def process_tabular_file(file_path: str, question: str, file_type: str) -> Dict[str, Any]:
    """
    Process CSV/Excel files with intelligent routing
    """
    print(f"\n{'='*60}")
    print(f"Processing {file_type.upper()} file")
    print(f"{'='*60}\n")

    # Load data
    if file_type == 'csv':
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

    # Classify data type
    data_type = classify_data_type(df, question)
    print(f"Data type classified as: {data_type}\n")

    if data_type == "informational":
        # Use Q&A/Glossary approach with Agentic RAG
        return process_informational_data(df, question, file_path)

    else:
        # Process as tabular data
        return process_tabular_data(df, question)

def process_informational_data(df: pd.DataFrame, question: str, file_path: str) -> Dict[str, Any]:
    """
    Process informational data (Q&A, Glossary) using Agentic RAG
    """
    print("Using Q&A/Glossary approach with Agentic RAG\n")

    # Convert dataframe to text documents
    temp_file = "temp_informational.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(df.to_string())

    # Use Agentic RAG
    result = process_document_file(temp_file, question, 'text')

    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)

    result["data_type"] = "informational"
    return result

def process_tabular_data(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    """
    Process tabular data using NVIDIA LLAMA and/or Gemini
    """
    print("Processing as tabular data for analysis\n")

    # Check if visualization is needed
    viz_needed = check_visualization_intent(question)
    print(f"Visualization needed: {viz_needed}\n")

    if viz_needed:
        # Use NVIDIA LLAMA for code generation and Gemini for insights
        return process_with_visualization(df, question)
    else:
        # Use only Gemini for insights
        return process_for_insights(df, question)

def process_with_visualization(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    """
    Process request with visualization using NVIDIA LLAMA
    """
    print("Generating visualization with NVIDIA LLAMA\n")

    try:
        # Generate code using NVIDIA LLAMA
        code = generate_analysis_code(df, question, should_plot=True)
        print("Generated code:\n", code, "\n")

        # Execute code
        result, error, plot_created = execute_code(code, df)

        if error:
            return {
                "answer": f"Error executing code: {error}",
                "error": True,
                "code": code
            }

        plot_url = None

        # Handle visualization result appropriately
        if isinstance(result, plt.Figure):
            # Save the matplotlib figure to static directory
            import uuid
            plot_id = str(uuid.uuid4())
            filename = f"plot_{plot_id}.png"
            filepath = os.path.join("static", filename)

            os.makedirs("static", exist_ok=True)  # Ensure static directory exists
            result.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(result)  # Close the figure to free memory

            plot_url = f"/static/{filename}"
            plot_created = True
        elif isinstance(result, plt.Axes):
            # Handle case where result is an Axes object
            fig = result.get_figure()
            if fig:
                plot_id = str(uuid.uuid4())
                filename = f"plot_{plot_id}.png"
                filepath = os.path.join("static", filename)

                os.makedirs("static", exist_ok=True)
                fig.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close(fig)

                plot_url = f"/static/{filename}"
                plot_created = True

        # Generate insights using NVIDIA LLAMA
        insights = generate_insight_with_nvidia_gemini(df, question, result, plot_created)

        return {
            "answer": insights,
            "plot_url": plot_url,
            "code": code,
            "data_type": "tabular",
            "strategy": "nvidia_llama_visualization",
            "visualization_created": plot_created
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            "answer": f"Error in visualization process: {str(e)}\nDetails: {error_details}",
            "error": True
        }

def process_for_insights(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    """
    Process request for insights only using Gemini (or NVIDIA LLAMA fallback)
    """
    print("Generating insights with Gemini\n")

    try:
        # First, try to get numerical result if possible
        if nvidia_client:
            code = generate_analysis_code(df, question, should_plot=False)
            result, error, _ = execute_code(code, df)

            if error:
                result = "Unable to compute numerical result"
        else:
            result = "Analysis result"

        # Generate insights using Gemini
        if gemini_llm:
            insights = generate_insight_with_gemini(df, question, result)
        elif nvidia_client:
            insights = generate_insight_with_nvidia_gemini(df, question, result, False)
        else:
            insights = f"Result: {result}"

        return {
            "answer": insights,
            "result": str(result),
            "data_type": "tabular",
            "strategy": "gemini_insights"
        }

    except Exception as e:
        return {
            "answer": f"Error generating insights: {str(e)}",
            "error": True
        }

# ===========================
# FastAPI Application
# ===========================

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import shutil
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Super Agentic RAG System",
    description="Intelligent document and data analysis with NVIDIA LLAMA and Gemini",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Session storage (in production, use Redis or database)
sessions: Dict[str, Dict[str, Any]] = {}

# Pydantic models
class QueryRequest(BaseModel):
    session_id: str
    query: str

class SessionResponse(BaseModel):
    session_id: str
    file_name: str
    file_type: str
    message: str

class QueryResponse(BaseModel):
    answer: str
    plot_url: Optional[str] = None
    strategy: Optional[str] = None
    tools_used: Optional[list] = None
    data_type: Optional[str] = None
    code: Optional[str] = None
    error: bool = False

# ===========================
# API Endpoints
# ===========================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard"""
    return HTMLResponse(content=get_dashboard_html(), status_code=200)

@app.post("/api/upload", response_model=SessionResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file and create a new session
    Supported files: PDF, Word, PowerPoint, TXT, CSV, Excel
    """
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())

        # Save uploaded file
        file_extension = os.path.splitext(file.filename)[1].lower()
        file_path = f"uploads/{session_id}{file_extension}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Detect file type
        file_type = detect_file_type(file_path)

        if file_type == 'unknown':
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: PDF, Word, PowerPoint, TXT, CSV, Excel"
            )

        # Create session
        sessions[session_id] = {
            "file_name": file.filename,
            "file_path": file_path,
            "file_type": file_type,
            "created_at": datetime.now().isoformat(),
            "queries": []
        }

        return SessionResponse(
            session_id=session_id,
            file_name=file.filename,
            file_type=file_type,
            message=f"File uploaded successfully. Type: {file_type.upper()}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query the uploaded document
    The system will intelligently route based on file type
    """
    try:
        # Validate session
        if request.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found. Please upload a file first.")

        session = sessions[request.session_id]
        file_path = session["file_path"]

        # Check if file still exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found. Please re-upload.")

        # Process query using Super Agentic RAG
        result = super_agentic_rag(file_path, request.query)

        # Handle plot URL if exists
        plot_url = result.get("plot_url")
        if plot_url and result.get("visualization_created", False):
            # If we have a proper URL (not a file path), return as is
            # This allows the frontend to load the static file directly
            pass
        elif result.get("plot_url") and os.path.exists(result["plot_url"].lstrip("/")):
            # Convert plot to base64 as fallback
            plot_path = result["plot_url"].lstrip("/")
            with open(plot_path, "rb") as img_file:
                plot_data = base64.b64encode(img_file.read()).decode('utf-8')
                plot_url = f"data:image/png;base64,{plot_data}"
        elif result.get("plot_url") and os.path.exists(result["plot_url"].replace("/static/", "static/")):
            # Alternative path construction
            plot_path = result["plot_url"].replace("/static/", "static/")
            with open(plot_path, "rb") as img_file:
                plot_data = base64.b64encode(img_file.read()).decode('utf-8')
                plot_url = f"data:image/png;base64,{plot_data}"

        # Store query in session
        session["queries"].append({
            "query": request.query,
            "answer": result.get("answer"),
            "timestamp": datetime.now().isoformat()
        })

        return QueryResponse(
            answer=result.get("answer", "No answer generated"),
            plot_url=plot_url,
            strategy=result.get("strategy"),
            tools_used=result.get("tools_used"),
            data_type=result.get("data_type"),
            code=result.get("code"),
            error=result.get("error", False)
        )

    except HTTPException:
        raise
    except Exception as e:
        return QueryResponse(
            answer=f"Error processing query: {str(e)}",
            error=True
        )

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    return {
        "session_id": session_id,
        "file_name": session["file_name"],
        "file_type": session["file_type"],
        "created_at": session["created_at"],
        "query_count": len(session["queries"])
    }

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its associated files"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # Delete file
    if os.path.exists(session["file_path"]):
        os.remove(session["file_path"])

    # Remove session
    del sessions[session_id]

    return {"message": "Session deleted successfully"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "timestamp": datetime.now().isoformat()
    }

# ===========================
# HTML Dashboard
# ===========================

def get_dashboard_html():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Super Agentic RAG - Intelligent Document & Data Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #8b5cf6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #1e1b4b;
            --light: #f8fafc;
            --gray-50: #f9fafb;
            --gray-100: #f1f5f9;
            --gray-200: #e2e8f0;
            --gray-300: #cbd5e1;
            --gray-400: #94a3b8;
            --gray-500: #64748b;
            --gray-600: #475569;
            --gray-700: #334155;
            --gray-800: #1e293b;
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }

        * {
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }

        .container-main {
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Header */
        .header {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: var(--shadow-lg);
            margin-bottom: 2rem;
        }

        .header-title {
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header-subtitle {
            color: var(--gray-600);
            font-size: 1rem;
        }

        /* Main Grid */
        .main-grid {
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 2rem;
            align-items: start;
        }

        /* Upload Card */
        .upload-card {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: var(--shadow-lg);
            position: sticky;
            top: 2rem;
        }

        .upload-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--dark);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .upload-area {
            border: 2px dashed var(--gray-300);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: var(--gray-50);
        }

        .upload-area:hover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.05);
        }

        .upload-area.dragover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.1);
            transform: scale(1.02);
        }

        .upload-icon {
            font-size: 3rem;
            color: var(--primary);
            margin-bottom: 1rem;
        }

        .upload-text {
            color: var(--gray-700);
            font-weight: 500;
            margin-bottom: 0.5rem;
        }

        .upload-hint {
            color: var(--gray-500);
            font-size: 0.875rem;
        }

        .file-input {
            display: none;
        }

        .file-info {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            border: 1px solid var(--primary);
            border-radius: 12px;
            padding: 1rem;
            margin-top: 1rem;
        }

        .file-info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .file-info-row:last-child {
            margin-bottom: 0;
        }

        .file-info-label {
            color: var(--gray-600);
            font-size: 0.875rem;
        }

        .file-info-value {
            color: var(--dark);
            font-weight: 600;
            font-size: 0.875rem;
        }

        .badge-type {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        /* Chat Card */
        .chat-card {
            background: white;
            border-radius: 20px;
            box-shadow: var(--shadow-lg);
            display: flex;
            flex-direction: column;
            height: calc(100vh - 250px);
            min-height: 600px;
        }

        .chat-header {
            padding: 1.5rem 2rem;
            border-bottom: 1px solid var(--gray-200);
        }

        .chat-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--dark);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            background: var(--gray-50);
        }

        .message {
            margin-bottom: 1.5rem;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message-user {
            display: flex;
            justify-content: flex-end;
        }

        .message-user .message-content {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 1rem 1.25rem;
            border-radius: 18px 18px 4px 18px;
            max-width: 70%;
            box-shadow: var(--shadow);
        }

        .message-assistant {
            display: flex;
            justify-content: flex-start;
        }

        .message-assistant .message-content {
            background: white;
            color: var(--gray-800);
            padding: 1rem 1.25rem;
            border-radius: 18px 18px 18px 4px;
            max-width: 85%;
            box-shadow: var(--shadow);
            border: 1px solid var(--gray-200);
        }

        .message-text {
            line-height: 1.6;
            word-wrap: break-word;
        }

        .message-text h1, .message-text h2, .message-text h3 {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: var(--dark);
        }

        .message-text h1 {
            font-size: 1.5rem;
        }

        .message-text h2 {
            font-size: 1.25rem;
        }

        .message-text h3 {
            font-size: 1.1rem;
        }

        .message-text ul, .message-text ol {
            margin: 0.5rem 0;
            padding-left: 1.5rem;
        }

        .message-text li {
            margin: 0.25rem 0;
        }

        .message-text p {
            margin: 0.5rem 0;
        }

        .message-text strong {
            font-weight: 600;
            color: var(--dark);
        }

        .message-text code {
            background: var(--gray-100);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }

        .message-text pre {
            background: var(--gray-800);
            color: #e2e8f0;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            margin: 0.5rem 0;
        }

        .message-text pre code {
            background: none;
            padding: 0;
            color: inherit;
        }

        .message-image {
            margin-top: 1rem;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: var(--shadow);
        }

        .message-image img {
            width: 100%;
            height: auto;
            display: block;
        }

        .message-meta-info {
            margin-top: 0.75rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--gray-200);
            font-size: 0.75rem;
            color: var(--gray-500);
        }

        .meta-badge {
            display: inline-block;
            background: var(--gray-100);
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            margin-right: 0.5rem;
            margin-top: 0.25rem;
        }

        .chat-input-area {
            padding: 1.5rem 2rem;
            border-top: 1px solid var(--gray-200);
            background: white;
        }

        .input-group-custom {
            display: flex;
            gap: 0.75rem;
        }

        .input-custom {
            flex: 1;
            border: 2px solid var(--gray-200);
            border-radius: 12px;
            padding: 0.875rem 1.25rem;
            font-size: 0.95rem;
            transition: all 0.2s;
        }

        .input-custom:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }

        .btn-send {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            border: none;
            color: white;
            padding: 0.875rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-send:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .btn-send:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .loading-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--gray-500);
            padding: 1rem;
        }

        .spinner {
            border: 2px solid var(--gray-300);
            border-top: 2px solid var(--primary);
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--gray-400);
            text-align: center;
            padding: 2rem;
        }

        .empty-state i {
            font-size: 4rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }

        .empty-state-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--gray-600);
        }

        .empty-state-text {
            color: var(--gray-500);
        }

        .alert-custom {
            padding: 1rem 1.25rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            display: flex;
            align-items: start;
            gap: 0.75rem;
        }

        .alert-error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--danger);
            color: var(--danger);
        }

        .alert-success {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success);
            color: var(--success);
        }

        .alert-info {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid var(--primary);
            color: var(--primary);
        }

        /* Responsive */
        @media (max-width: 1200px) {
            .main-grid {
                grid-template-columns: 1fr;
            }

            .upload-card {
                position: static;
            }
        }

        @media (max-width: 768px) {
            body {
                padding: 1rem;
            }

            .header {
                padding: 1.5rem;
            }

            .header-title {
                font-size: 1.5rem;
            }

            .chat-card {
                height: calc(100vh - 200px);
            }
        }

        /* Code block styling */
        .code-block {
            background: var(--gray-800);
            color: #e2e8f0;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.875rem;
            margin-top: 0.75rem;
        }
    </style>
</head>
<body>
    <div class="container-main">
        <!-- Header -->
        <div class="header">
            <div class="header-title">
                <i class="bi bi-stars"></i> Super Agentic RAG
            </div>
            <div class="header-subtitle">
                Intelligent Document & Data Analysis powered by NVIDIA LLAMA and Gemini AI
            </div>
        </div>

        <!-- Main Grid -->
        <div class="main-grid">
            <!-- Upload Section -->
            <div class="upload-card">
                <div class="upload-title">
                    <i class="bi bi-cloud-upload"></i>
                    Upload Document
                </div>

                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">
                        <i class="bi bi-file-earmark-arrow-up"></i>
                    </div>
                    <div class="upload-text">
                        Click to upload or drag and drop
                    </div>
                    <div class="upload-hint">
                        PDF, Word, PowerPoint, TXT, CSV, or Excel
                    </div>
                    <input type="file" id="fileInput" class="file-input" accept=".pdf,.doc,.docx,.ppt,.pptx,.txt,.csv,.xls,.xlsx">
                </div>

                <div id="fileInfo" style="display: none;">
                </div>

                <div id="uploadAlert" style="display: none;">
                </div>
            </div>

            <!-- Chat Section -->
            <div class="chat-card">
                <div class="chat-header">
                    <div class="chat-title">
                        <i class="bi bi-chat-dots"></i>
                        Ask Questions
                    </div>
                </div>

                <div class="chat-messages" id="chatMessages">
                    <div class="empty-state">
                        <i class="bi bi-chat-square-text"></i>
                        <div class="empty-state-title">No conversation yet</div>
                        <div class="empty-state-text">Upload a document and start asking questions</div>
                    </div>
                </div>

                <div class="chat-input-area">
                    <div class="input-group-custom">
                        <input
                            type="text"
                            id="queryInput"
                            class="input-custom"
                            placeholder="Ask a question about your document..."
                            disabled
                        >
                        <button id="sendBtn" class="btn-send" disabled>
                            <i class="bi bi-send-fill"></i>
                            Send
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let sessionId = null;
        let fileName = null;
        let fileType = null;

        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const uploadAlert = document.getElementById('uploadAlert');
        const chatMessages = document.getElementById('chatMessages');
        const queryInput = document.getElementById('queryInput');
        const sendBtn = document.getElementById('sendBtn');

        // Upload area click
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });

        // Handle file upload
        async function handleFileUpload(file) {
            const formData = new FormData();
            formData.append('file', file);

            showAlert('Uploading file...', 'info');

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    sessionId = data.session_id;
                    fileName = data.file_name;
                    fileType = data.file_type;

                    showFileInfo(data);
                    showAlert('File uploaded successfully!', 'success');
                    enableChat();
                    clearChat();
                } else {
                    showAlert(data.detail || 'Upload failed', 'error');
                }
            } catch (error) {
                showAlert('Upload failed: ' + error.message, 'error');
            }
        }

        // Show file info
        function showFileInfo(data) {
            fileInfo.style.display = 'block';
            fileInfo.innerHTML = `
                <div class="file-info">
                    <div class="file-info-row">
                        <span class="file-info-label">File Name:</span>
                        <span class="file-info-value">${data.file_name}</span>
                    </div>
                    <div class="file-info-row">
                        <span class="file-info-label">File Type:</span>
                        <span class="badge-type">${data.file_type.toUpperCase()}</span>
                    </div>
                </div>
            `;
        }

        // Show alert
        function showAlert(message, type) {
            uploadAlert.style.display = 'block';
            let alertClass = 'alert-info';
            let icon = 'bi-info-circle';

            if (type === 'error') {
                alertClass = 'alert-error';
                icon = 'bi-exclamation-circle';
            } else if (type === 'success') {
                alertClass = 'alert-success';
                icon = 'bi-check-circle';
            }

            uploadAlert.innerHTML = `
                <div class="alert-custom ${alertClass}">
                    <i class="bi ${icon}"></i>
                    <div>${message}</div>
                </div>
            `;

            if (type !== 'error') {
                setTimeout(() => {
                    uploadAlert.style.display = 'none';
                }, 3000);
            }
        }

        // Enable chat
        function enableChat() {
            queryInput.disabled = false;
            sendBtn.disabled = false;
            queryInput.focus();
        }

        // Clear chat
        function clearChat() {
            chatMessages.innerHTML = '';
        }

        // Send query
        async function sendQuery() {
            const query = queryInput.value.trim();
            if (!query || !sessionId) return;

            // Add user message
            addMessage(query, 'user');
            queryInput.value = '';

            // Show loading
            showLoading();

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        session_id: sessionId,
                        query: query
                    })
                });

                const data = await response.json();

                // Remove loading
                removeLoading();

                // Add assistant message
                addMessage(data.answer, 'assistant', data);

            } catch (error) {
                removeLoading();
                addMessage('Error: ' + error.message, 'assistant', { error: true });
            }
        }

        // Add message to chat
        function addMessage(text, sender, data = {}) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message message-${sender}`;

            let metaInfo = '';
            if (sender === 'assistant' && data) {
                const metaTags = [];
                if (data.strategy) metaTags.push(`<span class="meta-badge">Strategy: ${data.strategy}</span>`);
                if (data.data_type) metaTags.push(`<span class="meta-badge">Type: ${data.data_type}</span>`);
                if (data.tools_used && data.tools_used.length > 0) {
                    metaTags.push(`<span class="meta-badge">Tools: ${data.tools_used.join(', ')}</span>`);
                }

                if (metaTags.length > 0) {
                    metaInfo = `<div class="message-meta-info">${metaTags.join('')}</div>`;
                }
            }

            let imageHtml = '';
            if (data.plot_url) {
                imageHtml = `<div class="message-image"><img src="${data.plot_url}" alt="Visualization"></div>`;
            }

            let codeHtml = '';
            if (data.code) {
                codeHtml = `<div class="code-block">${escapeHtml(data.code)}</div>`;
            }

            // Render markdown for assistant messages, escape HTML for user messages
            let textHtml;
            if (sender === 'assistant') {
                // Use marked.js to render markdown
                textHtml = marked.parse(text);
            } else {
                textHtml = escapeHtml(text);
            }

            messageDiv.innerHTML = `
                <div class="message-content">
                    <div class="message-text">${textHtml}</div>
                    ${imageHtml}
                    ${codeHtml}
                    ${metaInfo}
                </div>
            `;

            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Show loading indicator
        function showLoading() {
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message message-assistant';
            loadingDiv.id = 'loadingIndicator';
            loadingDiv.innerHTML = `
                <div class="message-content">
                    <div class="loading-indicator">
                        <div class="spinner"></div>
                        <span>Processing your query...</span>
                    </div>
                </div>
            `;
            chatMessages.appendChild(loadingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Remove loading indicator
        function removeLoading() {
            const loading = document.getElementById('loadingIndicator');
            if (loading) {
                loading.remove();
            }
        }

        // Escape HTML
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Send button click
        sendBtn.addEventListener('click', sendQuery);

        // Enter key to send
        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuery();
            }
        });
    </script>
</body>
</html>
'''

# ===========================
# Main Entry Point
# ===========================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("SUPER AGENTIC RAG - WEB INTERFACE")
    print("="*60)
    print("\nStarting server...")
    print("Open your browser and go to: http://localhost:8000")
    print("\nPress Ctrl+C to stop the server\n")
    print("Supported file types:")
    print("  - PDF, Word, PowerPoint, TXT → Agentic RAG with Hybrid Search (BM25 + FAISS)")
    print("  - CSV, Excel → Intelligent routing:")
    print("    • Informational data → Q&A/Glossary approach")
    print("    • Tabular data → NVIDIA LLAMA + Gemini analysis")
    print("\n")

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
