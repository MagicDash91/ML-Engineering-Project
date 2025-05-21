import pandas as pd
import polars as pl
import logging
import markdown
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from transformers import pipeline
import io
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load emotion classifier model
classifier = pipeline("text-classification", model="ayoubkirouane/BERT-Emotions-Classifier")
app.mount("/static", StaticFiles(directory="static"), name="static")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = "AIzaSyAMAYxkjP49QZRCg21zImWWAu7c3YHJ0a8"  # Fallback key (consider removing in production)
genai.configure(api_key=api_key)

# Configure LangSmith
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Global storage
uploaded_df = {}

# Stopwords
add_stopwords = [
    "the", "of", "is", "a", "in", "https", "yg", "gua", "gue", "lo", "lu", "gw", "and", "to", "for",
    "that", "it", "on", "this", "with", "as", "at", "are", "be", "by", "an", "or", "not",
]
custom_stopwords = "yang,dengan,dari,pada,untuk,dan,atau,ini,itu"  # Example
custom_stopword_list = [word.strip() for word in custom_stopwords.split(",")]
all_stopwords = add_stopwords + custom_stopword_list

def clean_text_data(df: pd.DataFrame, target_variable: str, all_stopwords: list) -> pd.DataFrame:
    logging.info("Cleaning text...")
    pl_df = pl.from_pandas(df)
    hyperlink_pattern = r"https?://\S+|www\.\S+"
    emoticon_pattern = r"[:;=X8B][-oO^']?[\)\(DPp\[\]{}@/\|\\<>*~]"
    emoji_pattern = r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]"
    number_pattern = r"\b\d+\b"
    special_char_pattern = r"[^a-zA-Z\s]"

    pl_df = pl_df.with_columns(
        pl.col(target_variable).cast(pl.Utf8).alias(target_variable)
    )

    pl_df = pl_df.with_columns(
        pl.col(target_variable)
        .str.replace(hyperlink_pattern, "", literal=False)
        .str.replace(emoticon_pattern, "", literal=False)
        .str.replace(emoji_pattern, "", literal=False)
        .str.replace(number_pattern, "", literal=False)
        .str.replace(special_char_pattern, "", literal=False)
        .str.replace(r"\s+", " ", literal=False)
        .str.strip_chars()
        .alias("cleaned_text")
    )

    stopwords_set = pl.Series("stopwords", all_stopwords)

    pl_df = pl_df.with_columns(
        pl.col("cleaned_text")
        .str.split(" ")
        .list.eval(
            pl.when(pl.element().str.to_lowercase().is_in(stopwords_set))
            .then(None)
            .otherwise(pl.element())
        )
        .list.drop_nulls()
        .list.join(" ")
        .alias("cleaned_text")
    )

    pl_df = pl_df.filter(pl.col("cleaned_text").str.len_chars() <= 512)
    return pl_df.to_pandas()

def render_html(columns=None, bar_chart_url=None, word_cloud_urls=None, insights=None):
    # Start base HTML with modernized Bootstrap design
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Emotion Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        :root {
            --primary-color: #4361ee;
            --secondary-color: #3a0ca3;
            --light-color: #f8f9fa;
            --dark-color: #212529;
            --success-color: #4cc9f0;
            --warning-color: #f72585;
            --info-color: #7209b7;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f7f9fc;
            color: var(--dark-color);
        }
        
        .app-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }
        
        .app-header {
            margin-bottom: 3rem;
            text-align: center;
        }
        
        .app-title {
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 1rem;
        }
        
        .app-description {
            color: #6c757d;
            max-width: 600px;
            margin: 0 auto;
        }
        
        .upload-card, .analysis-card {
            border: none;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 2rem;
            overflow: hidden;
        }
        
        .upload-card:hover, .analysis-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 25px rgba(0,0,0,0.1);
        }
        
        .card-header {
            background-color: #fff;
            border-bottom: 1px solid rgba(0,0,0,0.05);
            padding: 1.5rem;
        }
        
        .card-title {
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 0;
            display: flex;
            align-items: center;
        }
        
        .card-title i {
            margin-right: 10px;
        }
        
        .card-body {
            padding: 1.5rem;
        }
        
        .custom-btn {
            padding: 0.6rem 1.5rem;
            border-radius: 50px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background-color: var(--primary-color);
            border-color: var(--primary-color);
        }
        
        .btn-primary:hover {
            background-color: var(--secondary-color);
            border-color: var(--secondary-color);
        }
        
        .btn-success {
            background-color: var(--success-color);
            border-color: var(--success-color);
        }
        
        .btn-success:hover {
            background-color: var(--info-color);
            border-color: var(--info-color);
        }
        
        .file-upload {
            position: relative;
            overflow: hidden;
            margin-bottom: 1rem;
            background: #f8f9fa;
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .file-upload:hover {
            border-color: var(--primary-color);
        }
        
        .file-upload input[type="file"] {
            opacity: 0;
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }
        
        .file-upload-icon {
            font-size: 2.5rem;
            color: #adb5bd;
            margin-bottom: 1rem;
        }
        
        .file-upload-text {
            color: #6c757d;
        }
        
        .form-select {
            border-radius: 8px;
            padding: 0.75rem;
            border: 1px solid #ced4da;
            margin-bottom: 1rem;
        }
        
        .analysis-result {
            margin-top: 3rem;
        }
        
        .emotion-chart {
            background-color: #fff;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .emotion-chart img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }
        
        .wordcloud-section {
            margin-top: 3rem;
        }
        
        .wordcloud-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 2rem;
        }
        
        .wordcloud-card {
            background-color: #fff;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            overflow: hidden;
            transition: transform 0.3s ease;
        }
        
        .wordcloud-card:hover {
            transform: translateY(-5px);
        }
        
        .wordcloud-image {
            width: 100%;
            height: auto;
            border-radius: 8px 8px 0 0;
        }
        
        .wordcloud-content {
            padding: 1.5rem;
        }
        
        .wordcloud-title {
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .emotion-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 50px;
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        
        .badge-joy {
            background-color: #ffd166;
            color: #000;
        }
        
        .badge-optimism {
            background-color: #06d6a0;
            color: #000;
        }
        
        .badge-anger {
            background-color: #ef476f;
            color: #fff;
        }
        
        .badge-sadness {
            background-color: #118ab2;
            color: #fff;
        }
        
        .badge-love {
            background-color: #ff70a6;
            color: #000;
        }
        
        .badge-fear {
            background-color: #073b4c;
            color: #fff;
        }
        
        .badge-surprise {
            background-color: #ffd60a;
            color: #000;
        }
        
        .insights-container {
            padding: 1rem;
            border-radius: 8px;
            background-color: #f8f9fa;
            margin-top: 1rem;
        }
        
        .insights-container .markdown-content {
            font-size: 0.9rem;
            line-height: 1.6;
        }
        
        .insights-container h1,
        .insights-container h2,
        .insights-container h3 {
            font-size: 1.2rem;
            margin-top: 1rem;
        }
        
        .insights-container ul,
        .insights-container ol {
            padding-left: 1.5rem;
        }
        
        .insights-container li {
            margin-bottom: 0.5rem;
        }
        
        .file-info {
            background-color: #e9f7fe;
            border-left: 4px solid var(--success-color);
            padding: 0.75rem 1rem;
            margin-bottom: 1rem;
            border-radius: 0 8px 8px 0;
            display: flex;
            align-items: center;
        }
        
        .file-info i {
            font-size: 1.25rem;
            margin-right: 0.75rem;
            color: var(--primary-color);
        }
        
        .file-name {
            font-weight: 500;
            color: var(--dark-color);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: calc(100% - 2rem);
        }
        
        @media (max-width: 768px) {
            .wordcloud-grid {
                grid-template-columns: 1fr;
            }
            
            .app-title {
                font-size: 1.75rem;
            }
        }
    </style>
</head>
<body>
<div class="app-container">
    <div class="app-header">
        <h1 class="app-title"><i class="fas fa-brain"></i> Text Emotion Analysis</h1>
        <p class="app-description">Upload your data file containing text to analyze emotional sentiment and generate actionable insights.</p>
    </div>
"""

    # File upload section
    html += """
    <div class="card upload-card">
        <div class="card-header">
            <h5 class="card-title"><i class="fas fa-upload"></i> Upload Your Data</h5>
        </div>
        <div class="card-body">
"""

    # If a file has been uploaded already, show the filename
    if columns and 'filename' in uploaded_df:
        html += f"""
            <div class="file-info">
                <i class="fas fa-file-alt"></i>
                <span class="file-name">Currently analyzing: {uploaded_df.get('filename', 'Unknown file')}</span>
            </div>
"""

    html += """
            <form action="/upload" method="post" enctype="multipart/form-data">
                <div class="file-upload">
                    <input class="form-control" type="file" name="file" accept=".csv,.xlsx,.xls" required>
                    <div class="file-upload-icon">
                        <i class="fas fa-file-csv"></i>
                    </div>
                    <p class="file-upload-text">Drag and drop your CSV or Excel file here, or click to browse</p>
                    <p class="text-muted small">Supported formats: .csv, .xlsx, .xls</p>
                </div>
                <button type="submit" class="btn btn-primary custom-btn w-100">
                    <i class="fas fa-cloud-upload-alt"></i> Upload File
                </button>
            </form>
        </div>
    </div>
"""

    # If columns exist, add dropdown for selection
    if columns:
        html += """
    <div class="card analysis-card">
        <div class="card-header">
            <h5 class="card-title"><i class="fas fa-columns"></i> Select Text Column</h5>
        </div>
        <div class="card-body">
            <form action="/analyze" method="post">
                <div class="mb-3">
                    <label for="column" class="form-label">Select the text column to analyze:</label>
                    <select name="column" class="form-select" required>
"""
        for col in columns:
            html += f'            <option value="{col}">{col}</option>\n'
        html += """
                    </select>
                </div>
                <button type="submit" class="btn btn-success custom-btn w-100">
                    <i class="fas fa-chart-pie"></i> Analyze Emotions
                </button>
            </form>
        </div>
    </div>
"""

    # If bar_chart_url exists, show the bar chart
    if bar_chart_url:
        html += f"""
    <div class="analysis-result">
        <h2 class="mb-4 text-center">Analysis Results</h2>
        <div class="emotion-chart">
            <h5 class="card-title mb-3"><i class="fas fa-chart-bar"></i> Emotion Distribution</h5>
            <img src="/static/{bar_chart_url}" alt="Emotion Distribution" class="img-fluid">
        </div>
    </div>
"""

    # If word_cloud_urls exist, show the word clouds and insights
    if word_cloud_urls and insights:
        html += """
    <div class="wordcloud-section">
        <h3 class="mb-4 text-center">Emotional Insights & Word Clouds</h3>
        <div class="row">
"""
        for emotion, url in word_cloud_urls.items():
            badge_class = f"badge-{emotion.lower()}" if emotion.lower() in ["joy", "sadness", "anger", "fear", "love", "surprise", "optimism"] else "bg-secondary"
            
            html += f"""
        <div class="col-lg-6 mb-4">
            <div class="wordcloud-card">
                <div class="wordcloud-content">
                    <div class="wordcloud-title">
                        <span><i class="fas fa-cloud"></i> {emotion} Word Cloud</span>
                        <span class="emotion-badge {badge_class}">{emotion}</span>
                    </div>
                    <div class="mt-3">
                        <img src="/static/{url}" alt="{emotion} Word Cloud" class="img-fluid wordcloud-image">
                    </div>
                    <div class="insights-container mt-3">
                        <h5 class="mb-3"><i class="fas fa-lightbulb"></i> {emotion} Insights</h5>
                        <div class="markdown-content">
                            {insights.get(emotion, "No insights available.")}
                        </div>
                    </div>
                </div>
            </div>
        </div>
"""
        html += """
        </div>
    </div>
"""

    html += """
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    return html

@app.get("/", response_class=HTMLResponse)
async def upload_form():
    return HTMLResponse(render_html(columns=None, bar_chart_url=None, word_cloud_urls=None, insights=None))

@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return HTMLResponse("Unsupported file type. Please upload a CSV or Excel file.", status_code=400)
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        return HTMLResponse(f"Error reading file: {str(e)}", status_code=400)

    uploaded_df["data"] = df
    uploaded_df["filename"] = file.filename  # Store the filename

    # Find text columns (columns with strings longer than 30 chars)
    string_cols = []
    for col in df.select_dtypes(include="object").columns:
        if df[col].apply(lambda x: isinstance(x, str) and len(str(x)) > 30).any():
            string_cols.append(col)

    if not string_cols:
        return HTMLResponse("No suitable text columns found in the uploaded file. Please ensure your file contains columns with text content.", status_code=400)

    return HTMLResponse(render_html(columns=string_cols, bar_chart_url=None, word_cloud_urls=None, insights=None))

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_column(column: str = Form(...)):
    df = uploaded_df.get("data")
    if df is None:
        return HTMLResponse("No uploaded data found. Please upload a file first.", status_code=400)

    # Ensure the static directory exists
    os.makedirs("static", exist_ok=True)

    try:
        df_cleaned = clean_text_data(df, column, all_stopwords)
        df_cleaned["emotion"] = df_cleaned["cleaned_text"].apply(lambda x: classifier(x)[0]["label"] if x.strip() else "Unknown")

        emotion_counts = df_cleaned["emotion"].value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]

        # Generate and save the barchart
        plt.figure(figsize=(12, 6))
        sns.set(style="whitegrid")
        ax = sns.barplot(x="Emotion", y="Count", data=emotion_counts, palette="viridis")
        ax.set_title("Emotion Distribution", fontsize=16, fontweight='bold')
        ax.set_xlabel("Emotion", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()

        bar_chart_path = "static/emotion_distribution.png"
        plt.savefig(bar_chart_path)
        plt.close()

        logging.info(f"Bar chart saved to: {bar_chart_path}")

        # Generate and save word clouds for each emotion
        word_cloud_urls = {}
        insights = {}
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")

        for emotion in emotion_counts["Emotion"]:
            emotion_texts = df_cleaned[df_cleaned["emotion"] == emotion]["cleaned_text"].str.cat(sep=" ")
            if emotion_texts:
                # Generate word cloud
                wordcloud = WordCloud(width=1200, height=600, background_color='white', 
                                      max_words=100, colormap='viridis').generate(emotion_texts)
                wordcloud_path = f"static/{emotion}_wordcloud.png"
                wordcloud.to_file(wordcloud_path)
                word_cloud_urls[emotion] = os.path.basename(wordcloud_path)
                logging.info(f"Word cloud saved to: {wordcloud_path}")

                # Generate insights using Google Gemini with specific instruction to format as Markdown
                custom_question = f"""
                As a marketing consultant, analyze consumer insights derived from the word cloud for the {emotion} sentiment.
                Format your response as proper Markdown with:
                
                1. A clear heading for the {emotion} sentiment analysis
                2. Bullet points for key findings
                3. Nested bullet points for supporting details
                4. Bold text for important concepts
                5. Actionable insights clearly marked
                
                Focus on what the {emotion} sentiment reveals about consumer attitudes and how this can be used in marketing strategy.
                """
                
                try:
                    response = model.generate_content(custom_question)
                    # Convert response to HTML via Markdown
                    md_content = response.text
                    html_content = markdown.markdown(md_content)
                    insights[emotion] = html_content
                    logging.info(f"Insights generated for {emotion}")
                except Exception as e:
                    logging.error(f"Failed to generate insights for {emotion}: {str(e)}")
                    insights[emotion] = "<p>Failed to generate insights.</p>"
            else:
                logging.warning(f"No text data found for emotion: {emotion}")
                insights[emotion] = "<p>No text data available for analysis.</p>"

        return HTMLResponse(render_html(columns=None, bar_chart_url="emotion_distribution.png", 
                                        word_cloud_urls=word_cloud_urls, insights=insights))
                                        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        return HTMLResponse(f"An error occurred during analysis: {str(e)}", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000, log_level="info")