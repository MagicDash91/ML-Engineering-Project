import pandas as pd
import polars as pl
import logging
import markdown
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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

# Stopwords for text cleaning
add_stopwords = [
    "the", "of", "is", "a", "in", "and", "to", "for", "that", "it", "on", "this", "with", 
    "as", "at", "are", "be", "by", "an", "or", "not", "my", "i", "me", "we", "you", "your",
    "have", "has", "had", "will", "would", "could", "should", "can", "do", "does", "did"
]
custom_stopwords = "plant,crop,growing,grow,grown"  # Crop-specific stopwords
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

def render_html(columns=None, country_columns=None, plant_columns=None, selected_columns=None, 
                charts_urls=None, word_cloud_urls=None, insights=None):
    # Start base HTML with modernized Bootstrap design
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crop Analysis Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        :root {
            --primary-color: #2d8659;
            --secondary-color: #1e5e3f;
            --light-color: #f8f9fa;
            --dark-color: #212529;
            --success-color: #52c41a;
            --warning-color: #faad14;
            --info-color: #1890ff;
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
            background-color: var(--primary-color);
            border-color: var(--primary-color);
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
        
        .chart-container {
            background-color: #fff;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }
        
        .wordcloud-section {
            margin-top: 3rem;
        }
        
        .wordcloud-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
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
        
        .plant-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 50px;
            font-size: 0.75rem;
            text-transform: capitalize;
            font-weight: 600;
            letter-spacing: 0.5px;
            background-color: var(--success-color);
            color: #fff;
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
        <h1 class="app-title"><i class="fas fa-seedling"></i> Crop Analysis Dashboard</h1>
        <p class="app-description">Upload your crop data CSV file to analyze country distribution, plant types, and symptoms for agricultural insights.</p>
    </div>
"""

    # File upload section
    html += """
    <div class="card upload-card">
        <div class="card-header">
            <h5 class="card-title"><i class="fas fa-upload"></i> Upload Your Crop Data</h5>
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

    # If columns exist, add dropdowns for selection
    if columns:
        html += """
    <div class="card analysis-card">
        <div class="card-header">
            <h5 class="card-title"><i class="fas fa-columns"></i> Select Analysis Columns</h5>
        </div>
        <div class="card-body">
            <form action="/analyze" method="post">
                <div class="row">
                    <div class="col-md-4 mb-3">
                        <label for="country_column" class="form-label">Select Country Column:</label>
                        <select name="country_column" class="form-select" required>
"""
        for col in columns:
            html += f'            <option value="{col}">{col}</option>\n'
        html += """
                        </select>
                    </div>
                    <div class="col-md-4 mb-3">
                        <label for="plant_column" class="form-label">Select Plant Column:</label>
                        <select name="plant_column" class="form-select" required>
"""
        for col in columns:
            html += f'            <option value="{col}">{col}</option>\n'
        html += """
                        </select>
                    </div>
                    <div class="col-md-4 mb-3">
                        <label for="symptom_column" class="form-label">Select Symptom Column:</label>
                        <select name="symptom_column" class="form-select" required>
"""
        for col in columns:
            html += f'            <option value="{col}">{col}</option>\n'
        html += """
                        </select>
                    </div>
                </div>
                <button type="submit" class="btn btn-success custom-btn w-100">
                    <i class="fas fa-chart-pie"></i> Analyze Crop Data
                </button>
            </form>
        </div>
    </div>
"""

    # If charts exist, show the visualizations
    if charts_urls:
        html += f"""
    <div class="analysis-result">
        <h2 class="mb-4 text-center">Analysis Results</h2>
        <div class="row">
            <div class="col-md-6 mb-4">
                <div class="chart-container">
                    <h5 class="card-title mb-3"><i class="fas fa-globe"></i> Top 10 Countries</h5>
                    <img src="/static/{charts_urls['country']}" alt="Country Distribution" class="img-fluid">
                </div>
            </div>
            <div class="col-md-6 mb-4">
                <div class="chart-container">
                    <h5 class="card-title mb-3"><i class="fas fa-leaf"></i> Top 10 Plants</h5>
                    <img src="/static/{charts_urls['plant']}" alt="Plant Distribution" class="img-fluid">
                </div>
            </div>
        </div>
    </div>
"""

    # If word_cloud_urls exist, show the word clouds and insights
    if word_cloud_urls and insights:
        html += """
    <div class="wordcloud-section">
        <h3 class="mb-4 text-center">Plant-Specific Symptom Analysis</h3>
        <div class="wordcloud-grid">
"""
        for plant, url in word_cloud_urls.items():
            html += f"""
            <div class="wordcloud-card">
                <div class="wordcloud-content">
                    <div class="wordcloud-title">
                        <span><i class="fas fa-cloud"></i> {plant} Symptoms</span>
                        <span class="plant-badge">{plant}</span>
                    </div>
                    <div class="mt-3">
                        <img src="/static/{url}" alt="{plant} Symptom Word Cloud" class="img-fluid wordcloud-image">
                    </div>
                    <div class="insights-container mt-3">
                        <h5 class="mb-3"><i class="fas fa-lightbulb"></i> {plant} Analysis</h5>
                        <div class="markdown-content">
                            {insights.get(plant, "No insights available.")}
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
    return HTMLResponse(render_html(columns=None))

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
    uploaded_df["filename"] = file.filename

    # Get all columns for selection
    all_columns = df.columns.tolist()

    if not all_columns:
        return HTMLResponse("No columns found in the uploaded file.", status_code=400)

    return HTMLResponse(render_html(columns=all_columns))

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_crop_data(country_column: str = Form(...), 
                           plant_column: str = Form(...), 
                           symptom_column: str = Form(...)):
    df = uploaded_df.get("data")
    if df is None:
        return HTMLResponse("No uploaded data found. Please upload a file first.", status_code=400)

    # Ensure the static directory exists
    os.makedirs("static", exist_ok=True)

    try:
        # Clean the data
        df_clean = df.dropna(subset=[country_column, plant_column, symptom_column])
        
        # Generate country distribution chart (top 10)
        country_counts = df_clean[country_column].value_counts().head(10).reset_index()
        country_counts.columns = ["Country", "Count"]

        plt.figure(figsize=(12, 6))
        sns.set(style="whitegrid")
        ax = sns.barplot(x="Count", y="Country", data=country_counts, palette="viridis", orient='h')
        ax.set_title("Top 10 Countries by Crop Reports", fontsize=16, fontweight='bold')
        ax.set_xlabel("Number of Reports", fontsize=14)
        ax.set_ylabel("Country", fontsize=14)
        plt.tight_layout()

        country_chart_path = "static/country_distribution.png"
        plt.savefig(country_chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Generate plant distribution chart (top 10)
        plant_counts = df_clean[plant_column].value_counts().head(10).reset_index()
        plant_counts.columns = ["Plant", "Count"]

        plt.figure(figsize=(12, 6))
        sns.set(style="whitegrid")
        ax = sns.barplot(x="Count", y="Plant", data=plant_counts, palette="Set2", orient='h')
        ax.set_title("Top 10 Plants by Report Frequency", fontsize=16, fontweight='bold')
        ax.set_xlabel("Number of Reports", fontsize=14)
        ax.set_ylabel("Plant Type", fontsize=14)
        plt.tight_layout()

        plant_chart_path = "static/plant_distribution.png"
        plt.savefig(plant_chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        charts_urls = {
            'country': 'country_distribution.png',
            'plant': 'plant_distribution.png'
        }

        logging.info(f"Charts saved successfully")

        # Generate word clouds for each plant type (using top plants)
        word_cloud_urls = {}
        insights = {}
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")

        # Get top plants for word cloud generation
        top_plants = plant_counts['Plant'].head(8).tolist()  # Limit to top 8 plants for performance

        for plant in top_plants:
            plant_symptoms = df_clean[df_clean[plant_column] == plant][symptom_column]
            
            if len(plant_symptoms) > 0:
                # Clean the symptom text
                symptoms_text = ' '.join(plant_symptoms.astype(str))
                
                # Clean the text using our cleaning function
                temp_df = pd.DataFrame({symptom_column: [symptoms_text]})
                cleaned_df = clean_text_data(temp_df, symptom_column, all_stopwords)
                cleaned_text = cleaned_df['cleaned_text'].iloc[0] if not cleaned_df.empty else symptoms_text
                
                if cleaned_text and len(cleaned_text.strip()) > 10:
                    # Generate word cloud
                    try:
                        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                              max_words=80, colormap='Set2', 
                                              collocations=False).generate(cleaned_text)
                        wordcloud_path = f"static/{plant.replace(' ', '_').replace('/', '_')}_symptoms_wordcloud.png"
                        wordcloud.to_file(wordcloud_path)
                        word_cloud_urls[plant] = os.path.basename(wordcloud_path)
                        logging.info(f"Word cloud saved for {plant}")
                    except Exception as e:
                        logging.error(f"Failed to generate word cloud for {plant}: {str(e)}")
                        continue

                    # Generate insights using Google Gemini
                    custom_question = f"""
                    As an agricultural consultant, you are tasked with analyzing a set of symptom reports submitted by farmers about their {plant} crops.

                    These symptom reports have been analyzed using NLP techniques such as tokenization, frequency analysis, and clustering. Below are the top symptoms and keywords extracted based on frequency and contextual relevance (simulating the output of a wordcloud):

                    ### Symptom Keywords (Ranked by Frequency):
                    - [Include top 10–20 keywords here, e.g., 'yellow leaves', 'black spots', 'wilting', 'mold', 'drooping', 'rot', 'whiteflies', 'stunted growth', 'chewed leaves']

                    Your task is to interpret these symptoms and provide a professional assessment.

                    Format your response in proper Markdown with the following sections:

                    ## {plant.capitalize()} Crop Symptom Analysis

                    ### Key Symptoms Identified
                    - Based on the keyword list above, summarize the main symptoms affecting the crop.

                    ### Possible Causes
                    - Analyze the symptoms and provide a list of likely causes, such as:
                    - Specific crop diseases (e.g., bacterial leaf blight, powdery mildew)
                    - Pests (e.g., whiteflies, borers)
                    - Environmental conditions (e.g., drought stress, waterlogging)

                    ### Recommended Treatments
                    - Suggest specific treatments or interventions for the identified causes.
                    - Mention any pesticides, organic remedies, or pruning practices that may help.

                    ### Preventive Measures
                    - Suggest actions farmers can take to avoid these problems in the future.
                    - Could include crop rotation, resistant varieties, soil health monitoring, etc.

                    ### Additional Notes
                    - Provide any additional insights or tips from agronomic best practices.

                    Be clear, specific, and actionable. Your response will be read by agricultural extension workers and rural farmers, so aim for practical and easy-to-follow advice based on the above keywords.
                    """
                    
                    try:
                        response = model.generate_content(custom_question)
                        md_content = response.text
                        html_content = markdown.markdown(md_content)
                        insights[plant] = html_content
                        logging.info(f"Insights generated for {plant}")
                    except Exception as e:
                        logging.error(f"Failed to generate insights for {plant}: {str(e)}")
                        insights[plant] = f"<p>Analysis not available for {plant} at this time.</p>"
                else:
                    logging.warning(f"Insufficient text data for {plant}")
                    insights[plant] = f"<p>Insufficient symptom data available for {plant} analysis.</p>"
            else:
                logging.warning(f"No symptom data found for {plant}")
                insights[plant] = f"<p>No symptom data available for {plant}.</p>"

        return HTMLResponse(render_html(columns=None, 
                                        charts_urls=charts_urls,
                                        word_cloud_urls=word_cloud_urls, 
                                        insights=insights))
                                        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        return HTMLResponse(f"An error occurred during analysis: {str(e)}", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000, log_level="info")