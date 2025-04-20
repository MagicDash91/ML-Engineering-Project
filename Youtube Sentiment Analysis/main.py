from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import pandas as pd
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
from fpdf import FPDF
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from huggingface_hub import login
import polars as pl
import google.generativeai as genai
from markdown import markdown
from itertools import islice

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

gemini_model_2_flash = "gemini-2.0-flash"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>YouTube Sentiment Analysis</title>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background-color: #f8f9fa;
            }
            .container {
                max-width: 1200px;
            }
            .card {
                margin-bottom: 20px;
            }
            .card-header {
                background-color: #007bff;
                color: white;
            }
            .card-body img {
                max-width: 100%;
                height: auto;
            }
            .markdown-content {
                font-size: 16px;
                line-height: 1.6;
            }
        </style>
    </head>
    <body>
        <div class="container mt-5">
            <h1 class="mb-4 text-center">YouTube Sentiment Analysis</h1>
            <form action="/analyze" method="post">
                <div class="form-group">
                    <label for="youtube_url">YouTube URL</label>
                    <input type="text" class="form-control" id="youtube_url" name="youtube_url" required>
                </div>
                <div class="form-group">
                    <label for="custom_stopwords">Custom Stopwords (comma-separated)</label>
                    <input type="text" class="form-control" id="custom_stopwords" name="custom_stopwords">
                </div>
                <div class="form-group">
                    <label for="custom_question">Custom Question</label>
                    <textarea class="form-control" id="custom_question" name="custom_question" rows="3">Please provide insights based on the sentiment analysis:</textarea>
                </div>
                <button type="submit" class="btn btn-primary btn-block">Analyze</button>
            </form>
        </div>
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.3/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    </body>
    </html>
    """

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_youtube(
    youtube_url: str = Form(...),
    custom_stopwords: str = Form(...),
    custom_question: str = Form(...),
):
    api_key = "***************************************"
    hf_token = "****************************************"
    login(hf_token)

    # Initialize the downloader
    downloader = YoutubeCommentDownloader()

    # Fetch comments from a YouTube URL
    comments = downloader.get_comments_from_url(
        youtube_url,
        sort_by=SORT_BY_POPULAR
    )

    # Collect all "text" fields
    comment_texts = [comment['text'] for comment in islice(comments, 1000)]  # Collect up to 1000 comments

    # Create a DataFrame
    df = pd.DataFrame(comment_texts, columns=["comment"])

    if df.empty:
        return "<div class='alert alert-danger'>No comments found.</div>"

    add_stopwords = [
        "the",
        "of",
        "is",
        "a",
        "in",
        "https",
        "yg",
        "gua",
        "gue",
        "lo",
        "lu",
        "gw",
    ]
    custom_stopword_list = [word.strip() for word in custom_stopwords.split(",")]
    all_stopwords = add_stopwords + custom_stopword_list

    def clean_text_data(df: pd.DataFrame, target_variable: str, all_stopwords: list) -> pd.DataFrame:
        logging.info("Starting text cleaning process with Polars")
        
        # Convert Pandas DataFrame to Polars DataFrame
        pl_df = pl.from_pandas(df)
        
        # Define regex patterns
        hyperlink_pattern = r"https?://\S+|www\.\S+"
        emoticon_pattern = r"[:;=X8B][-oO^']?[\)\(DPp\[\]{}@/\|\\<>*~]"
        emoji_pattern = r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]"
        number_pattern = r"\b\d+\b"
        special_char_pattern = r"[^a-zA-Z\s]"  # Removes non-alphabet characters

        # Ensure target_variable is of type string
        pl_df = pl_df.with_columns(
            pl.col(target_variable).cast(pl.Utf8).alias(target_variable)
        )

        # Apply text cleaning steps (corrected)
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

        # Remove stopwords
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

        # Drop rows where cleaned_text length is greater than 512
        pl_df = pl_df.filter(pl.col("cleaned_text").str.len_chars() <= 512)

        logging.info(f"Text cleaning complete. Final dataframe shape: {pl_df.shape}")
        return pl_df.to_pandas()  # Convert back to Pandas DataFrame

    df = clean_text_data(df, "comment", all_stopwords)

    # Perform Sentiment Analysis
    pretrained = "mdhugol/indonesia-bert-sentiment-classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained, token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained, token=hf_token)
    sentiment_analysis = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer
    )
    label_index = {
        "LABEL_0": "positive",
        "LABEL_1": "neutral",
        "LABEL_2": "negative",
    }

    def analyze_sentiment(text):
        result = sentiment_analysis(text)
        label = label_index[result[0]["label"]]
        score = result[0]["score"]
        return pd.Series({"sentiment_label": label, "sentiment_score": score})

    df[["sentiment_label", "sentiment_score"]] = df["cleaned_text"].apply(
        analyze_sentiment
    )

    # Count the occurrences of each sentiment label
    sentiment_counts = df["sentiment_label"].value_counts()

    # Plot a bar chart using seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis"
    )
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment Label")
    plt.ylabel("Count")
    sentiment_plot_path = "static/sentiment_distribution.png"
    plt.savefig(sentiment_plot_path)
    plt.close()

    # Concatenate Cleaned text
    positive_text = " ".join(
        df[df["sentiment_label"] == "positive"]["cleaned_text"]
    )
    negative_text = " ".join(
        df[df["sentiment_label"] == "negative"]["cleaned_text"]
    )
    neutral_text = " ".join(
        df[df["sentiment_label"] == "neutral"]["cleaned_text"]
    )

    wordcloud_paths = []
    gemini_responses = {}

    # Create WordCloud Positive
    if positive_text:
        wordcloud = WordCloud(
            min_font_size=3,
            max_words=200,
            width=800,
            height=400,
            colormap="Set2",
            background_color="white",
        ).generate(positive_text)

        wordcloud_positive = "static/wordcloud_positive.png"
        wordcloud.to_file(wordcloud_positive)
        wordcloud_paths.append(wordcloud_positive)

        # Use Google Gemini API to generate content based on the uploaded image
        img = Image.open(wordcloud_positive)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(gemini_model_2_flash)

        try:
            response = model.generate_content(
                [
                    custom_question
                    + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud positive sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the positive sentiment analysis.",
                    img,
                ]
            )
            response.resolve()
            gemini_responses["positive"] = response.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_responses["positive"] = "Error: Failed to generate content with Gemini API."

    # Create WordCloud Negative
    if negative_text:
        wordcloud = WordCloud(
            min_font_size=3,
            max_words=200,
            width=800,
            height=400,
            colormap="Set2",
            background_color="white",
        ).generate(negative_text)

        wordcloud_negative = "static/wordcloud_negative.png"
        wordcloud.to_file(wordcloud_negative)
        wordcloud_paths.append(wordcloud_negative)

        img = Image.open(wordcloud_negative)
        try:
            response = model.generate_content(
                [
                    custom_question
                    + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud negative sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the negative sentiment analysis.",
                    img,
                ]
            )
            response.resolve()
            gemini_responses["negative"] = response.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_responses["negative"] = "Error: Failed to generate content with Gemini API."

    # Create WordCloud Neutral
    if neutral_text:
        wordcloud = WordCloud(
            min_font_size=3,
            max_words=200,
            width=800,
            height=400,
            colormap="Set2",
            background_color="white",
        ).generate(neutral_text)

        wordcloud_neutral = "static/wordcloud_neutral.png"
        wordcloud.to_file(wordcloud_neutral)
        wordcloud_paths.append(wordcloud_neutral)

        img = Image.open(wordcloud_neutral)
        try:
            response = model.generate_content(
                [
                    custom_question
                    + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.",
                    img,
                ]
            )
            response.resolve()
            gemini_responses["neutral"] = response.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_responses["neutral"] = "Error: Failed to generate content with Gemini API."

    # Combine WordClouds (Positive, Neutral, Negative)
    combined_wordcloud_path = None
    if len(wordcloud_paths) > 1:
        # Open the wordcloud images
        images = [Image.open(wc) for wc in wordcloud_paths]

        # Assuming all wordclouds are the same size, we can place them side by side
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        # Create a new image with combined width and max height
        combined_image = Image.new("RGB", (total_width, max_height))

        # Paste each image into the new combined image
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width

        # Save the combined image to the static folder
        combined_wordcloud_path = "static/wordcloud_combined.png"
        combined_image.save(combined_wordcloud_path)

    def generate_gemini_response(plot_path):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(gemini_model_2_flash)
        img = Image.open(plot_path)
        response = model.generate_content(
            [
                custom_question
                + " As a marketing consultant, I want to analyze consumer insights from the sentiment word clouds (positive, neutral, and negative) and the market context. Please summarize your explanation and findings in one concise paragraph and one another paragraph for business insight and recommendation to help me formulate actionable strategies.",
                img,
            ]
        )
        response.resolve()
        return response.text

    response_result = None
    if combined_wordcloud_path:
        response_result = generate_gemini_response(combined_wordcloud_path)

    def safe_encode(text):
            try:
                return text.encode("latin1", errors="replace").decode(
                    "latin1"
                )  # Replace invalid characters
            except Exception as e:
                return f"Error encoding text: {str(e)}"

    # Generate PDF with the results
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt="Sentiment Analysis Report", ln=True, align="C")

    # Sentiment Distribution Plot
    pdf.image(sentiment_plot_path, x=10, y=30, w=190)
    pdf.ln(100)

    # Positive WordCloud and response
    if "positive" in gemini_responses:
        pdf.add_page()
        pdf.cell(200, 10, txt="Positive WordCloud", ln=True, align="C")
        pdf.image(wordcloud_positive, x=10, y=30, w=190)
        pdf.add_page()

        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_responses["positive"]))

    # Negative WordCloud and response
    if "negative" in gemini_responses:
        pdf.add_page()
        pdf.cell(200, 10, txt="Negative WordCloud", ln=True, align="C")
        pdf.image(wordcloud_negative, x=10, y=30, w=190)
        pdf.add_page()

        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_responses["negative"]))

    pdf_file_path = os.path.join("static", "sentiment.pdf")
    pdf.output(pdf_file_path)

    pdf_file_path = os.path.join("static", "sentiment.pdf")
    pdf_file_path = pdf_file_path.replace("\\", "/")

    # Render the results in HTML
    result_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentiment Analysis Results</title>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 1200px;
            }}
            .card {{
                margin-bottom: 20px;
            }}
            .card-header {{
                background-color: #007bff;
                color: white;
            }}
            .card-body img {{
                max-width: 100%;
                height: auto;
            }}
            .markdown-content {{
                font-size: 16px;
                line-height: 1.6;
            }}
        </style>
    </head>
    <body>
        <div class="container mt-5">
            <h1 class="mb-4 text-center">Sentiment Analysis Results</h1>
            <div class="card mb-4">
                <div class="card-header">
                    Sentiment Distribution
                </div>
                <div class="card-body">
                    <img src="/static/sentiment_distribution.png" class="img-fluid" alt="Sentiment Distribution">
                </div>
            </div>
    """

    if "positive" in gemini_responses:
        result_html += f"""
            <div class="card mb-4">
                <div class="card-header">
                    Positive Sentiment Wordcloud
                </div>
                <div class="card-body">
                    <img src="/static/wordcloud_positive.png" class="img-fluid" alt="Positive Sentiment Wordcloud">
                    <div class="mt-3 markdown-content">
                        {markdown(gemini_responses["positive"])}
                    </div>
                </div>
            </div>
        """

    if "negative" in gemini_responses:
        result_html += f"""
            <div class="card mb-4">
                <div class="card-header">
                    Negative Sentiment Wordcloud
                </div>
                <div class="card-body">
                    <img src="/static/wordcloud_negative.png" class="img-fluid" alt="Negative Sentiment Wordcloud">
                    <div class="mt-3 markdown-content">
                        {markdown(gemini_responses["negative"])}
                    </div>
                </div>
            </div>
        """

    if "neutral" in gemini_responses:
        result_html += f"""
            <div class="card mb-4">
                <div class="card-header">
                    Neutral Sentiment Wordcloud
                </div>
                <div class="card-body">
                    <img src="/static/wordcloud_neutral.png" class="img-fluid" alt="Neutral Sentiment Wordcloud">
                    <div class="mt-3 markdown-content">
                        {markdown(gemini_responses["neutral"])}
                    </div>
                </div>
            </div>
        """

    if response_result:
        result_html += f"""
            <div class="card mb-4">
                <div class="card-header">
                    Conclusion
                </div>
                <div class="card-body markdown-content">
                    {markdown(response_result)}
                </div>
            </div>
        """

    result_html += f"""
            <div class="card mb-4">
                <div class="card-header">
                    Download PDF Report
                </div>
                <div class="card-body">
                    <a href="/static/sentiment.pdf" class="btn btn-primary btn-block">Download PDF</a>
                </div>
            </div>
            <a href="/" class="btn btn-secondary btn-block mt-4">Back to Home</a>
        </div>
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.3/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    </body>
    </html>
    """

    return result_html

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