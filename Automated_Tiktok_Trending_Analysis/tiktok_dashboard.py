#!/usr/bin/env python3
"""
TikTok Trending Analysis Dashboard - FastAPI Backend with AI Analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
import base64
import io
import os
import tempfile
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Text processing and word cloud
import re
import emoji
from wordcloud import WordCloud
from collections import Counter

# Google Gemini AI
import google.generativeai as genai
from PIL import Image
import markdown

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure Google Gemini AI
GEMINI_API_KEY = "AIzaSyAMAYxkjP49QZRCg21zImWWAu7c3YHJ0a8"
GEMINI_MODEL = "gemini-2.0-flash-exp"

logging.info("Configuring Google Gemini with API key")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

app = FastAPI(title="TikTok Trending Analysis Dashboard", description="AI-Powered TikTok Trending Videos Analysis Dashboard")

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'appdb',
    'user': 'appuser',
    'password': 'permataputihg101'
}

def get_database_engine():
    """Create database engine"""
    return create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")

def get_tiktok_data(limit: int = 20) -> pd.DataFrame:
    """Retrieve top TikTok videos data from database"""
    engine = get_database_engine()
    
    query = """
    SELECT 
        video_id,
        caption,
        username,
        author_name,
        like_count,
        comment_count,
        share_count,
        play_count,
        video_duration,
        music_title,
        music_author,
        scrape_date,
        created_at,
        created_at_db
    FROM tiktok_data
    ORDER BY like_count DESC, play_count DESC, comment_count DESC
    LIMIT %(limit)s
    """
    
    try:
        df = pd.read_sql(query, engine, params={'limit': limit})
        engine.dispose()
        
        if df.empty:
            raise ValueError("No TikTok data found in database")
        
        # Clean and process data
        df['caption'] = df['caption'].fillna('')
        df['username'] = df['username'].fillna('Unknown')
        df['author_name'] = df['author_name'].fillna('Unknown')
        
        # Ensure numeric columns
        numeric_cols = ['like_count', 'comment_count', 'share_count', 'play_count', 'video_duration']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        return df
    except Exception as e:
        engine.dispose()
        raise Exception(f"Database error: {str(e)}")

def clean_text_for_wordcloud(text: str) -> str:
    """Clean text by removing emojis, hashtags, numbers, symbols, and other unwanted characters"""
    if not text or pd.isna(text):
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    
    # Remove hashtags (keep the word, remove #)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters and symbols, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase and strip
    text = text.lower().strip()
    
    return text

def get_all_captions_cleaned(df: pd.DataFrame) -> str:
    """Get all captions as cleaned text for word cloud"""
    all_text = ""
    for caption in df['caption'].dropna():
        cleaned = clean_text_for_wordcloud(caption)
        if cleaned:
            all_text += cleaned + " "
    return all_text.strip()

def plot_to_base64_and_file(fig, temp_dir: str, filename: str) -> tuple[str, str]:
    """Convert matplotlib figure to base64 string and save as temporary file"""
    # Save as base64
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Save as temporary file for Gemini
    temp_path = os.path.join(temp_dir, filename)
    fig.savefig(temp_path, format='png', dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close(fig)
    return img_base64, temp_path

def analyze_with_gemini(image_path: str, prompt: str) -> str:
    """Analyze image using Google Gemini AI"""
    try:
        logging.info(f"Generating Gemini analysis for {image_path}")
        img = Image.open(image_path)
        
        response = model.generate_content(
            [prompt, img],
            generation_config={"temperature": 0.3},
        ).text
        
        # Convert to HTML markdown
        html_response = markdown.markdown(response)
        return html_response
        
    except Exception as e:
        logging.error(f"Gemini analysis error: {str(e)}")
        return f"<p><strong>AI Analysis Error:</strong> {str(e)}</p>"

def create_top_videos_plot(df: pd.DataFrame, temp_dir: str) -> tuple[str, str]:
    """Create top videos engagement plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TikTok Top Videos Engagement Analysis', fontsize=20, fontweight='bold', y=0.98)
    
    # Top 10 videos by likes
    top_likes = df.head(10)
    bars1 = axes[0, 0].barh(range(len(top_likes)), top_likes['like_count'], color='#ff0050')
    axes[0, 0].set_yticks(range(len(top_likes)))
    axes[0, 0].set_yticklabels([f"@{username}" for username in top_likes['username']], fontsize=10)
    axes[0, 0].set_xlabel('Likes Count')
    axes[0, 0].set_title('Top 10 Videos by Likes', fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        axes[0, 0].text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                       f'{int(width):,}', ha='left', va='center', fontsize=9)
    
    # Engagement distribution
    engagement_data = [df['like_count'].sum(), df['comment_count'].sum(), 
                      df['share_count'].sum(), df['play_count'].sum()]
    engagement_labels = ['Likes', 'Comments', 'Shares', 'Views']
    colors = ['#ff0050', '#00f2ea', '#25f4ee', '#fe2c55']
    
    wedges, texts, autotexts = axes[0, 1].pie(engagement_data, labels=engagement_labels, autopct='%1.1f%%', 
                                            colors=colors, startangle=90)
    axes[0, 1].set_title('Total Engagement Distribution', fontweight='bold')
    
    # Top creators
    creator_engagement = df.groupby('username').agg({
        'like_count': 'sum',
        'comment_count': 'sum',
        'video_id': 'count'
    }).sort_values('like_count', ascending=False).head(8)
    
    bars2 = axes[1, 0].bar(range(len(creator_engagement)), creator_engagement['like_count'], 
                          color='#25f4ee', alpha=0.8)
    axes[1, 0].set_xticks(range(len(creator_engagement)))
    axes[1, 0].set_xticklabels([f"@{creator}" for creator in creator_engagement.index], 
                              rotation=45, ha='right', fontsize=10)
    axes[1, 0].set_ylabel('Total Likes')
    axes[1, 0].set_title('Top Creators by Total Likes', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                       f'{int(height):,}', ha='center', va='bottom', fontsize=9, rotation=0)
    
    # Engagement rate analysis
    df['engagement_rate'] = (df['like_count'] + df['comment_count'] + df['share_count']) / df['play_count'] * 100
    df['engagement_rate'] = df['engagement_rate'].fillna(0)
    
    top_engagement = df.nlargest(10, 'engagement_rate')
    bars3 = axes[1, 1].barh(range(len(top_engagement)), top_engagement['engagement_rate'], 
                           color='#fe2c55', alpha=0.8)
    axes[1, 1].set_yticks(range(len(top_engagement)))
    axes[1, 1].set_yticklabels([f"@{username}" for username in top_engagement['username']], fontsize=10)
    axes[1, 1].set_xlabel('Engagement Rate (%)')
    axes[1, 1].set_title('Top 10 Videos by Engagement Rate', fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars3):
        width = bar.get_width()
        axes[1, 1].text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.1f}%', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Get base64 and save temp file
    img_base64, temp_path = plot_to_base64_and_file(fig, temp_dir, "top_videos.png")
    
    # Create Gemini analysis
    analysis_prompt = f"""
    As a social media analyst specializing in TikTok trends and viral content, analyze this comprehensive TikTok engagement visualization showing the top performing videos.

    Please provide a detailed analysis that includes:

    - **Top Performing Content Analysis**: Examine the top videos by likes and identify:
      - Common patterns among high-performing videos
      - Creator strategies that drive engagement
      - Content themes that resonate with audiences
    - **Engagement Distribution Insights**: Analyze the pie chart showing total engagement:
      - Which engagement type dominates (likes, comments, shares, views)
      - What this distribution reveals about audience behavior
      - Platform-specific engagement patterns
    - **Creator Performance Assessment**: Review the top creators chart:
      - Identify emerging vs established creators
      - Creator consistency and audience loyalty indicators
      - Content creation strategies of top performers
    - **Engagement Rate Analysis**: Examine the engagement rate rankings:
      - Videos with exceptional engagement rates vs view counts
      - Quality vs quantity content strategies
      - Niche content performance vs mainstream appeal
    - **Viral Content Patterns**: Based on the data:
      - Characteristics of videos that achieve viral status
      - Timing and algorithmic factors
      - Audience participation trends
    - **Strategic Recommendations**: For content creators and marketers:
      - Content optimization strategies
      - Engagement maximization techniques
      - Platform algorithm adaptation tips
    - **Market Insights**: What this data reveals about:
      - Current TikTok trends and preferences
      - Audience behavior shifts
      - Content consumption patterns

    Focus on actionable insights for content creators, social media managers, and marketing professionals.

    Important: 
    - Start directly with the analysis
    - Be professional and insightful in your response
    - Use markdown formatting for better readability
    """
    
    gemini_analysis = analyze_with_gemini(temp_path, analysis_prompt)
    
    return img_base64, gemini_analysis

def create_wordcloud_plot(df: pd.DataFrame, temp_dir: str) -> tuple[str, str]:
    """Create word cloud from TikTok captions"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('TikTok Captions Word Cloud & Content Analysis', fontsize=20, fontweight='bold')
    
    # Get all cleaned text
    all_text = get_all_captions_cleaned(df)
    
    if not all_text or len(all_text.strip()) < 10:
        # Create empty plots if no text
        axes[0].text(0.5, 0.5, 'No sufficient text data\nfor word cloud generation', 
                    ha='center', va='center', transform=axes[0].transAxes, fontsize=16)
        axes[0].set_title('Caption Word Cloud', fontweight='bold')
        axes[0].axis('off')
        
        axes[1].text(0.5, 0.5, 'No data available\nfor analysis', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=16)
        axes[1].set_title('Content Keywords Frequency', fontweight='bold')
        axes[1].axis('off')
    else:
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=600,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            random_state=42,
            collocations=False
        ).generate(all_text)
        
        axes[0].imshow(wordcloud, interpolation='bilinear')
        axes[0].set_title('Caption Word Cloud', fontweight='bold', fontsize=16)
        axes[0].axis('off')
        
        # Create word frequency analysis
        words = all_text.split()
        # Filter out common stop words and short words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                     'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                     'can', 'this', 'that', 'these', 'those', 'a', 'an', 'as', 'if', 'it',
                     'its', 'my', 'me', 'we', 'you', 'he', 'she', 'they', 'them', 'her', 'his'}
        
        filtered_words = [word for word in words if len(word) > 2 and word.lower() not in stop_words]
        word_freq = Counter(filtered_words)
        top_words = word_freq.most_common(15)
        
        if top_words:
            words_list, counts_list = zip(*top_words)
            bars = axes[1].barh(range(len(words_list)), counts_list, color='#ff6b6b', alpha=0.8)
            axes[1].set_yticks(range(len(words_list)))
            axes[1].set_yticklabels(words_list)
            axes[1].set_xlabel('Frequency')
            axes[1].set_title('Top 15 Keywords in Captions', fontweight='bold', fontsize=16)
            axes[1].grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                axes[1].text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                           f'{int(width)}', ha='left', va='center', fontsize=10)
        else:
            axes[1].text(0.5, 0.5, 'No keywords found', 
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=16)
            axes[1].set_title('Content Keywords Frequency', fontweight='bold')
            axes[1].axis('off')
    
    plt.tight_layout()
    
    # Get base64 and save temp file
    img_base64, temp_path = plot_to_base64_and_file(fig, temp_dir, "wordcloud.png")
    
    # Create Gemini analysis
    wordcloud_prompt = f"""
    As a content marketing expert and social media strategist, analyze this TikTok word cloud and keyword frequency visualization derived from trending video captions.

    Please provide a comprehensive analysis that includes:

    - **Content Theme Analysis**: Examine the dominant words and themes:
      - Primary content categories and topics
      - Emerging trends and popular subjects
      - Seasonal or cultural influences visible in the text
    - **Language and Communication Patterns**: Analyze how TikTok creators communicate:
      - Popular vocabulary and slang terms
      - Emotional language and sentiment indicators
      - Call-to-action patterns and engagement triggers
    - **Trend Identification**: Identify viral content patterns:
      - Trending hashtags concepts (even without the # symbol)
      - Popular challenges or meme references
      - Cultural moments or events being referenced
    - **Audience Engagement Insights**: What the language reveals about audience preferences:
      - Types of content that drive interaction
      - Emotional triggers that encourage engagement
      - Community language and inside references
    - **Content Strategy Recommendations**: Based on the word analysis:
      - High-impact keywords for content creators
      - Content themes to focus on
      - Language styles that resonate with audiences
    - **Platform-Specific Insights**: TikTok-unique communication patterns:
      - Algorithm-friendly language patterns
      - Platform culture and community expressions
      - Video description optimization opportunities
    - **Marketing Implications**: For brands and marketers:
      - Organic language to incorporate in campaigns
      - Authentic ways to connect with the TikTok audience
      - Content gaps and opportunities

    Focus on actionable insights that content creators, social media managers, and digital marketers can implement.

    Important: 
    - Start directly with the analysis
    - Be professional and strategic in your response
    - Use markdown formatting for better readability
    """
    
    gemini_analysis = analyze_with_gemini(temp_path, wordcloud_prompt)
    
    return img_base64, gemini_analysis

def create_trending_insights_plot(df: pd.DataFrame, temp_dir: str) -> tuple[str, str]:
    """Create trending insights and statistics plot"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('TikTok Trending Insights & Statistics', fontsize=20, fontweight='bold')
    
    # Video duration distribution
    duration_bins = [0, 15, 30, 60, 120, float('inf')]
    duration_labels = ['0-15s', '15-30s', '30-60s', '1-2m', '2m+']
    df['duration_category'] = pd.cut(df['video_duration'], bins=duration_bins, labels=duration_labels, right=False)
    duration_counts = df['duration_category'].value_counts().sort_index()
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
    wedges, texts, autotexts = axes[0, 0].pie(duration_counts.values, labels=duration_counts.index, 
                                            autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 0].set_title('Video Duration Distribution', fontweight='bold')
    
    # Engagement vs Views scatter
    # Filter out zeros to avoid log issues
    valid_data = df[(df['play_count'] > 0) & (df['like_count'] > 0)]
    if not valid_data.empty:
        scatter = axes[0, 1].scatter(valid_data['play_count'], valid_data['like_count'], 
                                   alpha=0.7, s=60, c=valid_data['comment_count'], 
                                   cmap='viridis', edgecolors='black', linewidth=0.5)
        axes[0, 1].set_xlabel('Play Count')
        axes[0, 1].set_ylabel('Like Count')
        axes[0, 1].set_title('Engagement vs Views Correlation', fontweight='bold')
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_yscale('log')
        plt.colorbar(scatter, ax=axes[0, 1], label='Comments')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Top music/audio
    music_data = df[df['music_title'].notna() & (df['music_title'] != '')]['music_title'].value_counts().head(8)
    if not music_data.empty:
        bars = axes[0, 2].barh(range(len(music_data)), music_data.values, color='#ff7675')
        axes[0, 2].set_yticks(range(len(music_data)))
        music_labels = [title[:20] + '...' if len(title) > 20 else title for title in music_data.index]
        axes[0, 2].set_yticklabels(music_labels, fontsize=10)
        axes[0, 2].set_xlabel('Usage Count')
        axes[0, 2].set_title('Popular Music/Audio', fontweight='bold')
        axes[0, 2].grid(axis='x', alpha=0.3)
    
    # Engagement metrics comparison
    metrics = ['like_count', 'comment_count', 'share_count']
    metric_labels = ['Likes', 'Comments', 'Shares']
    metric_means = [df[metric].mean() for metric in metrics]
    metric_maxs = [df[metric].max() for metric in metrics]
    
    x = np.arange(len(metric_labels))
    width = 0.35
    
    bars1 = axes[1, 0].bar(x - width/2, metric_means, width, label='Average', color='#74b9ff', alpha=0.8)
    bars2 = axes[1, 0].bar(x + width/2, metric_maxs, width, label='Maximum', color='#fd79a8', alpha=0.8)
    
    axes[1, 0].set_xlabel('Engagement Type')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Engagement Metrics: Average vs Maximum', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metric_labels)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Content length analysis (caption length)
    df['caption_length'] = df['caption'].str.len()
    length_bins = [0, 50, 100, 200, 300, float('inf')]
    length_labels = ['Very Short\n(0-50)', 'Short\n(50-100)', 'Medium\n(100-200)', 'Long\n(200-300)', 'Very Long\n(300+)']
    df['caption_length_category'] = pd.cut(df['caption_length'], bins=length_bins, labels=length_labels, right=False)
    length_counts = df['caption_length_category'].value_counts().sort_index()
    
    bars = axes[1, 1].bar(range(len(length_counts)), length_counts.values, 
                         color=['#00b894', '#00cec9', '#0984e3', '#6c5ce7', '#a29bfe'])
    axes[1, 1].set_xticks(range(len(length_counts)))
    axes[1, 1].set_xticklabels(length_counts.index, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Number of Videos')
    axes[1, 1].set_title('Caption Length Distribution', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Viral coefficient (engagement rate distribution)
    df['viral_score'] = (df['like_count'] + df['comment_count'] * 5 + df['share_count'] * 10) / df['play_count'] * 100
    df['viral_score'] = df['viral_score'].fillna(0).replace([np.inf, -np.inf], 0)
    
    axes[1, 2].hist(df['viral_score'], bins=20, color='#fd63a8', alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(df['viral_score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["viral_score"].mean():.2f}%')
    axes[1, 2].set_xlabel('Viral Score (%)')
    axes[1, 2].set_ylabel('Number of Videos')
    axes[1, 2].set_title('Viral Score Distribution', fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Get base64 and save temp file
    img_base64, temp_path = plot_to_base64_and_file(fig, temp_dir, "trending_insights.png")
    
    # Create Gemini analysis
    insights_prompt = f"""
    As a data scientist and social media analytics expert, analyze this comprehensive TikTok trending insights dashboard showing various performance metrics and patterns.

    Please provide a detailed analysis that includes:

    - **Video Duration Strategy Analysis**: Examine the duration distribution:
      - Optimal video lengths for engagement
      - Platform algorithm preferences for duration
      - Audience attention span patterns
      - Content type vs duration correlations
    - **Engagement-Views Correlation Insights**: Analyze the scatter plot:
      - Linear vs exponential engagement relationships
      - High-performing outliers and what makes them unique
      - Engagement efficiency patterns (high engagement per view)
      - Comment-driven content characteristics
    - **Music and Audio Trends**: Evaluate popular audio usage:
      - Trending sounds and their impact on virality
      - Music-driven content strategy opportunities
      - Audio selection best practices
      - Platform culture around sound usage
    - **Engagement Metrics Performance**: Compare likes, comments, and shares:
      - Which metrics indicate true viral potential
      - Engagement type preferences of the TikTok audience
      - Content types that drive different engagement behaviors
      - Optimization strategies for each metric type
    - **Content Length Optimization**: Analyze caption length distribution:
      - Optimal caption lengths for different content types
      - Information density vs engagement correlation
      - Storytelling approaches that work best
      - SEO and discoverability through caption optimization
    - **Virality Patterns**: Examine the viral score distribution:
      - Characteristics of highly viral content
      - Viral coefficient benchmarks and targets
      - Content elements that boost viral potential
      - Algorithm behavior patterns
    - **Strategic Content Recommendations**: Based on all insights:
      - Content creation best practices
      - Timing and optimization strategies
      - Audience engagement maximization techniques
      - Platform algorithm optimization tips
    - **Performance Benchmarking**: What creators and brands should aim for:
      - Realistic engagement rate targets
      - Content performance KPIs
      - Growth trajectory indicators

    Focus on actionable insights that can drive content strategy and performance improvement.

    Important: 
    - Start directly with the analysis
    - Be professional and data-driven in your response
    - Use markdown formatting for better readability
    """
    
    gemini_analysis = analyze_with_gemini(temp_path, insights_prompt)
    
    return img_base64, gemini_analysis

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main TikTok dashboard HTML"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TikTok Trending Analysis Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body {{
                background: linear-gradient(135deg, #ff0050 0%, #000000 100%);
                font-family: 'Arial', sans-serif;
                min-height: 100vh;
                color: #333;
            }}
            .dashboard-container {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
                margin: 20px auto;
                padding: 30px;
                max-width: 1500px;
            }}
            .header {{
                background: linear-gradient(135deg, #ff0050 0%, #fe2c55 100%);
                color: white;
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .header h1 {{
                font-size: 2.5rem;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .control-panel {{
                background: #f8f9fa;
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 30px;
                border: 2px solid #e9ecef;
                text-align: center;
            }}
            .analysis-grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 30px;
                margin-top: 30px;
            }}
            .analysis-section {{
                background: white;
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
                border: 1px solid #e9ecef;
            }}
            .analysis-section h4 {{
                color: #495057;
                border-bottom: 3px solid #ff0050;
                padding-bottom: 15px;
                margin-bottom: 25px;
                font-size: 1.4rem;
                font-weight: bold;
            }}
            .plot-and-analysis {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 25px;
                align-items: start;
            }}
            .plot-container {{
                background: #f8f9fa;
                border-radius: 10px;
                padding: 15px;
                border: 1px solid #dee2e6;
            }}
            .plot-image {{
                width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .ai-analysis {{
                background: #ffffff;
                border: 2px solid #ff0050;
                border-radius: 10px;
                padding: 20px;
                max-height: 600px;
                overflow-y: auto;
            }}
            .ai-analysis h6 {{
                color: #ff0050;
                margin-bottom: 15px;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .ai-analysis-content {{
                line-height: 1.6;
                color: #495057;
            }}
            .loading-spinner {{
                display: none;
                text-align: center;
                padding: 80px;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            }}
            .btn-analyze {{
                background: linear-gradient(135deg, #ff0050 0%, #fe2c55 100%);
                border: none;
                padding: 15px 40px;
                border-radius: 30px;
                color: white;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1px;
                transition: all 0.3s ease;
                font-size: 1.1rem;
            }}
            .btn-analyze:hover {{
                transform: translateY(-3px);
                box-shadow: 0 15px 30px rgba(255, 0, 80, 0.4);
                color: white;
                background: linear-gradient(135deg, #fe2c55 0%, #ff0050 100%);
            }}
            .stats-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 20px;
            }}
            .stats-number {{
                font-size: 2rem;
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .tiktok-icon {{
                color: #ff0050;
                font-size: 1.5rem;
                margin-right: 10px;
            }}
            @media (max-width: 768px) {{
                .plot-and-analysis {{
                    grid-template-columns: 1fr;
                }}
                .header h1 {{
                    font-size: 1.8rem;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <div class="dashboard-container">
                <div class="header">
                    <h1><i class="fab fa-tiktok tiktok-icon"></i>TikTok Trending Analysis Dashboard</h1>
                    <p class="mb-0 fs-5">AI-Powered Analysis of Top 20 Trending Videos with Word Cloud Insights</p>
                </div>
                
                <div class="control-panel">
                    <h3><i class="fas fa-chart-line"></i> Automatic TikTok Trending Analysis</h3>
                    <p class="mb-3">Click the button below to analyze the top 20 trending TikTok videos from our database</p>
                    <button class="btn btn-analyze" id="analyzeBtn" onclick="analyzeTikTok()">
                        <i class="fas fa-play"></i> Analyze Trending Videos
                    </button>
                </div>
                
                <div class="loading-spinner" id="loadingSpinner">
                    <div class="spinner-border text-danger" style="width: 4rem; height: 4rem;" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <h4 class="mt-4" style="color: #ff0050;">Analyzing TikTok Data...</h4>
                    <p>Generating visualizations and AI insights, please wait...</p>
                    <div class="progress mt-3" style="height: 8px;">
                        <div class="progress-bar progress-bar-striped progress-bar-animated bg-danger" 
                             role="progressbar" style="width: 100%"></div>
                    </div>
                </div>
                
                <div id="analysisResults" style="display: none;">
                    <!-- Stats Overview -->
                    <div class="row mb-4" id="statsOverview" style="display: none;">
                        <div class="col-md-3">
                            <div class="stats-card">
                                <div class="stats-number" id="totalVideos">0</div>
                                <div>Total Videos</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stats-card">
                                <div class="stats-number" id="totalLikes">0</div>
                                <div>Total Likes</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stats-card">
                                <div class="stats-number" id="totalViews">0</div>
                                <div>Total Views</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stats-card">
                                <div class="stats-number" id="avgEngagement">0%</div>
                                <div>Avg Engagement</div>
                            </div>
                        </div>
                    </div>

                    <div class="analysis-grid">
                        <!-- Top Videos Analysis -->
                        <div class="analysis-section">
                            <h4><i class="fas fa-trophy"></i> Top Videos Engagement Analysis</h4>
                            <div class="plot-and-analysis">
                                <div class="plot-container">
                                    <img id="topVideosPlot" class="plot-image" alt="Top Videos Analysis">
                                </div>
                                <div class="ai-analysis">
                                    <h6><i class="fas fa-robot"></i> AI Analysis</h6>
                                    <div id="topVideosAnalysis" class="ai-analysis-content">
                                        Loading AI insights...
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Word Cloud Analysis -->
                        <div class="analysis-section">
                            <h4><i class="fas fa-cloud"></i> Content Word Cloud & Keywords Analysis</h4>
                            <div class="plot-and-analysis">
                                <div class="plot-container">
                                    <img id="wordcloudPlot" class="plot-image" alt="Word Cloud Analysis">
                                </div>
                                <div class="ai-analysis">
                                    <h6><i class="fas fa-robot"></i> AI Analysis</h6>
                                    <div id="wordcloudAnalysis" class="ai-analysis-content">
                                        Loading AI insights...
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Trending Insights Analysis -->
                        <div class="analysis-section">
                            <h4><i class="fas fa-chart-bar"></i> Trending Insights & Performance Metrics</h4>
                            <div class="plot-and-analysis">
                                <div class="plot-container">
                                    <img id="trendingPlot" class="plot-image" alt="Trending Insights">
                                </div>
                                <div class="ai-analysis">
                                    <h6><i class="fas fa-robot"></i> AI Analysis</h6>
                                    <div id="trendingAnalysis" class="ai-analysis-content">
                                        Loading AI insights...
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            function formatNumber(num) {{
                if (num >= 1000000) {{
                    return (num / 1000000).toFixed(1) + 'M';
                }} else if (num >= 1000) {{
                    return (num / 1000).toFixed(1) + 'K';
                }} else {{
                    return num.toLocaleString();
                }}
            }}

            async function analyzeTikTok() {{
                // Show loading spinner
                document.getElementById('loadingSpinner').style.display = 'block';
                document.getElementById('analysisResults').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = true;
                
                try {{
                    const response = await fetch('/analyze');
                    const data = await response.json();
                    
                    if (response.ok) {{
                        // Display stats
                        document.getElementById('totalVideos').textContent = data.total_videos;
                        document.getElementById('totalLikes').textContent = formatNumber(data.total_likes);
                        document.getElementById('totalViews').textContent = formatNumber(data.total_views);
                        document.getElementById('avgEngagement').textContent = data.avg_engagement.toFixed(1) + '%';
                        document.getElementById('statsOverview').style.display = 'flex';
                        
                        // Display the plots
                        document.getElementById('topVideosPlot').src = 'data:image/png;base64,' + data.top_videos_plot;
                        document.getElementById('wordcloudPlot').src = 'data:image/png;base64,' + data.wordcloud_plot;
                        document.getElementById('trendingPlot').src = 'data:image/png;base64,' + data.trending_plot;
                        
                        // Display AI analysis
                        document.getElementById('topVideosAnalysis').innerHTML = data.top_videos_analysis;
                        document.getElementById('wordcloudAnalysis').innerHTML = data.wordcloud_analysis;
                        document.getElementById('trendingAnalysis').innerHTML = data.trending_analysis;
                        
                        // Show results
                        document.getElementById('analysisResults').style.display = 'block';
                    }} else {{
                        alert('Error: ' + data.detail);
                    }}
                }} catch (error) {{
                    alert('Network error: ' + error.message);
                }} finally {{
                    // Hide loading spinner
                    document.getElementById('loadingSpinner').style.display = 'none';
                    document.getElementById('analyzeBtn').disabled = false;
                }}
            }}
            
            // Auto-load on page load
            window.addEventListener('load', function() {{
                // Optional: uncomment to auto-load on page load
                // analyzeTikTok();
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/analyze")
async def analyze_tiktok():
    """API endpoint to analyze TikTok trending data and return visualizations with AI insights"""
    temp_dir = None
    try:
        # Get TikTok data
        df = get_tiktok_data(limit=20)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No TikTok data found in database")
        
        # Calculate summary stats
        total_videos = len(df)
        total_likes = df['like_count'].sum()
        total_views = df['play_count'].sum()
        total_engagement = df['like_count'].sum() + df['comment_count'].sum() + df['share_count'].sum()
        avg_engagement = (total_engagement / total_views * 100) if total_views > 0 else 0
        
        # Create temporary directory for plot files
        temp_dir = tempfile.mkdtemp()
        logging.info(f"Created temporary directory: {temp_dir}")
        
        # Generate all plots with AI analysis
        top_videos_plot, top_videos_analysis = create_top_videos_plot(df, temp_dir)
        wordcloud_plot, wordcloud_analysis = create_wordcloud_plot(df, temp_dir)
        trending_plot, trending_analysis = create_trending_insights_plot(df, temp_dir)
        
        return {
            "total_videos": total_videos,
            "total_likes": int(total_likes),
            "total_views": int(total_views),
            "total_engagement": int(total_engagement),
            "avg_engagement": float(avg_engagement),
            "top_videos_plot": top_videos_plot,
            "top_videos_analysis": top_videos_analysis,
            "wordcloud_plot": wordcloud_plot,
            "wordcloud_analysis": wordcloud_analysis,
            "trending_plot": trending_plot,
            "trending_analysis": trending_analysis,
            "data_summary": {
                "date_range": {
                    "latest_scrape": str(df['scrape_date'].max()) if 'scrape_date' in df.columns else "Unknown",
                    "earliest_scrape": str(df['scrape_date'].min()) if 'scrape_date' in df.columns else "Unknown"
                },
                "top_creator": df.loc[df['like_count'].idxmax(), 'username'] if not df.empty else "Unknown",
                "most_liked_video": int(df['like_count'].max()) if not df.empty else 0
            }
        }
    
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logging.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                logging.error(f"Error cleaning up temp dir: {cleanup_error}")

@app.get("/api/stats")
async def get_tiktok_stats():
    """Get TikTok data statistics"""
    try:
        df = get_tiktok_data(limit=50)  # Get more data for stats
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No TikTok data found")
        
        stats = {
            "total_videos": len(df),
            "total_likes": int(df['like_count'].sum()),
            "total_comments": int(df['comment_count'].sum()),
            "total_shares": int(df['share_count'].sum()),
            "total_views": int(df['play_count'].sum()),
            "avg_likes": float(df['like_count'].mean()),
            "avg_comments": float(df['comment_count'].mean()),
            "avg_shares": float(df['share_count'].mean()),
            "avg_views": float(df['play_count'].mean()),
            "top_creators": df.groupby('username')['like_count'].sum().nlargest(5).to_dict(),
            "latest_scrape_date": str(df['scrape_date'].max()) if 'scrape_date' in df.columns else "Unknown"
        }
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting TikTok Trending Analysis Dashboard...")
    print("📱 Dashboard will be available at: http://localhost:8001")
    print("🔍 Database connection: PostgreSQL")
    print("📊 Features: Top Videos Analysis, Word Cloud, Trending Insights, AI Analysis")
    print("🤖 AI Analysis powered by Google Gemini")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)