#!/usr/bin/env python3
"""
Stock Analysis Dashboard - FastAPI Backend with Frontend with AI Analysis
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Time series analysis
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

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
GEMINI_API_KEY = "******************************************"
GEMINI_MODEL = "gemini-2.0-flash"

logging.info("Configuring Google Gemini with API key")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

app = FastAPI(title="Stock Analysis Dashboard", description="AI-Powered Time Series Analysis Dashboard for Stock Data")

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'appdb',
    'user': 'appuser',
    'password': '********************'
}

def get_database_engine():
    """Create database engine"""
    return create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")

def get_stock_data(symbol: str, years: int = 3) -> pd.DataFrame:
    """Retrieve stock data for the last N years"""
    engine = get_database_engine()
    
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365 * years)
    
    query = """
    SELECT "Date", "Open", "High", "Low", "Close", "Volume", "Symbol"
    FROM stock_data 
    WHERE "Symbol" = %(symbol)s 
    AND "Date" >= %(start_date)s 
    AND "Date" <= %(end_date)s
    ORDER BY "Date" ASC
    """
    
    try:
        df = pd.read_sql(query, engine, params={'symbol': symbol, 'start_date': start_date, 'end_date': end_date})
        engine.dispose()
        
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Convert Date to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        return df
    except Exception as e:
        engine.dispose()
        raise Exception(f"Database error: {str(e)}")

def plot_to_base64_and_file(fig, temp_dir: str, filename: str) -> tuple[str, str]:
    """Convert matplotlib figure to base64 string and save as temporary file"""
    # Save as base64
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Save as temporary file for Gemini
    temp_path = os.path.join(temp_dir, filename)
    fig.savefig(temp_path, format='png', dpi=100, bbox_inches='tight')
    
    plt.close(fig)
    return img_base64, temp_path

def analyze_with_gemini(image_path: str, prompt: str) -> str:
    """Analyze image using Google Gemini AI"""
    try:
        logging.info(f"Generating Gemini analysis for {image_path}")
        img = Image.open(image_path)
        
        response = model.generate_content(
            [prompt, img],
            generation_config={"temperature": 0},
        ).text
        
        # Convert to HTML markdown
        html_response = markdown.markdown(response)
        return html_response
        
    except Exception as e:
        logging.error(f"Gemini analysis error: {str(e)}")
        return f"<p><strong>AI Analysis Error:</strong> {str(e)}</p>"

def create_autocorrelation_plot(df: pd.DataFrame, symbol: str, temp_dir: str) -> tuple[str, str]:
    """Create autocorrelation plot using Close prices"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate autocorrelation for Close prices
    close_prices = df['Close'].dropna()
    
    # Plot ACF
    plot_acf(close_prices, lags=40, ax=ax, alpha=0.05)
    ax.set_title(f'Autocorrelation Function - {symbol} (Close Prices)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Lags', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Get base64 and save temp file
    img_base64, temp_path = plot_to_base64_and_file(fig, temp_dir, f"autocorr_{symbol}.png")
    
    # Create Gemini analysis
    autocorr_prompt = f"""
    As a financial analyst specializing in time series analysis, analyze this autocorrelation function (ACF) plot for {symbol} stock. 

    Please provide a comprehensive analysis that includes:

    - **Autocorrelation Patterns**: Describe the correlation structure and what it reveals about the stock's price behavior
    - **Statistical Significance**: Identify lags that show significant autocorrelation (outside the confidence bands)
    - **Time Series Properties**: Assess whether the series shows random walk behavior, mean reversion, or other patterns
    - **Trading Implications**: What does this autocorrelation structure suggest for:
      - Market efficiency for this stock
      - Potential predictability of future price movements
      - Risk management considerations
    - **Technical Analysis Insights**: How might technical analysts interpret these patterns
    - **Recommended Actions**: Based on these findings, what analytical approaches or trading strategies might be most appropriate

    Focus on practical insights that would be valuable for investors, traders, and portfolio managers.

    Important: 
    - Start directly with the analysis
    - Be professional and technical in your response
    - Use markdown formatting for better readability
    """
    
    gemini_analysis = analyze_with_gemini(temp_path, autocorr_prompt)
    
    return img_base64, gemini_analysis

def create_differenced_plot(df: pd.DataFrame, symbol: str, temp_dir: str) -> tuple[str, str]:
    """Create differenced plot to show stationarity"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    close_prices = df['Close'].dropna()
    
    # Original series
    axes[0].plot(close_prices.index, close_prices.values, color='blue', linewidth=1.5)
    axes[0].set_title(f'{symbol} - Original Close Prices', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price ($)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # First difference
    diff_prices = close_prices.diff().dropna()
    axes[1].plot(diff_prices.index, diff_prices.values, color='red', linewidth=1.5)
    axes[1].set_title(f'{symbol} - First Difference (Close Prices)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Price Difference ($)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Get base64 and save temp file
    img_base64, temp_path = plot_to_base64_and_file(fig, temp_dir, f"diff_{symbol}.png")
    
    # Create Gemini analysis
    diff_prompt = f"""
    As a quantitative analyst specializing in time series modeling, analyze this differencing plot for {symbol} stock which shows both the original price series and its first difference.

    Please provide a comprehensive analysis that includes:

    - **Stationarity Assessment**: Compare the original series vs the differenced series in terms of:
      - Trend behavior (upward/downward/stationary)
      - Variance stability over time
      - Mean reversion properties
    - **Statistical Properties**: Evaluate the transformation effectiveness:
      - Has differencing successfully removed trends?
      - Are there any remaining patterns in the differenced series?
      - Evidence of seasonality or cyclical behavior
    - **Market Behavior Insights**: What do these patterns reveal about:
      - Price discovery mechanisms for this stock
      - Volatility clustering or heteroskedasticity
      - Market regime changes or structural breaks
    - **Modeling Implications**: Based on this analysis:
      - Appropriateness for ARIMA or other time series models
      - Further preprocessing requirements
      - Potential forecasting challenges or opportunities
    - **Risk Assessment**: What does the differenced series suggest about:
      - Price volatility characteristics
      - Extreme movement frequency
      - Potential for mean reversion
    - **Investment Strategy**: How might this inform trading decisions and portfolio management

    Focus on actionable insights for quantitative trading and risk management applications.

    Important: 
    - Start directly with the analysis
    - Be professional and technical in your response
    - Use markdown formatting for better readability
    """
    
    gemini_analysis = analyze_with_gemini(temp_path, diff_prompt)
    
    return img_base64, gemini_analysis

def create_moving_average_plot(df: pd.DataFrame, symbol: str, temp_dir: str) -> tuple[str, str]:
    """Create moving average plot with multiple periods"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    close_prices = df['Close'].dropna()
    
    # Calculate moving averages
    ma_20 = close_prices.rolling(window=20).mean()
    ma_50 = close_prices.rolling(window=50).mean()
    ma_200 = close_prices.rolling(window=200).mean()
    
    # Plot the data
    ax.plot(close_prices.index, close_prices.values, label='Close Price', color='black', linewidth=1, alpha=0.7)
    ax.plot(ma_20.index, ma_20.values, label='MA 20', color='blue', linewidth=2)
    ax.plot(ma_50.index, ma_50.values, label='MA 50', color='orange', linewidth=2)
    ax.plot(ma_200.index, ma_200.values, label='MA 200', color='red', linewidth=2)
    
    ax.set_title(f'{symbol} - Moving Averages Analysis', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Get base64 and save temp file
    img_base64, temp_path = plot_to_base64_and_file(fig, temp_dir, f"ma_{symbol}.png")
    
    # Create Gemini analysis
    ma_prompt = f"""
    As a technical analysis expert and portfolio manager, analyze this moving average chart for {symbol} stock showing the price action with 20-day, 50-day, and 200-day moving averages.

    Please provide a comprehensive technical analysis that includes:

    - **Trend Analysis**: Evaluate the overall trend and momentum by examining:
      - Relationship between price and different MA periods
      - Moving average convergence/divergence patterns
      - Trend strength and sustainability indicators
    - **Support and Resistance Levels**: Identify where moving averages act as:
      - Dynamic support during uptrends
      - Dynamic resistance during downtrends
      - Key breakout or breakdown levels
    - **Trading Signals**: Analyze current and historical signals such as:
      - Golden crosses (shorter MA above longer MA) and death crosses
      - Price crossovers above/below key MAs
      - MA slope directions and convergence patterns
    - **Market Regime Identification**: Determine current market phase:
      - Bull market (price above ascending MAs)
      - Bear market (price below descending MAs)
      - Sideways/consolidation periods
    - **Risk Management Insights**: How the MA structure informs:
      - Stop-loss placement strategies
      - Position sizing considerations
      - Entry and exit timing optimization
    - **Investment Strategy Recommendations**: Based on the MA analysis:
      - Short-term trading opportunities
      - Long-term investment positioning
      - Portfolio allocation adjustments
    - **Market Psychology**: What the MA relationships reveal about investor sentiment and institutional behavior

    Focus on actionable insights for both active traders and long-term investors.

    Important: 
    - Start directly with the analysis
    - Be professional and technical in your response
    - Use markdown formatting for better readability
    """
    
    gemini_analysis = analyze_with_gemini(temp_path, ma_prompt)
    
    return img_base64, gemini_analysis

def create_high_plot(df: pd.DataFrame, symbol: str, temp_dir: str) -> tuple[str, str]:
    """Create line chart for High prices"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    high_prices = df['High'].dropna()
    
    # Create line plot
    ax.plot(high_prices.index, high_prices.values, color='green', linewidth=2, alpha=0.8)
    
    # Fill area under the curve
    ax.fill_between(high_prices.index, high_prices.values, alpha=0.3, color='green')
    
    # Add trend line
    x_numeric = np.arange(len(high_prices))
    z = np.polyfit(x_numeric, high_prices.values, 1)
    p = np.poly1d(z)
    ax.plot(high_prices.index, p(x_numeric), "--", color='red', alpha=0.8, linewidth=2, label='Trend Line')
    
    ax.set_title(f'{symbol} - High Prices Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('High Price ($)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Add statistics box
    stats_text = f'Max: ${high_prices.max():.2f}\nMin: ${high_prices.min():.2f}\nMean: ${high_prices.mean():.2f}\nStd: ${high_prices.std():.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Get base64 and save temp file
    img_base64, temp_path = plot_to_base64_and_file(fig, temp_dir, f"high_{symbol}.png")
    
    # Create Gemini analysis
    high_prompt = f"""
    As a market analyst specializing in price action and momentum analysis, examine this high price chart for {symbol} stock which shows the daily high prices over time with trend line and key statistics.

    Please provide a comprehensive analysis that includes:

    - **Price Momentum Analysis**: Evaluate the high price progression:
      - Overall trend direction and strength (using the trend line)
      - Momentum acceleration or deceleration phases
      - Breakout patterns and new high formations
    - **Resistance Level Mapping**: Identify key resistance zones:
      - Historical high levels that acted as resistance
      - Current resistance challenges and breakthrough potential
      - Psychological price barriers (round numbers, all-time highs)
    - **Volatility Assessment**: Analyze price expansion characteristics:
      - High-to-high volatility patterns
      - Periods of range expansion vs contraction
      - Market stress and euphoria indicators
    - **Market Structure Analysis**: Evaluate the price architecture:
      - Higher highs progression in uptrends
      - Failure to make new highs as warning signals
      - Distribution patterns at major tops
    - **Statistical Insights**: Interpret the statistical data shown:
      - Significance of current levels vs historical ranges
      - Standard deviation implications for future moves
      - Mean reversion vs trend continuation probabilities
    - **Trading and Investment Implications**: Based on high price analysis:
      - Optimal entry points for long positions
      - Profit-taking zones and target levels
      - Risk management using high price patterns
    - **Market Sentiment Indicators**: What high price action reveals about:
      - Institutional accumulation or distribution
      - Retail investor behavior and FOMO patterns
      - Market cycle positioning

    Focus on actionable insights for momentum trading, breakout strategies, and risk management.

    Important: 
    - Start directly with the analysis
    - Be professional and technical in your response
    - Use markdown formatting for better readability
    """
    
    gemini_analysis = analyze_with_gemini(temp_path, high_prompt)
    
    return img_base64, gemini_analysis

def get_available_symbols() -> list:
    """Get list of available stock symbols"""
    engine = get_database_engine()
    try:
        query = 'SELECT DISTINCT "Symbol" FROM stock_data ORDER BY "Symbol"'
        result = engine.execute(text(query))
        symbols = [row[0] for row in result]
        engine.dispose()
        return symbols
    except Exception as e:
        engine.dispose()
        return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY']  # Fallback

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard HTML"""
    symbols = get_available_symbols()
    symbols_options = ''.join([f'<option value="{symbol}">{symbol}</option>' for symbol in symbols])
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Analysis Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                font-family: 'Arial', sans-serif;
                min-height: 100vh;
            }}
            .dashboard-container {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                margin: 20px auto;
                padding: 30px;
                max-width: 1400px;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .control-panel {{
                background: #f8f9fa;
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 30px;
                border: 2px solid #e9ecef;
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
                border-bottom: 3px solid #667eea;
                padding-bottom: 15px;
                margin-bottom: 25px;
                font-size: 1.3rem;
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
            }}
            .ai-analysis {{
                background: #ffffff;
                border: 2px solid #667eea;
                border-radius: 10px;
                padding: 20px;
                max-height: 600px;
                overflow-y: auto;
            }}
            .ai-analysis h6 {{
                color: #667eea;
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
                padding: 50px;
            }}
            .btn-analyze {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                padding: 12px 30px;
                border-radius: 25px;
                color: white;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1px;
                transition: all 0.3s ease;
            }}
            .btn-analyze:hover {{
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
                color: white;
            }}
            .stock-selector {{
                border: 2px solid #e9ecef;
                border-radius: 10px;
                padding: 10px;
            }}
            .stock-selector:focus {{
                border-color: #667eea;
                box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
            }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <div class="dashboard-container">
                <div class="header">
                    <h1><i class="fas fa-chart-line"></i> AI-Powered Stock Analysis Dashboard</h1>
                    <p class="mb-0">Advanced Time Series Analysis with Google Gemini AI Insights</p>
                </div>
                
                <div class="control-panel">
                    <div class="row align-items-center">
                        <div class="col-md-4">
                            <label for="stockSelector" class="form-label fw-bold">
                                <i class="fas fa-search"></i> Select Stock Symbol:
                            </label>
                            <select class="form-select stock-selector" id="stockSelector">
                                <option value="">Choose a stock...</option>
                                {symbols_options}
                            </select>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label fw-bold">
                                <i class="fas fa-calendar"></i> Analysis Period:
                            </label>
                            <div class="form-control-plaintext">Last 3 Years</div>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label fw-bold">&nbsp;</label>
                            <div>
                                <button class="btn btn-analyze" id="analyzeBtn" onclick="analyzeStock()">
                                    <i class="fas fa-analytics"></i> Analyze Stock
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="loading-spinner" id="loadingSpinner">
                    <div class="spinner-border text-primary" style="width: 3rem; height: 3rem;" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <h4 class="mt-3">Analyzing Stock Data...</h4>
                    <p>Generating visualizations, please wait...</p>
                </div>
                
                <div id="analysisResults" style="display: none;">
                    <div class="analysis-grid">
                        <!-- Autocorrelation Analysis -->
                        <div class="analysis-section">
                            <h4><i class="fas fa-wave-square"></i> Autocorrelation Analysis</h4>
                            <div class="plot-and-analysis">
                                <div class="plot-container">
                                    <img id="autocorrPlot" class="plot-image" alt="Autocorrelation Plot">
                                </div>
                                <div class="ai-analysis">
                                    <h6><i class="fas fa-robot"></i> AI Analysis</h6>
                                    <div id="autocorrAnalysis" class="ai-analysis-content">
                                        Loading AI insights...
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Differenced Series Analysis -->
                        <div class="analysis-section">
                            <h4><i class="fas fa-chart-area"></i> Differenced Series Analysis</h4>
                            <div class="plot-and-analysis">
                                <div class="plot-container">
                                    <img id="diffPlot" class="plot-image" alt="Differenced Plot">
                                </div>
                                <div class="ai-analysis">
                                    <h6><i class="fas fa-robot"></i> AI Analysis</h6>
                                    <div id="diffAnalysis" class="ai-analysis-content">
                                        Loading AI insights...
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Moving Averages Analysis -->
                        <div class="analysis-section">
                            <h4><i class="fas fa-chart-line"></i> Moving Averages Analysis</h4>
                            <div class="plot-and-analysis">
                                <div class="plot-container">
                                    <img id="movingAvgPlot" class="plot-image" alt="Moving Average Plot">
                                </div>
                                <div class="ai-analysis">
                                    <h6><i class="fas fa-robot"></i> AI Analysis</h6>
                                    <div id="maAnalysis" class="ai-analysis-content">
                                        Loading AI insights...
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- High Prices Analysis -->
                        <div class="analysis-section">
                            <h4><i class="fas fa-arrow-trend-up"></i> High Prices Trend Analysis</h4>
                            <div class="plot-and-analysis">
                                <div class="plot-container">
                                    <img id="highPlot" class="plot-image" alt="High Plot">
                                </div>
                                <div class="ai-analysis">
                                    <h6><i class="fas fa-robot"></i> AI Analysis</h6>
                                    <div id="highAnalysis" class="ai-analysis-content">
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
            async function analyzeStock() {{
                const symbol = document.getElementById('stockSelector').value;
                
                if (!symbol) {{
                    alert('Please select a stock symbol');
                    return;
                }}
                
                // Show loading spinner
                document.getElementById('loadingSpinner').style.display = 'block';
                document.getElementById('analysisResults').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = true;
                
                try {{
                    const response = await fetch(`/analyze/${{symbol}}`);
                    const data = await response.json();
                    
                    if (response.ok) {{
                        // Display the plots
                        document.getElementById('autocorrPlot').src = 'data:image/png;base64,' + data.autocorr_plot;
                        document.getElementById('diffPlot').src = 'data:image/png;base64,' + data.diff_plot;
                        document.getElementById('movingAvgPlot').src = 'data:image/png;base64,' + data.ma_plot;
                        document.getElementById('highPlot').src = 'data:image/png;base64,' + data.high_plot;
                        
                        // Display AI analysis
                        document.getElementById('autocorrAnalysis').innerHTML = data.autocorr_analysis;
                        document.getElementById('diffAnalysis').innerHTML = data.diff_analysis;
                        document.getElementById('maAnalysis').innerHTML = data.ma_analysis;
                        document.getElementById('highAnalysis').innerHTML = data.high_analysis;
                        
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
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/analyze/{symbol}")
async def analyze_stock(symbol: str):
    """API endpoint to analyze stock and return visualizations with AI insights"""
    temp_dir = None
    try:
        # Get stock data
        df = get_stock_data(symbol.upper(), years=3)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Create temporary directory for plot files
        temp_dir = tempfile.mkdtemp()
        logging.info(f"Created temporary directory: {temp_dir}")
        
        # Generate all plots with AI analysis
        autocorr_plot, autocorr_analysis = create_autocorrelation_plot(df, symbol.upper(), temp_dir)
        diff_plot, diff_analysis = create_differenced_plot(df, symbol.upper(), temp_dir)
        ma_plot, ma_analysis = create_moving_average_plot(df, symbol.upper(), temp_dir)
        high_plot, high_analysis = create_high_plot(df, symbol.upper(), temp_dir)
        
        return {
            "symbol": symbol.upper(),
            "data_points": len(df),
            "date_range": {
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date())
            },
            "autocorr_plot": autocorr_plot,
            "autocorr_analysis": autocorr_analysis,
            "diff_plot": diff_plot,
            "diff_analysis": diff_analysis,
            "ma_plot": ma_plot,
            "ma_analysis": ma_analysis,
            "high_plot": high_plot,
            "high_analysis": high_analysis
        }
    
    except Exception as e:
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

@app.get("/api/symbols")
async def get_symbols():
    """Get available stock symbols"""
    try:
        symbols = get_available_symbols()
        return {"symbols": symbols}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Stock Analysis Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8000")
    print("🔍 Database connection: PostgreSQL")
    print("📈 Features: Autocorrelation, Differencing, Moving Averages, High Prices")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)