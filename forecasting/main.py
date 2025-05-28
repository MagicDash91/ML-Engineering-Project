from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import io
import json
import warnings
warnings.filterwarnings('ignore')
import logging
import os
from pathlib import Path
from datetime import datetime
from fastapi.staticfiles import StaticFiles

# Statistical and ML libraries
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from datetime import datetime, timedelta
import seaborn as sns
from joblib import Parallel, delayed
import multiprocessing
from functools import lru_cache
import hashlib

app = FastAPI(title="Sales Demand Forecasting with SARIMA")

# Get the absolute path to the static directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PLOTS_DIR = os.path.join(STATIC_DIR, "plots")

# Create directories if they don't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# Mount static directory with absolute path
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('forecasting.log'),
        logging.StreamHandler()
    ]
)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sales Demand Forecasting - SARIMA</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin: 20px auto;
            padding: 30px;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: rgba(102, 126, 234, 0.05);
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            border-color: #764ba2;
            background: rgba(118, 75, 162, 0.05);
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 25px;
            padding: 12px 30px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        .card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }
        .metric-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            text-align: center;
            padding: 20px;
            border-radius: 15px;
            margin: 10px 0;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 10px 0;
        }
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .loading {
            display: none;
            text-align: center;
            padding: 50px;
        }
        .spinner-border {
            width: 3rem;
            height: 3rem;
            border-width: 0.3em;
        }
        .results-section {
            display: none;
        }
        .nav-pills .nav-link.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .forecast-highlight {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="main-container">
            <div class="text-center mb-5">
                <h1 class="display-4 fw-bold text-primary">
                    <i class="fas fa-chart-line me-3"></i>
                    Sales Demand Forecasting
                </h1>
                <p class="lead text-muted">Advanced SARIMA Time Series Analysis</p>
            </div>

            <!-- Upload Section -->
            <div class="upload-section">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title text-center mb-4">
                            <i class="fas fa-upload me-2"></i>Upload Your Data
                        </h3>
                        <form id="uploadForm" enctype="multipart/form-data">
                            <div class="upload-area">
                                <i class="fas fa-cloud-upload-alt fa-3x text-primary mb-3"></i>
                                <h5>Drag & Drop your CSV file here</h5>
                                <p class="text-muted">or click to browse</p>
                                <input type="file" id="csvFile" name="file" accept=".csv" class="form-control" style="display: none;">
                            </div>
                            <div class="text-center mt-4">
                                <button type="button" class="btn btn-primary" onclick="document.getElementById('csvFile').click()">
                                    <i class="fas fa-folder-open me-2"></i>Choose File
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>

            <!-- Column Selection -->
            <div id="columnSelection" class="mt-4" style="display: none;">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">
                            <i class="fas fa-columns me-2"></i>Select Columns
                        </h3>
                        <form id="columnForm">
                            <div class="row">
                                <div class="col-md-6">
                                    <label for="dateColumn" class="form-label">Date Column</label>
                                    <select id="dateColumn" name="date_column" class="form-select" required>
                                        <option value="">Select date column...</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label for="targetColumn" class="form-label">Target Value Column</label>
                                    <select id="targetColumn" name="target_column" class="form-select" required>
                                        <option value="">Select target column...</option>
                                    </select>
                                </div>
                            </div>
                            <div class="text-center mt-4">
                                <button type="submit" class="btn btn-primary">
                                    <i class="fas fa-rocket me-2"></i>Start Analysis
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>

            <!-- Loading -->
            <div id="loading" class="loading">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <h4 class="mt-3">Analyzing your data...</h4>
                <p class="text-muted">This may take a few moments</p>
            </div>

            <!-- Results -->
            <div id="results" class="results-section">
                <!-- Navigation Tabs -->
                <ul class="nav nav-pills nav-justified mb-4" id="resultTabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="overview-tab" data-bs-toggle="pill" data-bs-target="#overview" type="button" role="tab">
                            <i class="fas fa-chart-bar me-2"></i>Overview
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="diagnostics-tab" data-bs-toggle="pill" data-bs-target="#diagnostics" type="button" role="tab">
                            <i class="fas fa-stethoscope me-2"></i>Diagnostics
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="forecast-tab" data-bs-toggle="pill" data-bs-target="#forecast" type="button" role="tab">
                            <i class="fas fa-crystal-ball me-2"></i>Forecast
                        </button>
                    </li>
                </ul>

                <div class="tab-content" id="resultTabsContent">
                    <!-- Overview Tab -->
                    <div class="tab-pane fade show active" id="overview" role="tabpanel">
                        <div class="row">
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <i class="fas fa-database fa-2x mb-2"></i>
                                    <div class="metric-value" id="dataPoints">-</div>
                                    <div>Data Points</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <i class="fas fa-chart-line fa-2x mb-2"></i>
                                    <div class="metric-value" id="maeValue">-</div>
                                    <div>MAE</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <i class="fas fa-bullseye fa-2x mb-2"></i>
                                    <div class="metric-value" id="rmseValue">-</div>
                                    <div>RMSE</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <i class="fas fa-percentage fa-2x mb-2"></i>
                                    <div class="metric-value" id="mapeValue">-</div>
                                    <div>MAPE (%)</div>
                                </div>
                            </div>
                        </div>
                        <div class="chart-container">
                            <canvas id="overviewChart" height="100"></canvas>
                        </div>
                    </div>

                    <!-- Diagnostics Tab -->
                    <div class="tab-pane fade" id="diagnostics" role="tabpanel">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <h5><i class="fas fa-wave-square me-2"></i>Autocorrelation Function</h5>
                                    <img id="acfPlot" class="img-fluid" alt="ACF Plot">
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <h5><i class="fas fa-wave-square me-2"></i>Partial Autocorrelation Function</h5>
                                    <img id="pacfPlot" class="img-fluid" alt="PACF Plot">
                                </div>
                            </div>
                        </div>
                        <div class="chart-container">
                            <h5><i class="fas fa-calendar-alt me-2"></i>Seasonal Decomposition</h5>
                            <img id="decompositionPlot" class="img-fluid" alt="Seasonal Decomposition">
                        </div>
                    </div>

                    <!-- Forecast Tab -->
                    <div class="tab-pane fade" id="forecast" role="tabpanel">
                        <div class="forecast-highlight">
                            <h4><i class="fas fa-crystal-ball me-2"></i>30-Day Forecast Results</h4>
                            <p>SARIMA model predictions with confidence intervals</p>
                        </div>
                        <div class="chart-container">
                            <canvas id="forecastChart" height="100"></canvas>
                        </div>
                        <div class="row mt-4">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-body">
                                        <h5><i class="fas fa-info-circle me-2"></i>Model Information</h5>
                                        <div id="modelInfo"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-body">
                                        <h5><i class="fas fa-download me-2"></i>Download Forecast</h5>
                                        <p class="text-muted">Export your forecast results</p>
                                        <button class="btn btn-outline-primary" onclick="downloadForecast()">
                                            <i class="fas fa-file-csv me-2"></i>Download CSV
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let forecastData = null;
        let plotPaths = null;

        // File upload handling
        document.getElementById('csvFile').addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                uploadFile(e.target.files[0]);
            }
        });

        // Drag and drop
        const uploadArea = document.querySelector('.upload-area');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = 'rgba(118, 75, 162, 0.1)';
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = 'rgba(102, 126, 234, 0.05)';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = 'rgba(102, 126, 234, 0.05)';
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                uploadFile(files[0]);
            }
        });

        async function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    populateColumnSelectors(data.columns);
                    document.getElementById('columnSelection').style.display = 'block';
                } else {
                    alert('Error uploading file. Please check the format.');
                }
            } catch (error) {
                console.error('Upload error:', error);
                alert('Error uploading file.');
            }
        }

        function populateColumnSelectors(columns) {
            const dateSelect = document.getElementById('dateColumn');
            const targetSelect = document.getElementById('targetColumn');

            dateSelect.innerHTML = '<option value="">Select date column...</option>';
            targetSelect.innerHTML = '<option value="">Select target column...</option>';

            columns.date_columns.forEach(col => {
                dateSelect.innerHTML += `<option value="${col}">${col}</option>`;
            });

            columns.numeric_columns.forEach(col => {
                targetSelect.innerHTML += `<option value="${col}">${col}</option>`;
            });
        }

        // Form submission
        document.getElementById('columnForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            document.getElementById('loading').style.display = 'block';
            document.getElementById('columnSelection').style.display = 'none';

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    displayResults(data);
                } else {
                    alert('Error analyzing data.');
                }
            } catch (error) {
                console.error('Analysis error:', error);
                alert('Error analyzing data.');
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });

        function displayResults(data) {
            if (!data) {
                console.error('No data received');
                alert('Error: No data received from server');
                return;
            }

            forecastData = data;
            plotPaths = data.plot_paths;
            
            try {
                // Update metrics with null checks
                const metrics = data.metrics || {};
                document.getElementById('dataPoints').textContent = metrics.data_points || 0;
                document.getElementById('maeValue').textContent = (metrics.mae || 0).toFixed(2);
                document.getElementById('rmseValue').textContent = (metrics.rmse || 0).toFixed(2);
                document.getElementById('mapeValue').textContent = (metrics.mape || 0).toFixed(2);

                // Display plots using static files with error handling
                if (plotPaths) {
                    try {
                        document.getElementById('acfPlot').src = '/static/' + (plotPaths.acf || '');
                        document.getElementById('pacfPlot').src = '/static/' + (plotPaths.pacf || '');
                        document.getElementById('decompositionPlot').src = '/static/' + (plotPaths.decomposition || '');
                    } catch (error) {
                        console.error('Error loading plots:', error);
                        // Fallback to base64 if static files fail
                        if (data.plots) {
                            document.getElementById('acfPlot').src = 'data:image/png;base64,' + (data.plots.acf || '');
                            document.getElementById('pacfPlot').src = 'data:image/png;base64,' + (data.plots.pacf || '');
                            document.getElementById('decompositionPlot').src = 'data:image/png;base64,' + (data.plots.decomposition || '');
                        }
                    }
                }

                // Model info with null checks
                const modelInfo = data.model_info || {};
                document.getElementById('modelInfo').innerHTML = `
                    <p><strong>Model:</strong> SARIMA${modelInfo.order || 'N/A'}</p>
                    <p><strong>AIC:</strong> ${(modelInfo.aic || 0).toFixed(2)}</p>
                    <p><strong>BIC:</strong> ${(modelInfo.bic || 0).toFixed(2)}</p>
                    <p><strong>Log Likelihood:</strong> ${(modelInfo.llf || 0).toFixed(2)}</p>
                `;

                // Create charts with null checks
                if (data.historical && data.fitted_values && data.forecast) {
                    createOverviewChart(data);
                    createForecastChart(data);
                } else {
                    console.error('Missing data for charts');
                }

                document.getElementById('results').style.display = 'block';
            } catch (error) {
                console.error('Error displaying results:', error);
                alert('Error displaying results. Please check the console for details.');
            }
        }

        function createOverviewChart(data) {
            try {
                const ctx = document.getElementById('overviewChart').getContext('2d');
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.historical.dates || [],
                        datasets: [{
                            label: 'Actual',
                            data: data.historical.values || [],
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            fill: true,
                            tension: 0.4
                        }, {
                            label: 'Fitted',
                            data: data.fitted_values || [],
                            borderColor: '#764ba2',
                            backgroundColor: 'rgba(118, 75, 162, 0.1)',
                            fill: false,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Historical Data vs Model Fit'
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: false
                            }
                        }
                    }
                });
            } catch (error) {
                console.error('Error creating overview chart:', error);
            }
        }

        function createForecastChart(data) {
            try {
                const ctx = document.getElementById('forecastChart').getContext('2d');
                
                const allDates = [...(data.historical.dates || []), ...(data.forecast.dates || [])];
                const historicalData = [...(data.historical.values || []), ...new Array(data.forecast.dates?.length || 0).fill(null)];
                const forecastData = [...new Array(data.historical.dates?.length || 0).fill(null), ...(data.forecast.values || [])];
                const upperBound = [...new Array(data.historical.dates?.length || 0).fill(null), ...(data.forecast.upper_bound || [])];
                const lowerBound = [...new Array(data.historical.dates?.length || 0).fill(null), ...(data.forecast.lower_bound || [])];

                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: allDates,
                        datasets: [{
                            label: 'Historical',
                            data: historicalData,
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            fill: false,
                            tension: 0.4
                        }, {
                            label: 'Forecast',
                            data: forecastData,
                            borderColor: '#f093fb',
                            backgroundColor: 'rgba(240, 147, 251, 0.2)',
                            fill: false,
                            tension: 0.4,
                            borderDash: [5, 5]
                        }, {
                            label: 'Upper Bound',
                            data: upperBound,
                            borderColor: '#4facfe',
                            backgroundColor: 'rgba(79, 172, 254, 0.1)',
                            fill: '+1',
                            tension: 0.4,
                            borderWidth: 1
                        }, {
                            label: 'Lower Bound',
                            data: lowerBound,
                            borderColor: '#4facfe',
                            backgroundColor: 'rgba(79, 172, 254, 0.1)',
                            fill: false,
                            tension: 0.4,
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: '30-Day Sales Forecast with Confidence Intervals'
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: false
                            }
                        }
                    }
                });
            } catch (error) {
                console.error('Error creating forecast chart:', error);
            }
        }

        function downloadForecast() {
            if (!forecastData) return;

            const csv = [
                ['Date', 'Forecast', 'Lower Bound', 'Upper Bound'],
                ...forecastData.forecast.dates.map((date, i) => [
                    date,
                    forecastData.forecast.values[i].toFixed(2),
                    forecastData.forecast.lower_bound[i].toFixed(2),
                    forecastData.forecast.upper_bound[i].toFixed(2)
                ])
            ].map(row => row.join(',')).join('\\n');

            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'sales_forecast.csv';
            a.click();
            window.URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>
"""

# Global variable to store uploaded data
uploaded_data = None

# Add this after the global uploaded_data declaration
model_cache = {}

def get_cache_key(ts, order, seasonal_order):
    """Generate a cache key for the model"""
    # Create a hash of the time series data and parameters
    data_hash = hashlib.md5(ts.values.tobytes()).hexdigest()
    param_str = f"{order}_{seasonal_order}"
    return f"{data_hash}_{param_str}"

@lru_cache(maxsize=32)
def fit_cached_model(ts_values, order, seasonal_order):
    """Cache and fit SARIMA model"""
    ts = pd.Series(ts_values, index=pd.date_range(start='2000-01-01', periods=len(ts_values)))
    if seasonal_order[3] > 0:
        model = SARIMAX(ts, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False)
    else:
        model = SARIMAX(ts, order=order,
                       enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)

@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    return HTML_TEMPLATE

def clean_data(df):
    logging.info("Starting data cleaning process")
    # Drop columns with only one unique value
    unique_value_columns = [col for col in df.columns if df[col].nunique() == 1]
    if len(unique_value_columns) > 0:
        logging.info(
            f"Dropping columns with only one unique value: {unique_value_columns}"
        )
    df.drop(columns=unique_value_columns, inplace=True)

    # Drop columns where the name contains "id", "number", or "phone"
    # Removed "date" from the list to preserve date columns
    id_cols = [
        col
        for col in df.columns
        if any(x in col.lower() for x in ["id", "number", "phone"])
    ]
    if len(id_cols) > 0:
        logging.info(f"Dropping ID/number/phone columns: {id_cols}")
    df.drop(columns=id_cols, inplace=True)

    # Keep datetime columns for time series analysis
    logging.info("Preserving datetime columns for time series analysis")

    logging.info(f"Data cleaning complete. Final dataframe shape: {df.shape}")
    return df

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global uploaded_data
    
    try:
        # Get file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        # Read the uploaded file content
        content = await file.read()
        
        if file_extension == ".csv":
            logging.info("Trying to read CSV file with common delimiters...")
            
            possible_delimiters = [",", ";", "\t", "|"]
            detected_delim = None
            
            # Try reading the file with common delimiters
            for delim in possible_delimiters:
                try:
                    df = pd.read_csv(io.StringIO(content.decode('utf-8')), delimiter=delim)
                    
                    # If only one column is detected, try another delimiter
                    if df.shape[1] > 1:
                        detected_delim = delim
                        logging.info(f"Successfully read CSV with delimiter: {repr(delim)}")
                        break
                except Exception as e:
                    logging.warning(f"Failed to read CSV with delimiter {repr(delim)}: {str(e)}")
                    continue
            
            # If no valid delimiter is found
            if detected_delim is None:
                raise HTTPException(
                    status_code=400,
                    detail="Could not determine delimiter or read CSV file."
                )
                
        elif file_extension == ".xlsx":
            logging.info("Reading Excel file...")
            try:
                df = pd.read_excel(io.BytesIO(content))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error reading Excel file: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file format. Please upload a CSV or Excel file."
            )
        
        # Clean the data
        df = clean_data(df)
        uploaded_data = df
        
        # Analyze columns
        date_columns = []
        numeric_columns = []
        
        for col in uploaded_data.columns:
            # Check for date columns
            if uploaded_data[col].dtype == 'datetime64[ns]':
                date_columns.append(col)
            elif uploaded_data[col].dtype == 'object':
                # Check if it contains dates or 4-digit years
                sample_values = uploaded_data[col].dropna().head(10).astype(str)
                if any(len(val) == 4 and val.isdigit() for val in sample_values):
                    date_columns.append(col)
                elif any(bool(pd.to_datetime(val, errors='coerce')) for val in sample_values):
                    date_columns.append(col)
            
            # Check for numeric columns
            if uploaded_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_columns.append(col)
        
        return {
            "status": "success",
            "columns": {
                "date_columns": date_columns,
                "numeric_columns": numeric_columns
            }
        }
    
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

def save_plot(fig, filename):
    """Save plot to static directory and return the path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(PLOTS_DIR, f"{filename}_{timestamp}.png")
    fig.savefig(filepath, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved plot to {filepath}")
    # Return relative path for web access
    return f"plots/{os.path.basename(filepath)}"

def create_plot_base64(fig, filename):
    """Convert matplotlib figure to base64 string and save to static directory"""
    # Save to static directory
    filepath = save_plot(fig, filename)
    
    # Also create base64 for immediate display
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64, filepath

def prepare_time_series(df, date_col, target_col):
    """Prepare time series data"""
    # Handle different date formats
    if df[date_col].dtype == 'object':
        # Try to parse dates
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    
    # Remove rows with invalid dates
    df = df.dropna(subset=[date_col, target_col])
    
    # Sort by date
    df = df.sort_values(date_col)
    
    # Set date as index
    df.set_index(date_col, inplace=True)
    
    # Ensure target is numeric
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    
    # Infer and set frequency
    ts = df[target_col]
    
    # Try to infer frequency automatically
    try:
        inferred_freq = pd.infer_freq(ts.index)
        if inferred_freq:
            ts.index.freq = inferred_freq
        else:
            # If can't infer, try to determine manually
            time_diffs = ts.index.to_series().diff().dropna()
            most_common_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else None
            
            if most_common_diff:
                if most_common_diff == pd.Timedelta(days=1):
                    ts.index.freq = 'D'  # Daily
                elif most_common_diff == pd.Timedelta(days=7):
                    ts.index.freq = 'W'  # Weekly
                elif most_common_diff.days >= 28 and most_common_diff.days <= 31:
                    ts.index.freq = 'M'  # Monthly
                elif most_common_diff.days >= 90 and most_common_diff.days <= 92:
                    ts.index.freq = 'Q'  # Quarterly
                elif most_common_diff.days >= 365 and most_common_diff.days <= 366:
                    ts.index.freq = 'A'  # Annual
    except:
        # If all else fails, assume daily frequency
        ts.index.freq = 'D'
    
    return ts

def find_best_sarima_params(ts, max_p=2, max_d=1, max_q=2, max_P=1, max_D=1, max_Q=1):
    """Find best SARIMA parameters using AIC with parallel processing"""
    
    # Determine seasonal period based on frequency
    freq = ts.index.freq
    if freq:
        freq_str = str(freq)
        if 'D' in freq_str:  # Daily
            s = 7  # Weekly seasonality
        elif 'W' in freq_str:  # Weekly
            s = 4  # Monthly seasonality (4 weeks)
        elif 'M' in freq_str:  # Monthly
            s = 12  # Yearly seasonality
        elif 'Q' in freq_str:  # Quarterly
            s = 4  # Yearly seasonality
        elif 'A' in freq_str or 'Y' in freq_str:  # Annual
            s = 1  # No seasonality for annual data
        else:
            s = min(12, len(ts) // 4)  # Default
    else:
        s = min(12, len(ts) // 4)  # Default when frequency unknown
    
    # Skip seasonal component if not enough data or s=1
    if len(ts) < 2 * s or s == 1:
        max_P = max_D = max_Q = 0
        s = 0
    
    def fit_model(params):
        p, d, q, P, D, Q = params
        try:
            if s > 0:
                model = SARIMAX(ts, order=(p, d, q), 
                              seasonal_order=(P, D, Q, s),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
            else:
                model = SARIMAX(ts, order=(p, d, q),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
            
            fitted_model = model.fit(disp=False)
            return (fitted_model.aic, (p, d, q, P, D, Q))
        except:
            return (float('inf'), None)
    
    # Generate parameter combinations
    param_combinations = []
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for D in range(max_D + 1):
                        for Q in range(max_Q + 1):
                            param_combinations.append((p, d, q, P, D, Q))
    
    # Use parallel processing to fit models
    n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_model)(params) for params in param_combinations
    )
    
    # Find best model
    best_aic = float('inf')
    best_params = None
    
    for aic, params in results:
        if aic < best_aic and params is not None:
            best_aic = aic
            best_params = params
    
    if best_params is None:
        return ((1, 1, 1), (0, 0, 0, 0))
    
    p, d, q, P, D, Q = best_params
    return ((p, d, q), (P, D, Q, s))

def handle_nan(obj):
    """Convert NaN values to None for JSON serialization"""
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    elif isinstance(obj, dict):
        return {key: handle_nan(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [handle_nan(item) for item in obj]
    return obj

@app.post("/analyze")
async def analyze_data(date_column: str = Form(...), target_column: str = Form(...)):
    global uploaded_data, model_cache
    
    if uploaded_data is None:
        logging.error("No data uploaded")
        return JSONResponse(
            status_code=400,
            content={"detail": "No data uploaded"}
        )
    
    try:
        logging.info(f"Starting analysis with date_column={date_column}, target_column={target_column}")
        
        # Validate columns exist
        if date_column not in uploaded_data.columns:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Date column '{date_column}' not found in data"}
            )
        if target_column not in uploaded_data.columns:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Target column '{target_column}' not found in data"}
            )
        
        # Clean the data again to ensure consistency
        df = clean_data(uploaded_data.copy())
        logging.info(f"Data cleaned. Shape: {df.shape}")
        
        # Prepare time series
        ts = prepare_time_series(df, date_column, target_column)
        logging.info(f"Time series prepared. Length: {len(ts)}")
        
        if len(ts) < 30:
            logging.error("Insufficient data points")
            return JSONResponse(
                status_code=400,
                content={"detail": "Need at least 30 data points for analysis"}
            )
        
        # Find best SARIMA parameters
        logging.info("Finding best SARIMA parameters...")
        best_params = find_best_sarima_params(ts)
        order, seasonal_order = best_params
        logging.info(f"Best parameters found: order={order}, seasonal_order={seasonal_order}")
        
        # Check cache for existing model
        cache_key = get_cache_key(ts, order, seasonal_order)
        if cache_key in model_cache:
            logging.info("Using cached model")
            fitted_model = model_cache[cache_key]
        else:
            logging.info("Fitting new model")
            fitted_model = fit_cached_model(tuple(ts.values), order, seasonal_order)
            model_cache[cache_key] = fitted_model
        
        # Generate fitted values
        logging.info("Generating fitted values")
        fitted_values = fitted_model.fittedvalues
        
        # Calculate metrics
        logging.info("Calculating metrics")
        mae = mean_absolute_error(ts, fitted_values)
        rmse = np.sqrt(mean_squared_error(ts, fitted_values))
        mape = np.mean(np.abs((ts - fitted_values) / ts)) * 100
        logging.info(f"Metrics calculated - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        # Generate forecast
        logging.info("Generating forecast")
        forecast_steps = 30
        forecast = fitted_model.get_forecast(steps=forecast_steps)
        forecast_values = forecast.predicted_mean
        confidence_int = forecast.conf_int()
        
        # Create forecast dates
        last_date = ts.index[-1]
        freq = ts.index.freq
        
        if isinstance(last_date, pd.Timestamp):
            if freq:
                forecast_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq=freq)[1:]
            else:
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                             periods=forecast_steps, freq='D')
        else:
            forecast_dates = range(len(ts), len(ts) + forecast_steps)
        
        # Create plots
        logging.info("Creating visualization plots")
        plots = {}
        plot_paths = {}
        
        # ACF plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_acf(ts, ax=ax, lags=min(40, len(ts)//4))
        ax.set_title('Autocorrelation Function (ACF)')
        plots['acf'], plot_paths['acf'] = create_plot_base64(fig, 'acf')
        
        # PACF plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_pacf(ts, ax=ax, lags=min(40, len(ts)//4))
        ax.set_title('Partial Autocorrelation Function (PACF)')
        plots['pacf'], plot_paths['pacf'] = create_plot_base64(fig, 'pacf')
        
        # Seasonal decomposition
        if len(ts) >= 24:
            logging.info("Performing seasonal decomposition")
            decomposition = seasonal_decompose(ts, model='additive', period=min(12, len(ts)//2))
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            axes[0].plot(decomposition.observed)
            axes[0].set_title('Original Time Series')
            axes[0].grid(True)
            
            axes[1].plot(decomposition.trend)
            axes[1].set_title('Trend Component')
            axes[1].grid(True)
            
            axes[2].plot(decomposition.seasonal)
            axes[2].set_title('Seasonal Component')
            axes[2].grid(True)
            
            axes[3].plot(decomposition.resid)
            axes[3].set_title('Residual Component')
            axes[3].grid(True)
            
            plt.tight_layout()
            plots['decomposition'], plot_paths['decomposition'] = create_plot_base64(fig, 'decomposition')
        else:
            logging.info("Not enough data for seasonal decomposition, creating simple plot")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(ts.index, ts.values)
            ax.set_title('Time Series Data')
            ax.grid(True)
            plots['decomposition'], plot_paths['decomposition'] = create_plot_base64(fig, 'decomposition')
        
        # Prepare response data with null checks
        response_data = {
            "metrics": {
                "data_points": int(len(ts)),
                "mae": float(mae) if not np.isnan(mae) else 0.0,
                "rmse": float(rmse) if not np.isnan(rmse) else 0.0,
                "mape": float(mape) if not np.isnan(mape) else 0.0
            },
            "model_info": {
                "order": f"({order[0]}, {order[1]}, {order[2]})({seasonal_order[0]}, {seasonal_order[1]}, {seasonal_order[2]}, {seasonal_order[3]})",
                "aic": float(fitted_model.aic) if hasattr(fitted_model, 'aic') else 0.0,
                "bic": float(fitted_model.bic) if hasattr(fitted_model, 'bic') else 0.0,
                "llf": float(fitted_model.llf) if hasattr(fitted_model, 'llf') else 0.0
            },
            "historical": {
                "dates": [str(date) for date in ts.index],
                "values": [float(val) if not np.isnan(val) else 0.0 for val in ts.values]
            },
            "fitted_values": [float(val) if not np.isnan(val) else 0.0 for val in fitted_values.values],
            "forecast": {
                "dates": [str(date) for date in forecast_dates],
                "values": [float(val) if not np.isnan(val) else 0.0 for val in forecast_values.values],
                "upper_bound": [float(val) if not np.isnan(val) else 0.0 for val in confidence_int.iloc[:, 1].values],
                "lower_bound": [float(val) if not np.isnan(val) else 0.0 for val in confidence_int.iloc[:, 0].values]
            },
            "plots": plots,
            "plot_paths": plot_paths
        }
        
        logging.info("Analysis completed successfully")
        return JSONResponse(content=handle_nan(response_data))
        
    except Exception as e:
        logging.error(f"Error in analysis: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error in analysis: {str(e)}"}
        )

@app.get("/download-forecast")
async def download_forecast():
    """Endpoint to download forecast data as CSV"""
    # This would be implemented if needed for direct download
    pass

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