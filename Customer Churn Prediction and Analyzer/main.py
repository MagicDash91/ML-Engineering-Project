import os
import pandas as pd
import boto3
from boto3.dynamodb.conditions import Attr
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, FileResponse
from jinja2 import Template
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import markdown
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load environment variables
load_dotenv()

gemini_model_2_flash = "gemini-2.0-flash"
api = os.getenv("api")

# Initialize DynamoDB resource
dynamodb = boto3.resource(
    "dynamodb",
    aws_access_key_id=os.getenv("AWS_ACCES_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name="ap-southeast-1"
)

table = dynamodb.Table("churn")

# FastAPI app
app = FastAPI(title="Churn Data API", version="1.0")

# HTML Template for analyze.html
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Churn Data Analysis</title>
    
    <!-- Bootstrap 5 CDN -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <style>
        body {
            background-color: #f8f9fa;
        }
        .container {
            margin-top: 50px;
        }
        .card {
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        table {
            margin-top: 20px;
        }
        .table-container {
            max-height: 500px;
            max-width: 1400px;
            overflow: auto;
            border: 1px solid #dee2e6;
            border-radius: 8px;
        }
        thead {
            position: sticky;
            top: 0;
            background-color: #343a40;
            color: white;
            z-index: 1;
        }
    </style>
</head>
<body>

<div class="container">
    <h1 class="text-center my-4">📊 Churn Data Analysis</h1>

    <div class="card p-4">
        <form action="/churn/" method="get">
            <div class="row">
                <div class="col-md-6">
                    <label for="start_date" class="form-label">Start Date:</label>
                    <input type="date" id="start_date" name="start_date" class="form-control">
                </div>
                <div class="col-md-6">
                    <label for="end_date" class="form-label">End Date:</label>
                    <input type="date" id="end_date" name="end_date" class="form-control">
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-md-12">
                    <label for="question" class="form-label">Question:</label>
                    <input type="text" id="question" name="question" class="form-control" placeholder="Ask about the data">
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-md-12 d-flex justify-content-end">
                    <button type="submit" class="btn btn-primary">Fetch Data</button>
                </div>
            </div>
        </form>
    </div>

    {% if data and data|length > 0 %}
        <h2 class="mt-4">Results:</h2>
        <div class="table-container">
            <table class="table table-striped table-hover">
                <thead class="table-dark">
                    <tr>
                        {% for column in data[0].keys() %}
                            <th>{{ column }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for row in data %}
                        <tr>
                            {% for value in row.values() %}
                                <td>{{ value }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Summary Section -->
        <div class="card p-4 mt-4">
            <h3>📄 Summary</h3>
            <p>{{ summary | safe }}</p>
        </div>

        <!-- Feature Importance Section -->
        <div class="card p-4 mt-4">
            <h3>📊 Feature Importance</h3>
            <img src="/feature-importance" alt="Feature Importance Chart" class="img-fluid rounded mx-auto d-block" style="max-width: 80%;">
        </div>
        
    {% elif data is not none %}
        <div class="alert alert-warning mt-4" role="alert">
            ⚠️ No data found for the selected date range. Please try different dates.
        </div>
    {% endif %}
</div>

<!-- Bootstrap Bundle JS (for interactive features) -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

</body>
</html>
"""

# Serve HTML Page
@app.get("/", response_class=HTMLResponse)
def serve_homepage():
    template = Template(html_template)
    return template.render(data=None, summary="")

# Define the path where the image is stored
IMAGE_PATH = "feature_importance.png"

@app.get("/feature-importance")
async def get_feature_importance():
    if os.path.exists(IMAGE_PATH):
        return FileResponse(IMAGE_PATH, media_type="image/png")
    return {"error": "Image not found"}

# Fetch churn data and render it in HTML
@app.get("/churn/", response_class=HTMLResponse)
def fetch_churn_data(
    request: Request,
    start_date: str = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date (YYYY-MM-DD)"),
    question: str = Query(None, description="Your question")
):
    """Fetch churn data from DynamoDB filtered by transaction_date and render it in HTML"""
    try:
        # Define filter expression
        filter_expression = None

        if start_date and end_date:
            filter_expression = Attr("transaction_date").between(start_date, end_date)
        elif start_date:
            filter_expression = Attr("transaction_date").gte(start_date)
        elif end_date:
            filter_expression = Attr("transaction_date").lte(end_date)

        # Fetch data from DynamoDB
        scan_kwargs = {}
        if filter_expression:
            scan_kwargs["FilterExpression"] = filter_expression

        response = table.scan(**scan_kwargs)

        # Convert items to DataFrame
        data = response.get("Items", [])
        df = pd.DataFrame(data)

        if df.empty:
            return template.render(data=None, summary="No data found for the selected date range.")

        # Save DataFrame to Excel
        df.to_excel("churn_prediction_result.xlsx", index=False)

        # Convert data to dictionary format for rendering
        data_dict = df.to_dict(orient="records")

        # Configure LLM and generate summary
        genai.configure(api_key=api)
        llm = ChatGoogleGenerativeAI(model=gemini_model_2_flash, google_api_key=api, temperature=0)

        # Load the result CSV data for analysis
        loader = UnstructuredExcelLoader("churn_prediction_result.xlsx", mode="elements")
        docs = loader.load()

        template1 = f"""
        Based on the following retrieved context:
        {{text}}
        
        You are a skilled Data Analyst tasked with performing a customer churn analysis. Please focus on the following aspects:
        
        1. **Churn Drivers**:
        - Identify the main factors contributing to customer churn based on the dataset.
        - Explain how these factors influence customer behavior and churn probability.
        
        2. **Churn Prediction Insights**:
        - Analyze patterns and trends among customers who are likely to churn and those who are not.
        - Highlight the key differences between these groups in terms of their characteristics and behaviors.
        
        3. **Actionable Strategies**:
        - Suggest potential strategies to reduce customer churn, such as targeted retention campaigns, personalized offers, or service improvements.
        - Provide specific actions for addressing the primary drivers of churn identified in the analysis.
        
        4. **Customer Segmentation**:
        - Based on the analysis, segment customers into groups such as 'High Risk of Churn', 'Moderate Risk', and 'Low Risk'.
        - For each segment:
            - Provide a descriptive name.
            - Explain the key characteristics of customers in this segment.
        
        5. **Engagement Recommendations**:
        - Propose targeted engagement strategies for each customer segment to maximize retention and lifetime value.
        
        6. **Business Impact**:
        - Quantify the potential impact of the suggested strategies on customer retention and overall business performance.
        
        The goal is to derive actionable insights that can directly inform retention strategies and improve customer satisfaction.
        
        {question}
        """

        prompt_template = PromptTemplate.from_template(template1)
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="text"
        )
        response = stuff_chain.invoke(docs)
        summary = response["output_text"]
        # Convert Markdown to HTML
        summary = markdown.markdown(summary)

        # Preprocess the DataFrame
        # 1. Drop columns containing 'id' in their name (case insensitive)
        id_columns = [col for col in df.columns if "id" in col.lower()]
        id_columns_data = df[id_columns]  # Preserve ID columns
        df = df.drop(columns=id_columns)

        # 2. Drop categorical columns with more than 10 unique values
        categorical_columns = df.select_dtypes(include=["object"]).columns
        high_cardinality_cols = [
            col for col in categorical_columns if df[col].nunique() > 10 and col != "Churn"
        ]
        df = df.drop(columns=high_cardinality_cols)

        # 3. Handle missing values
        if not df.mode().empty:  
            df = df.fillna(df.mode().iloc[0])  
        else:  
            df = df.fillna(0)  

        # 4. Label Encoding
        for col in df.select_dtypes(include=["object"]).columns:
            label_encoder = preprocessing.LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col].astype(str))  # Convert to string to avoid NaN issues
            print(f"{col}: {df[col].unique()}")

        # Select the features (X) and the target variable (y)
        X = df.drop('Churn', axis=1)
        y = df['Churn']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Perform GridSearchCV
        rfc = RandomForestClassifier(class_weight='balanced')
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'max_features': ['sqrt', 'log2', None],
            'random_state': [0, 42]
        }
        grid_search = GridSearchCV(rfc, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        # ✅ Apply the best model to make predictions
        y_pred = best_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print("Best Parameters:", grid_search.best_params_)
        print(f"Model Accuracy: {accuracy:.4f}")

        # Feature importance
        imp_df = pd.DataFrame({
            "Feature Name": X_train.columns,
            "Importance": best_model.feature_importances_
        })

        # Sort and select top 10 features
        fi = imp_df.sort_values(by="Importance", ascending=False)
        fi2 = fi.head(10)

        # Plot the feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(data=fi2, x='Importance', y='Feature Name')
        plt.title('Top 10 Feature Importance Each Attribute (Random Forest)', fontsize=18)
        plt.xlabel('Importance', fontsize=16)
        plt.ylabel('Feature Name', fontsize=16)

        # Save the plot as PNG
        image_path = "feature_importance.png"
        plt.savefig(image_path, format='png', bbox_inches='tight')
        plt.close()

    except Exception as e:
        data_dict = {"error": str(e)}
        summary = ""

    # Render HTML with data
    template = Template(html_template)
    return template.render(data=data_dict, summary=summary)

# Run the FastAPI application with Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)