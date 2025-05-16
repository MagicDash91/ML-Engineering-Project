# Fraud Prediction Web Application

This project implements a **Fraud Detection** system that predicts fraudulent transactions using a Random Forest machine learning model. It provides an interactive web interface built with **FastAPI** and **Bootstrap**, allowing users to select a date range, view transaction predictions, and visualize total transaction amounts over time.

Additionally, the project integrates advanced **LangChain** components with **Google Generative AI** for enhanced fraud data analysis and natural language insights.

---

## Features

- Upload or fetch transaction data from PostgreSQL database
- Filter transactions by date range
- Predict fraud on transactions using a pre-trained Random Forest model (`random_forest_model.pkl`)
- Display fraud prediction results in a responsive HTML table
- Visualize total transaction amount over selected dates using Seaborn line charts
- Interactive UI built with FastAPI and Bootstrap for easy usability
- Rich fraud analysis and insights powered by LangChain with Google Generative AI and LangSmith
- Support for PDF, CSV document loading and embedding with FAISS vector stores for deep content analysis

---

## Technologies & Tools Used

- **FastAPI**: Backend web framework for API and UI handling
- **Pandas**: Data manipulation and processing
- **Seaborn & Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning (Random Forest model)
- **PostgreSQL & Psycopg2**: Database for storing and retrieving transactions
- **LangChain**: Framework for building LLM-powered chains and workflows
- **Google Generative AI** via `langchain_google_genai`: For generating natural language fraud analysis
- **LangSmith**: For managing LangChain development and tracing
- **FAISS**: Vector database for semantic search over documents
- **Bootstrap**: Responsive frontend design
- **Jinja2 Templates**: Server-side HTML templating
- **dotenv**: Managing environment variables securely

---

## Project Structure

- `app.py`: Main FastAPI application with routes for UI and prediction
- `random_forest_model.pkl`: Pre-trained fraud detection model
- `dummy_simple_transaction_data.csv`: Sample transaction dataset
- `templates/`: HTML templates using Jinja2 and Bootstrap
- `static/`: Static files including saved visualizations
- `utils.py` (optional): Helper functions for DB connection, plotting, and data processing
- `.env`: Environment variables including DB credentials and API keys

---

## Project Screenshots :

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Fraud_Detection_Analysis/static/r1.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Fraud_Detection_Analysis/static/r2.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Fraud_Detection_Analysis/static/r3.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Fraud_Detection_Analysis/static/r4.JPG)

## Setup & Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/magicdash91/fraud-prediction-app.git
    cd fraud-prediction-app
    ```

2. **Create and activate a Python virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set environment variables**

    Create a `.env` file with your database credentials and Google API keys, e.g.:

    ```
    POSTGRES_USER=postgres
    POSTGRES_PASSWORD=your_password
    POSTGRES_DB=fraud
    POSTGRES_HOST=localhost
    POSTGRES_PORT=5432

    GOOGLE_API_KEY=your_google_api_key
    ```

5. **Run the FastAPI app**
    ```bash
    uvicorn app:app --reload
    ```

6. **Access the web app**

    Open your browser and navigate to [http://localhost:8000](http://localhost:8000).

---

## Usage

- Select a date range for transaction data analysis.
- The app fetches transaction records from PostgreSQL.
- Predicts fraud status using the Random Forest model.
- Displays results in a table (without date column).
- Shows a line chart of total transaction amount over the selected dates.
- Generates detailed fraud analysis using Google Generative AI and LangChain prompt chains.

---

## Extending the Project

- Integrate more advanced ML models or retrain on updated datasets.
- Add user authentication and role-based access.
- Extend document loader support (Excel, JSON, etc.) for richer analysis.
- Deploy with Docker and Kubernetes for scalable production use.
- Add real-time streaming data and live prediction dashboards.

---

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://python.langchain.com/en/latest/)
- [Google Generative AI](https://developers.generativeai.google/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Seaborn](https://seaborn.pydata.org/)
- Bootstrap

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to open issues or submit pull requests for improvements!

---


