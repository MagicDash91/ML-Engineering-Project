# Automatic Churn Data Analysis Prediction

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Customer%20Churn%20Prediction%20and%20Analyzer/static/churn (2).png)

## Overview

This is a web application built using FastAPI that allows users to analyze churn data stored in a DynamoDB table. The application provides a user-friendly interface to fetch and analyze churn data based on specified date ranges and questions. It uses LangChain for generating summaries and insights, and Scikit-Learn for machine learning tasks such as feature importance analysis.

## Features

- **Data Fetching**: Retrieve churn data from DynamoDB based on date ranges.
- **Natural Language Summaries**: Generate detailed summaries and insights using LangChain and Google Generative AI.
- **Machine Learning Analysis**: Perform feature importance analysis using Random Forest Classifier and visualize the results.
- **User-Friendly Interface**: A simple and intuitive web interface to input date ranges and questions.

## Requirements

- Python 3.8+
- FastAPI
- Boto3 (AWS SDK for Python)
- Pandas
- Scikit-Learn
- Seaborn
- Matplotlib
- Jinja2
- LangChain
- Google Generative AI
- Uvicorn (for running the server)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/MagicDash91/ML-Engineering-Project.git
   cd ML-Engineering-Project
Install Dependencies

pip install -r requirements.txt
Set Up Environment Variables Create a .env file in the root directory with the following environment variables:

AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_KEY=your_aws_secret_key
api=your_google_api_key
Running the Application
Start the Server

uvicorn main:app --reload
Access the Application Open your web browser and go to http://localhost:8000.
Usage
Select Date Range: Enter the start and end dates in the form to specify the date range for data retrieval.
Ask a Question: Enter your question in the textarea to get insights and summaries.
Fetch Data: Click "Fetch Data" to retrieve and analyze the data.
View Results: The application will display the fetched data, a summary, and a chart showing feature importance.
Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.

Development Setup
Clone the Repository

git clone https://github.com/MagicDash91/ML-Engineering-Project.git
cd ML-Engineering-Project
Install Dependencies

pip install -r requirements.txt
Set Up Environment Variables Create a .env file in the root directory with the necessary environment variables.
Run the Application

uvicorn main:app --reload
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
FastAPI
Boto3
Pandas
Scikit-Learn
Seaborn
Matplotlib
Jinja2
LangChain
Google Generative AI
Contact
For any questions or support, please contact MagicDash91.

Application Screenshot



### Explanation

1. **Overview**: Provides a brief description of the project.
2. **Features**: Lists the key features of the application.
3. **Requirements**: Lists the necessary software and libraries.
4. **Installation**: Detailed steps to clone the repository, install dependencies, and set up environment variables.
5. **Running the Application**: Instructions to start the server and access the application.
6. **Usage**: Guides users on how to use the application.
7. **Contributing**: Encourages contributions and provides guidance on submitting issues and pull requests.
8. **Development Setup**: Additional steps for setting up the development environment.
9. **License**: Specifies the license under which the project is released.
10. **Acknowledgments**: Lists the libraries and tools used in the project.
11. **Contact**: Provides a way to contact the maintainer.

### Additional Notes

- **Environment Variables**: Ensure that the `.env` file contains the correct AWS credentials and Google API key.
- **Images**: Replace `screenshot.png` with an actual screenshot of your application if available.
- **Dependencies**: Ensure that `requirements.txt` lists all necessary dependencies. Here is an example `requirements.txt`:

  ```plaintext
  fastapi
  boto3
  pandas
  scikit-learn
  seaborn
  matplotlib
  jinja2
  langchain
  langchain-google-genai
  google-generativeai
  uvicorn
  python-dotenv
