# Chat With Your Database

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Chat%20with%20Your%20Database/static/chat.png)

## Overview

This is a web application built using FastAPI that allows users to interact with their databases through natural language queries. The application uses LangChain to generate SQL queries based on user input and executes them against a specified database.

## Features

- **Natural Language Queries**: Users can ask questions in plain English, and the application will generate and execute the corresponding SQL queries.
- **Multiple Database Support**: Currently supports PostgreSQL and MySQL. Additional databases can be added easily.
- **Interactive UI**: A simple and intuitive web interface to input database credentials and questions.
- **Visualization**: Automatically generates charts from query results.

## Requirements

- Python 3.8+
- FastAPI
- LangChain
- Matplotlib
- Uvicorn (for running the server)
- Database drivers (e.g., psycopg2 for PostgreSQL, mysql-connector-python for MySQL)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
Install Dependencies

pip install -r requirements.txt
Set Up Environment Variables Create a .env file in the root directory with your Google API key:

GOOGLE_API_KEY=your_google_api_key_here
Running the Application
Start the Server

uvicorn main:app --reload
Access the Application Open your web browser and go to http://localhost:9000.
Usage
Select Database Type: Choose the type of your database (e.g., PostgreSQL, MySQL).
Enter Database Credentials: Fill in the host, port, database name, user, and password.
Ask a Question: Enter your question in the textarea and click "Submit Query".
View Results: The application will display the agent's response, console output, and any generated charts.
Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.

License
This project is licensed under the MIT License - see the LICENSE file for details.



### Additional Notes

- **Static Files**: Ensure that the `static` directory contains the necessary images (`logo.png`, `postgre.png`, `mysql.avif`, etc.) for the UI.
- **Environment Variables**: Make sure to replace `your_google_api_key_here` with your actual Google API key.
- **Dependencies**: You might need to create a `requirements.txt` file listing all the dependencies. Here’s an example:

  ```plaintext
  fastapi
  langchain
  langchain-experimental
  langchain-community
  langchain-google-genai
  matplotlib
  uvicorn
  psycopg2-binary  # For PostgreSQL
  mysql-connector-python  # For MySQL
