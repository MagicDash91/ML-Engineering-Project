# AI-Powered Indonesian Legal Document Analysis

## Project Overview

This project is a FastAPI web application that leverages advanced AI technologies, including Google Gemini (Gemini 2.0), LangChain, and LangGraph, to analyze Indonesian legal documents. Users can upload multiple legal files (PDF, Word, Excel, CSV, etc.), ask detailed questions about the content, and receive accurate, structured answers powered by a custom language model prompt focused on Indonesian law.

## Features

- Upload multiple legal documents in various formats.
- Extract and combine content from uploaded files using LangChain's UnstructuredFileLoader.
- Use a custom language model prompt template specialized for interpreting Indonesian legal texts.
- Intelligent question classification to query Wikipedia, academic papers (arXiv), and current events (Tavily) tools if relevant.
- A multi-step LangGraph workflow that classifies questions, searches relevant sources, and summarizes results.
- User-friendly web interface built with FastAPI and Bootstrap.
- Supports detailed and structured responses, focusing on legal analysis, comparisons, summaries, and numeric data interpretation.
- Designed specifically to understand and analyze Indonesian laws and regulations.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/indonesian-legal-doc-analysis.git
cd indonesian-legal-doc-analysis
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Set environment variables by creating a `.env` file with:

```env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

## Usage

Run the FastAPI app:

```bash
uvicorn main:app --host 127.0.0.1 --port 9000 --reload
```

Open your browser and go to `http://127.0.0.1:9000/` to upload legal documents and ask questions.

## How It Works

- Users upload legal documents (PDF, DOCX, CSV, XLSX).
- The app extracts text from these documents using LangChain loaders.
- The question from the user is combined with the extracted content and fed into a LangGraph state machine:
  - **Classify**: Determines which external knowledge tools to query.
  - **Search**: Queries Wikipedia, arXiv, and Tavily if needed.
  - **Summarize**: Combines results and provides a final AI-generated answer.
- The answer is returned in a clean, formatted HTML page.

## Custom Prompt Template

The language model uses a detailed prompt template tailored to Indonesian legal analysis, emphasizing:

- Accurate interpretation of laws and regulations.
- Detailed and structured answers.
- Numeric data comparison and aggregation where relevant.
- Clear explanations with citations from the provided documents.

## Technologies Used

- FastAPI
- LangChain & LangGraph
- Google Gemini (Google Generative AI)
- WikipediaAPI, arXiv API, Tavily API
- Bootstrap for frontend styling
- Python 3.9+

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/yourusername/indonesian-legal-doc-analysis/issues).

## License

This project is licensed under the MIT License.

---

*Created by Your Name - 2025*
