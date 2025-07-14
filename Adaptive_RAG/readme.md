# 🧠 Adaptive RAG System

An intelligent document analysis system that uses Retrieval-Augmented Generation (RAG) with self-reflection capabilities, built with FastAPI, LangChain, and Google's Gemini AI.

## 🌟 Features

- **📄 PDF Document Processing**: Upload and analyze PDF documents
- **🔍 Intelligent Retrieval**: Vector-based document search using FAISS
- **🤖 Self-Reflection**: Automatic quality checking and answer validation
- **🌐 Web Search Fallback**: Integrates web search when documents lack information
- **🔄 Adaptive Iterations**: Rewrites questions and retries when needed
- **✅ Hallucination Detection**: Validates answers against source documents
- **🎯 Relevance Grading**: Grades retrieved documents for relevance
- **📊 Process Transparency**: Shows workflow steps and decision points

## 🏗️ Architecture

The system uses a LangGraph-based workflow with the following components:

1. **Document Retrieval**: Finds relevant chunks from uploaded PDFs
2. **Relevance Grading**: Evaluates if retrieved documents are relevant
3. **Answer Generation**: Creates responses based on retrieved context
4. **Self-Reflection**: Checks for hallucinations and answer quality
5. **Question Rewriting**: Improves questions for better retrieval
6. **Web Search**: Falls back to web search when documents are insufficient

## Project Screenshots :

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Adaptive_RAG/a1.jpg)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Adaptive_RAG/a2.jpg)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Adaptive_RAG/a3.jpg)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Adaptive_RAG/a4.webp)


## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google AI API key
- Tavily API key (optional, for web search)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/adaptive-rag-system.git
cd adaptive-rag-system
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

### Required Dependencies

Create a `requirements.txt` file with:
```txt
fastapi==0.104.1
uvicorn==0.24.0
langchain==0.0.340
langchain-community==0.0.3
langchain-google-genai==0.0.6
langgraph==0.0.19
faiss-cpu==1.7.4
pypdf==3.17.1
python-dotenv==1.0.0
python-multipart==0.0.6
google-generativeai==0.3.1
tavily-python==0.3.0
markdown==3.5.1
```

### API Keys Setup

1. **Google AI API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file

2. **Tavily API Key** (Optional):
   - Visit [Tavily](https://tavily.com)
   - Sign up and get your API key
   - Add it to your `.env` file

3. **LangChain API Key** (Optional):
   - Visit [LangSmith](https://smith.langchain.com)
   - Get your API key for tracing
   - Add it to your `.env` file

## 🖥️ Usage

### Starting the Server

```bash
python main.py
```

The server will start at `http://127.0.0.1:8000`

### Using the Web Interface

1. Open your browser and go to `http://127.0.0.1:8000`
2. Upload a PDF document
3. Enter your question about the document
4. Select the maximum number of iterations (1-4)
5. Click "Analyze Document"
6. View the results with process information

### API Endpoints

- `GET /`: Main upload form
- `POST /analyze`: Analyze document with question
- `GET /health`: Health check endpoint

## 🔧 Configuration

### Adjusting Parameters

You can modify various parameters in the code:

```python
# Text splitting parameters
chunk_size=1000
chunk_overlap=200

# LLM temperature
temperature=0.1

# Maximum iterations
max_iterations=3

# Retrieval parameters
k=3  # Number of documents to retrieve
```

### Customizing Prompts

The system uses several prompt templates that can be customized:

- `GRADING_PROMPT`: Document relevance grading
- `GENERATION_PROMPT`: Answer generation
- `HALLUCINATION_GRADER_PROMPT`: Hallucination detection
- `ANSWER_GRADER_PROMPT`: Answer quality assessment
- `QUESTION_REWRITER_PROMPT`: Question rewriting
- `WEB_SEARCH_PROMPT`: Web search integration

## 🔄 Workflow Details

### Adaptive RAG Process

1. **Initial Retrieval**: System retrieves relevant documents based on the question
2. **Relevance Check**: Grades retrieved documents for relevance
3. **Decision Point**: 
   - If relevant → Generate answer
   - If not relevant → Perform web search
4. **Self-Reflection**: Checks generated answer for:
   - Hallucinations (grounded in facts)
   - Question answering quality
5. **Iteration Decision**:
   - If good quality → Return answer
   - If poor quality → Rewrite question and retry
   - If max iterations reached → Return current answer

### Web Search Integration

When documents are not relevant, the system:
1. Performs web search using Tavily
2. Combines web results with document context
3. Generates comprehensive answer

## 📊 Monitoring and Debugging

### Process Information

The system provides detailed process information:
- Number of iterations used
- Document relevance grade
- Hallucination check results
- Answer quality assessment
- Web search usage indicator

### LangChain Tracing

Enable LangChain tracing by setting the environment variables:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_api_key
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the RAG framework
- **Google Gemini**: For the LLM capabilities
- **FAISS**: For vector similarity search
- **Tavily**: For web search capabilities
- **FastAPI**: For the web framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/adaptive-rag-system/issues) page
2. Create a new issue with detailed description
3. Include error messages and steps to reproduce

## 🔮 Future Enhancements

- [ ] Support for multiple document formats (DOCX, TXT, etc.)
- [ ] Multi-language support
- [ ] Conversation history and context
- [ ] Advanced document preprocessing
- [ ] Custom embedding models
- [ ] Batch processing capabilities
- [ ] REST API for integration
- [ ] Docker containerization
- [ ] Cloud deployment guides

---

**Built with ❤️ using FastAPI, LangChain, and Google Gemini AI**
