# Gemini MoE Document Analyzer

An advanced AI-powered document analysis system that uses **Mixture of Experts (MoE)** with Google Gemini to provide comprehensive insights from multiple file types. The system features specialized AI experts that analyze documents from different perspectives and synthesizes their findings into actionable insights.

## Features

### Core Functionality
- **Multi-file Upload**: Support for PDF, DOC, DOCX, PPT, PPTX, XLS, XLSX, CSV, TXT files
- **Mixture of Experts**: 5 specialized AI personas for comprehensive analysis
- **Interactive Chat**: RAG-powered Q&A with your documents
- **Session Management**: Multi-user support with isolated sessions
- **Real-time Processing**: Asynchronous document processing and analysis

### AI Expert Personas
1. **Summarization Expert** - Concise, structured summaries and key point extraction
2. **Insight Expert** - Strategic business analysis and actionable recommendations  
3. **Research Expert** - Academic rigor and evidence-based analysis
4. **Financial Expert** - Financial analysis and market insights
5. **Technical Expert** - Technical documentation and system analysis

### Two Analysis Modes
- **MoE Analysis**: Comprehensive multi-expert analysis with synthesis
- **Chat Mode**: Quick conversational Q&A using retrieval-augmented generation

## Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **AI Models**: Google Gemini 2.0 Flash
- **Document Processing**: LangChain UnstructuredFileLoader
- **Vector Storage**: FAISS with Google Embeddings
- **Expert Selection**: TF-IDF + keyword matching
- **Frontend**: Bootstrap 5, Vanilla JavaScript
- **Text Processing**: RecursiveCharacterTextSplitter

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/gemini-moe-document-analyzer.git
   cd gemini-moe-document-analyzer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Gemini API**:
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Replace `GEMINI_API_KEY` in `main.py` with your actual key:
   ```python
   GEMINI_API_KEY = "your_api_key_here"
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

5. **Access the application**:
   Open your browser and go to `http://127.0.0.1:8000`

## Requirements

```
fastapi
uvicorn[standard]
langchain
langchain-google-genai
google-generativeai
scikit-learn
pandas
numpy
python-multipart
aiofiles
faiss-cpu
unstructured[all-docs]
python-magic-bin
markdown
```

## Usage

### Document Upload and Processing
1. **Upload Files**: Drag and drop or browse to select multiple documents
2. **Process Documents**: Click "Process Documents" to extract and chunk text
3. **Session Created**: System creates vector embeddings for efficient retrieval

### MoE Analysis
1. **Ask Questions**: Enter analysis questions in the AI Analysis panel
2. **Expert Selection**: System automatically selects relevant experts
3. **Multi-perspective Analysis**: Receive comprehensive insights from multiple AI specialists
4. **Synthesis**: Get integrated summary combining expert perspectives

### Chat Mode
1. **Interactive Q&A**: Ask specific questions about document content
2. **Context Retrieval**: System finds relevant document chunks
3. **Direct Answers**: Get focused responses based on document context

## API Endpoints

- `POST /upload` - Upload and process multiple files
- `POST /analyze` - Perform MoE analysis on documents
- `POST /chat` - Chat with documents using RAG
- `GET /sessions/{session_id}` - Get session information
- `DELETE /sessions/{session_id}` - Delete session
- `GET /health` - Health check

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File Upload   │───▶│  Document        │───▶│   Vector Store  │
│   (Multi-type)  │    │  Processing      │    │   (FAISS)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   User Query    │───▶│  Expert          │◄────────────┘
└─────────────────┘    │  Selection       │
                       └──────────────────┘
                                │
                       ┌──────────────────┐
                       │  Mixture of      │
                       │  Experts         │
                       │  (Parallel)      │
                       └──────────────────┘
                                │
                       ┌──────────────────┐
                       │  Response        │
                       │  Synthesis       │
                       └──────────────────┘
```

## Configuration

### Expert Selection Parameters
- `max_experts`: Maximum number of experts to use (default: 3)
- `similarity_threshold`: Minimum similarity score for expert selection
- `keyword_boost`: Weight for keyword-based expert selection

### Document Processing
- `chunk_size`: Text chunk size for processing (default: 2000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `max_documents`: Maximum documents to use for context (default: 10)

## Example Usage

### Basic Analysis
```python
# Upload documents through the web interface
# Ask: "Summarize the key findings from these reports"
# Result: Multi-expert analysis with synthesis
```

### Chat Interaction
```python
# After uploading documents
# Chat: "What did the report say about Q3 revenue?"
# Result: Direct answer based on document content
```

## Performance Considerations

- **Processing Time**: Scales with document size and number of experts
- **Memory Usage**: Proportional to document count and embedding size
- **API Costs**: Gemini API calls scale with expert usage and document length
- **Concurrent Sessions**: Limited by available memory and API rate limits

## Troubleshooting

### Common Issues
1. **API Key Error**: Ensure your Gemini API key is valid and has sufficient quota
2. **File Processing Error**: Check file format support and file integrity
3. **Memory Issues**: Reduce chunk size or document count for large files
4. **Slow Response**: Consider reducing max_experts or document context

### Debug Mode
Enable debug logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/yourusername/gemini-moe-document-analyzer.git
cd gemini-moe-document-analyzer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Testing

```bash
# Run tests (when implemented)
pytest tests/

# Manual testing
python main.py
# Navigate to http://127.0.0.1:8000
# Upload test documents and verify functionality
```

## Deployment

### Local Deployment
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Security Considerations

- **API Key Management**: Use environment variables for API keys
- **File Upload Security**: Implement file type validation and size limits
- **Session Management**: Add session expiration and cleanup
- **Input Sanitization**: Validate all user inputs

## Limitations

- Requires Google Gemini API key (paid service after free tier)
- Processing time increases with document size and complexity
- Memory usage scales with number of documents and embeddings
- Single-threaded document processing (can be optimized)
- Limited to text-based document analysis

## Future Enhancements

- [ ] Support for additional file formats (audio, video)
- [ ] Custom expert persona creation
- [ ] Advanced visualization of expert analysis
- [ ] Integration with cloud storage services
- [ ] Multi-language document support
- [ ] Batch processing capabilities
- [ ] Export analysis results to various formats
- [ ] User authentication and authorization
- [ ] Advanced caching mechanisms
- [ ] Distributed processing support

## Changelog

### v1.0.0 (Current)
- Initial release with basic MoE functionality
- Multi-file upload support
- Interactive chat interface
- Session management

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Gemini AI for powering the expert analysis
- LangChain for document processing infrastructure
- FastAPI for the robust web framework
- Bootstrap for the responsive UI components
- FAISS for efficient vector similarity search

## Citation

If you use this project in your research, please cite:
```bibtex
@software{gemini_moe_analyzer,
  title={Gemini MoE Document Analyzer},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/gemini-moe-document-analyzer}
}
```

## Support

For issues and questions:
- Create an issue in this repository
- Check the [documentation](https://github.com/yourusername/gemini-moe-document-analyzer/wiki)
- Review existing issues for solutions

## Contact

- **Author**: [Michael Wiryaseputra]
- **Email**: [michwirja@gmail.com]

---

**Note**: This project requires a Google Gemini API key to function. Make sure to keep your API key secure and never commit it to version control.
