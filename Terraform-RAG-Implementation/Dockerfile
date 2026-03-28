FROM python:3.12-slim

# Install Ollama
RUN apt-get update && apt-get install -y curl zstd && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model so container starts faster
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000

# Start Ollama in background, pull model, then run the app
CMD ["sh", "-c", "ollama serve & sleep 5 && ollama pull qwen3.5:cloud && python main.py"]
