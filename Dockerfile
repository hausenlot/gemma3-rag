FROM python:3.11-slim

WORKDIR /app

# System deps for chromadb native bits
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's baked into the image
# (avoids a slow download on first container start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app.py .

# /data/chroma is where ChromaDB persists vectors
VOLUME ["/data/chroma"]

EXPOSE 8329

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8329", "--reload"]
