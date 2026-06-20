FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-meridian.txt .
RUN pip install --no-cache-dir -r requirements-meridian.txt

# Download embedding model at build time so workers don't re-download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
