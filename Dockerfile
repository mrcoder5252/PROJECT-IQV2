# Project-IQ: Production Container Dockerfile
FROM python:3.11-slim

# System metadata
LABEL maintainer="Project-IQ Team"
LABEL description="Academic Project Intelligence Platform"

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=8501

# Install system dependencies & Tesseract OCR engine
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Copy project files
COPY . .

# Create logs and data directories
RUN mkdir -p logs data/raw data/extracted data/cleaned data/chunks data/chroma_db data/reports

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to verify Streamlit responsiveness
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch production dashboard
CMD ["streamlit", "run", "web/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
