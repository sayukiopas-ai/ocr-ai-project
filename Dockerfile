FROM python:3.12-slim

# System deps for pdf2image (poppler) and image processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py app.py ./
COPY modules/ modules/
COPY pages/ pages/
COPY styles/ styles/
COPY .streamlit/ .streamlit/

# Create data directories
RUN mkdir -p uploads data

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
