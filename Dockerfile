# ==============================================================================
# DEPLOYMENT: Hugging Face Spaces (Docker)
# PROJECT: DEPRESSION-DETECTION-USING-TWEETS
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
# Note: Path is relative to the repository root where Dockerfile resides
COPY "Source Code/requirements.txt" ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model required for the NLP pipeline
RUN python -m spacy download en_core_web_lg

# Copy project source code
COPY "Source Code/" ./

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Run the Flask application
CMD ["python", "app.py"]

