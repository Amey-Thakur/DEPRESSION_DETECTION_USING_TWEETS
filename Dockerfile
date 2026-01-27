# ==============================================================================
# DEPLOYMENT: Hugging Face Spaces (Docker)
# PROJECT: DEPRESSION-DETECTION-USING-TWEETS
# ==============================================================================

FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Force legacy setuptools behavior
ENV SETUPTOOLS_USE_DISTUTILS=stdlib

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Fix for legacy builds: downgrade tools and disable build isolation
RUN pip install --upgrade "pip<23.1" "setuptools<58.0" "wheel<0.41.0"
RUN pip install "packaging<22.0" "cython<3.0"

# Install project dependencies
# Note: Path is relative to the repository root where Dockerfile resides
COPY "Source Code/requirements.txt" ./
# --no-build-isolation is critical to allow our downgraded tools to handle legacy metadata
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Download spaCy model required for the NLP pipeline
RUN python -m spacy download en_core_web_lg

# Copy project source code
COPY "Source Code/" ./

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Run the Flask application
CMD ["python", "app.py"]
