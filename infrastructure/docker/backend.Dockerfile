# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set environment variables for Python optimization
ENV PYTHONUNBUFFERED=1 
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libopenblas-dev \
    liblapack-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements.txt and install Python dependencies
COPY src/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend source code
COPY src/backend/ /app/src/backend/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/forecasts /app/logs

# Install the backend package in development mode
WORKDIR /app/src/backend
RUN pip install -e .
WORKDIR /app

# Set up healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import os; os.path.exists('/app/logs/pipeline.log') or exit(1)"

# Set the entrypoint to run the backend service
ENTRYPOINT ["python", "-m", "src.backend.orchestration.pipeline"]