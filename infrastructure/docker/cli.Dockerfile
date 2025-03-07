# Base image: Python 3.10 slim variant
FROM python:3.10-slim

# Set environment variables for Python optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Set working directory for the application
WORKDIR /app

# Install system dependencies required for building some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements file first to leverage Docker cache
COPY src/cli/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the CLI source code and backend code
COPY src/cli/ /app/src/cli/
COPY src/backend/ /app/src/backend/

# Install the CLI package in development mode
RUN pip install -e .

# Set the entrypoint to the CLI application
ENTRYPOINT ["python", "-m", "src.cli.main"]

# Default command shows help if no arguments are provided
CMD ["--help"]