FROM python:3.11-slim

# Build identifier to help force rebuilds and fingerprint deployments
ARG BUILD_ID="2025-10-10T06:10Z"
ENV BUILD_ID=${BUILD_ID}
LABEL org.opencontainers.image.created=${BUILD_ID}

WORKDIR /app

# Install system dependencies including gcc for compilation
RUN apt-get update && apt-get install -y \
    gcc \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC
ENV BUILD_ID=${BUILD_ID}

# Make start script executable
RUN chmod +x start.py

# Run the bot with startup script
CMD ["python", "start.py"]
