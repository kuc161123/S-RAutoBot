FROM python:3.11-slim

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

# Make start script executable
RUN chmod +x start.py

# Run the bot with startup script
CMD ["python", "start.py"]