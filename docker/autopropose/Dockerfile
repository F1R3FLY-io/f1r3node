FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY autopropose.py .
COPY config.yml .

# Create directory for contract files
RUN mkdir -p /contracts

# Run as non-root user for security
RUN useradd -m -u 1000 proposer && chown -R proposer:proposer /app
USER proposer

# Health check to ensure the service is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "autopropose.py", "--config", "config.yml"] 