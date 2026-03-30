FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Non-root user for security
RUN useradd -m -u 1000 wagmi && chown -R wagmi:wagmi /app
USER wagmi

# Health check - verify core modules can be imported (no external API calls)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from core.event_bus import EventBus; from core.finance import to_decimal; print('OK')" \
    || exit 1

# Entry point
CMD ["python", "main.py"]
