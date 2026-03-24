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

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import aiohttp, asyncio; asyncio.run(aiohttp.ClientSession().get('https://open-api-vst.bingx.com/openApi/swap/v2/quote/ticker?symbol=BTC-USDT'))" \
    || exit 1

# Entry point
CMD ["python", "main.py"]
