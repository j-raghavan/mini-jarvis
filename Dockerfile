# Clean production build
FROM python:3.11-slim

WORKDIR /app

# Install essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak \
    libespeak1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories first
RUN mkdir -p models data services static/audio

# Copy application files
# COPY backend/main.py .
# COPY backend/models/weather_classifier.py ./models/
# COPY backend/data/preprocessor.json ./data/
# COPY backend/data/intent_mapping.json ./data/
# COPY backend/models/best_model_epoch_17.pth ./models/best_model.pth

COPY backend/. .


# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]