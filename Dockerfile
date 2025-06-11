# Ultra-minimal production build
FROM python:3.11-slim

WORKDIR /app

# Install only essential runtime dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak \
    libespeak1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Copy and install requirements in one step
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge \
    && find /usr/local -type d -name __pycache__ -exec rm -r {} + || true \
    && find /usr/local -name "*.pyc" -delete

# Copy only the absolute essentials
# COPY backend/main.py .
# COPY backend/models/weather_classifier.py ./models/
# COPY backend/data/preprocessor.json ./data/
# COPY backend/data/intent_mapping.json ./data/

# # Copy only your best model (rename for consistency)
# COPY backend/models/best_model_epoch_17.pth ./models/best_model.pth

# # Copy services if they exist
# COPY backend/services/ ./services/ 2>/dev/null || echo "No services directory found"

COPY backend/. .
# Create audio directory
RUN mkdir -p static/audio

# Environment variables
ENV PYTHONPATH=/app \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Minimal health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=2 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use gunicorn with single worker for minimal memory usage
CMD ["gunicorn", "main:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "1", "--max-requests", "1000", "--max-requests-jitter", "100"]