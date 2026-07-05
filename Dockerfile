# The Playwright base image ships Chromium + its OS dependencies preinstalled,
# so the JS-aware scraper works inside the container with no extra setup. The
# pinned tag matches playwright==1.48.0 in requirements.txt.
FROM mcr.microsoft.com/playwright/python:v1.48.0-jammy

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/hf

# Install Python dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Default: run the API + chat UI. Scraping/ingestion are run as one-off commands
# (see README) against the same image.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
