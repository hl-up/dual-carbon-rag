# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/app/data
WORKDIR /app

COPY requirements.lock .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir --no-deps chromadb==1.5.9 && \
    python -m pip install --timeout 120 --retries 5 -r requirements.lock
COPY . .

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
