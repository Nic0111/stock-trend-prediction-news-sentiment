FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt backend/requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -r backend/requirements.txt

COPY backend ./backend
COPY pipeline ./pipeline
COPY models/model-registry.json ./models/model-registry.json
COPY data/stock-metadata.csv ./data/stock-metadata.csv

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
