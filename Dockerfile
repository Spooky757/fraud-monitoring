FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

RUN mkdir -p data/incoming artifacts/champion monitoring/reference monitoring/reports monitoring/state

ENV PYTHONPATH=/app/src
ENV FRAUD_MONITORING_ENV=prod

ENTRYPOINT ["python", "-m", "fraud_monitoring.cli"]
CMD ["--help"]
