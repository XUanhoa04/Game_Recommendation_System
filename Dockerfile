FROM python:3.11-slim

WORKDIR /app

# System deps for scientific wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_DEBUG=0
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# Expect data/ artifacts mounted or baked into image
CMD ["python", "app.py"]
