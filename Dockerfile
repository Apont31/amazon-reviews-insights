
FROM python:3.11-slim

WORKDIR /app

# System deps (optional but useful for torch CPU wheels)
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy code and artifacts
COPY serve ./serve
COPY artifacts ./artifacts
COPY models ./models
COPY docs ./docs

# Install python deps
RUN pip install --no-cache-dir -r serve/requirements.txt

EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
