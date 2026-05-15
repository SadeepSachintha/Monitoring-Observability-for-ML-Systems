FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
# Ensure the model files are copied if they exist (they might be created in local dir during dev)
COPY model.pk[l] .
COPY baseline_stats.jso[n] .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
