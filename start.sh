#!/usr/bin/env bash
set -e

echo "Starting Voice Genetics full-stack container..."

cd /app

echo "Starting FastAPI backend on port 8000..."
python -m uvicorn app.main:app \
  --app-dir /app/backend \
  --host 0.0.0.0 \
  --port 8000 &

BACKEND_PID=$!

echo "Waiting for backend health check..."
for i in {1..90}; do
  if curl -s http://127.0.0.1:8000/health > /dev/null; then
    echo "Backend is healthy."
    break
  fi

  if [ "$i" -eq 90 ]; then
    echo "Backend failed to start."
    exit 1
  fi

  sleep 1
done

echo "Starting Streamlit frontend on port 8501..."
cd /app/frontend

streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false

kill $BACKEND_PID