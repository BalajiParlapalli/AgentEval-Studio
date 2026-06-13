#!/bin/bash
set -e

# HF Spaces uses port 7860 for the public URL
# We run FastAPI on 8000 (internal) and Streamlit on 7860 (public)

export DB_PATH="/app/results/agenteval.db"
export API_BASE_URL="http://localhost:8000/api/v1"

# Start FastAPI backend in background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait for backend to be ready
echo "Waiting for FastAPI backend..."
for i in {1..20}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend ready."
        break
    fi
    sleep 1
done

# Start Streamlit on HF Space's required port
streamlit run ui/streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
