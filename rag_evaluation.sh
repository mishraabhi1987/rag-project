#!/bin/bash
# ================================================
# RAG Evaluation Run Script
# Usage: bash rag_evaluation.sh
#
# What this does:
#   1. Starts FastAPI server in BACKGROUND
#   2. Waits for it to be ready
#   3. Runs evaluation (evaluate.py)
#   4. Stops the server
#   5. Cleans up server.log
# ================================================

echo "🧪 RAG Evaluation Cycle Starting..."
echo "----------------------------------"

# Activate venv
source rag_env/bin/activate

# Start server in background, save its process ID
echo "🌐 Starting server in background..."
uvicorn app:app --port 8000 > server.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready (max 10 seconds)
echo "⏳ Waiting for server to be ready..."
for i in {1..10}; do
    if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
        echo "✅ Server is ready!"
        break
    fi
    sleep 1
done

# Run evaluation
echo ""
echo "🧪 Running evaluation..."
python3 evaluate.py

# Cleanup: stop the server
echo ""
echo "🛑 Stopping server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null

# Cleanup: remove server log (regenerated each run)
rm -f server.log

echo "----------------------------------"
echo "✅ Evaluation cycle complete!"