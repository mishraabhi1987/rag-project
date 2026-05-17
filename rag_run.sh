#!/bin/bash
# ================================================
# RAG Pipeline — Mac Daily Run Script
# Usage: bash rag_run.sh
#
# What this does:
#   1. Updates the vector DB with any new docs
#   2. Starts the FastAPI server (blocks - server stays up)
#
# What this does NOT do:
#   - Run evaluation (run `python3 evaluate.py` separately
#     in another terminal when needed)
# ================================================

echo "🚀 Starting RAG Pipeline..."
echo "----------------------------------"

# Activate Virtual Environment
source rag_env/bin/activate

# STEP 1 — Run the Vector DB pipeline
# Scans .txt files, chunks them, embeds and stores in ChromaDB
# Comment this out if ChromaDB is already up to date
echo "📦 Step 1: Updating Vector DB..."
python3 rag_vector_db.py

echo "----------------------------------"

# STEP 2 — Start the FastAPI backend server
# This powers the chat API that the frontend talks to
# NOTE: This is a BLOCKING command — server keeps running until Ctrl+C
echo "🌐 Step 2: Starting FastAPI server on http://localhost:8000"
echo ""
echo "💡 To run evaluation:"
echo "   Open new terminal → source rag_env/bin/activate → python3 evaluate.py"
echo ""
uvicorn app:app --reload --port 8000