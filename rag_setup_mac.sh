#!/bin/bash
# ================================================
# RAG Pipeline — Mac Setup Script
# Run this ONCE to set everything up!
# Usage: bash rag_setup_mac.sh
# ================================================

echo "🚀 RAG Pipeline Setup Starting..."
echo "----------------------------------"

# STEP 1 — Create Virtual Environment
echo "📦 Step 1: Creating virtual environment..."
python3 -m venv rag_env
echo "✅ Virtual environment created!"

# STEP 2 — Activate Virtual Environment
echo ""
echo "⚡ Step 2: Activating virtual environment..."
source rag_env/bin/activate
echo "✅ Activated!"

# STEP 3 — Install Core RAG Packages
# (Required for the Vector DB pipeline — indexing, embeddings, retrieval)
echo ""
echo "📥 Step 3: Installing RAG core packages..."
echo "(This may take a few minutes first time)"
pip install langchain \
            langchain-community \
            langchain-text-splitters \
            chromadb \
            sentence-transformers \
            langchain-huggingface \
            langchain-chroma
echo "✅ RAG core packages installed!"

#langchain                - Core LangChain framework — connects all RAG components together
#langchain-community      - Community integrations — provides TextLoader, DirectoryLoader
#langchain-text-splitters - Text splitting utilities — splits documents into chunks
#chromadb                 - Vector database — stores and searches embeddings locally
#sentence-transformers    - Embedding model library — converts text to vectors using HuggingFace models
#langchain-huggingface    - LangChain + HuggingFace integration — loads HuggingFaceEmbeddings into LangChain
#langchain-chroma         - LangChain + ChromaDB integration — connects LangChain retriever to ChromaDB

# STEP 4 — Install Backend / API Packages
# (Required for FastAPI server and Claude API integration)
# Comment out this step if you only need the vector DB pipeline
echo ""
echo "📥 Step 4: Installing backend & API packages..."
pip install fastapi \
            uvicorn \
            anthropic \
            python-dotenv \
            requests
echo "✅ Backend packages installed!"

#fastapi       - Web framework, handles incoming HTTP requests (receives POST /chat request from frontend)
#uvicorn       - Server that runs FastAPI (listens on localhost:8000)
#anthropic     - Official Anthropic Python SDK (connects and talks to Claude API)
#python-dotenv - Reads the .env file (loads ANTHROPIC_API_KEY into the app)
#requests      - HTTP client library — used by evaluate.py to call the /chat endpoint

# STEP 5 — Install RAGAS Evaluation Packages
# (Required for evaluate.py — measures RAG quality with metrics like Faithfulness)
# Comment out this step if you don't need evaluation
echo ""
echo "📥 Step 5: Installing RAGAS evaluation packages..."
pip install ragas \
            langchain-anthropic \
            datasets \
            pandas
echo "✅ RAGAS evaluation packages installed!"

#ragas              - RAG evaluation framework — provides metrics (Faithfulness, Answer Relevancy, Context Precision)
#langchain-anthropic - LangChain + Anthropic integration — wraps Claude as judge LLM for RAGAS
#datasets           - HuggingFace Datasets library — converts eval records to RAGAS-compatible format
#pandas             - Data manipulation library — handles RAGAS results DataFrame for display & analysis

# ------------------------------------------------
# OPTIONAL — Uncomment the ones you need
# ------------------------------------------------

# PDF support — to index PDF files into ChromaDB
# pip install pypdf

# Excel / CSV support — to index spreadsheets into ChromaDB
# pip install pandas openpyxl

# Web scraping — to index website content into ChromaDB
# pip install beautifulsoup4 requests

# Progress bars for long evaluation runs (Part 3+)
pip install tqdm

# ------------------------------------------------

echo ""
echo "================================================"
echo "🎉 Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Activate venv:  source rag_env/bin/activate"
echo "  2. Run pipeline:   bash rag_run.sh"
echo "  3. Run evaluation: python3 evaluate.py"
echo ""