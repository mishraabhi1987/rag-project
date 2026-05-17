# ================================================================
# app.py — FastAPI Backend for RAG Chatbot
# ================================================================
# Purpose: Production-ready RAG system with RAGAS evaluation support.
# Flow:    Frontend → POST /chat → ChromaDB → Claude API → Response
# Run:     uvicorn app:app --reload --port 8000
# Docs:    http://localhost:8000/docs
# ================================================================

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

CHROMA_DB = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

TOP_K = int(os.environ.get("TOP_K", 3))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 1024))

# Temperature must be 0.0 for RAG — deterministic, factual outputs.
# Non-zero values introduce randomness and break RAGAS evaluation reliability.
TEMPERATURE = 0.0


# ----------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------

app = FastAPI(title="RAG Chatbot API")

# TODO: Restrict allow_origins to specific domains in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------
# ChromaDB & embeddings (loaded once at startup)
# ----------------------------------------------------------------

print("🔢 Loading ChromaDB and embedding model...")

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vectorstore = Chroma(
    persist_directory=CHROMA_DB,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

print("✅ ChromaDB loaded successfully!")


# ----------------------------------------------------------------
# Claude API client
# ----------------------------------------------------------------

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    print("⚠️  WARNING: ANTHROPIC_API_KEY not set!")

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ----------------------------------------------------------------
# Request / Response schemas
# ----------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]   # Source filenames for UI display
    contexts: list[str]  # Full chunk text — required for RAGAS evaluation


# ----------------------------------------------------------------
# Main chat endpoint
# ----------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    RAG flow: retrieve relevant chunks from ChromaDB, then generate
    a context-grounded answer using Claude.
    """

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    print(f"\n🔍 Searching for: {req.question}")

    # Retrieval: semantic search over the vector database
    try:
        docs = retriever.invoke(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChromaDB error: {str(e)}")

    if not docs:
        return ChatResponse(
            answer="I couldn't find any relevant information in the knowledge base for your question.",
            sources=[],
            contexts=[],
        )

    # Build context string for Claude with source markers for attribution
    context_parts = [
        f"[Source {i}]\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    ]
    context = "\n\n".join(context_parts)

    # dict.fromkeys() preserves insertion order while removing duplicates;
    # set() would randomize order and produce inconsistent UI display.
    sources = list(dict.fromkeys(
        os.path.basename(doc.metadata.get("source", "unknown"))
        for doc in docs
    ))

    # Full chunk text — exposed for RAGAS evaluation pipeline
    contexts = [doc.page_content for doc in docs]

    print(f"📌 Found {len(docs)} relevant chunks from: {sources}")

    # Generation: strict rules enforce Faithfulness and Answer Relevancy
    system_prompt = """You are a helpful RAG (Retrieval-Augmented Generation) assistant.
Your job is to answer user questions based ONLY on the provided context.

Rules:
- Answer based strictly on the given context
- If the answer is not in the context, say "I don't have enough information in the knowledge base to answer this."
- Be concise and clear
- Do not make up information
- Do NOT infer, assume, or elaborate beyond what is explicitly stated in the context
- Answer ONLY what was asked. No tangential information.
- Never mention the context, sources, or documents in your answer. Just answer directly and naturally."""

    user_prompt = f"""Context from knowledge base:
{context}

Question: {req.question}

Please answer based on the context above."""

    print("🤖 Calling Claude API...")

    try:
        response = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        answer = response.content[0].text
        print(f"✅ Claude responded ({response.usage.output_tokens} tokens used)")

    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key. Check ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit reached. Try again in a moment.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude API error: {str(e)}")

    return ChatResponse(answer=answer, sources=sources, contexts=contexts)


# ----------------------------------------------------------------
# Health check
# ----------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "running",
        "model": CLAUDE_MODEL,
        "chunks_in_db": vectorstore._collection.count(),
        "config": {
            "top_k": TOP_K,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
    }