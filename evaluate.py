# ================================================================
# evaluate.py — RAGAS Evaluation Runner
# ================================================================
# Purpose: Evaluate RAG pipeline (app.py) quality using RAGAS metrics
#
# Flow:
#   1. Setup judge LLM (Claude) + embeddings (HuggingFace)
#   2. Load test_dataset.json
#   3. Hit /chat endpoint for each test question
#   4. Run RAGAS Faithfulness metric on collected data
#   5. Print overall + per-question scores
#
# Pre-requisites:
#   - app.py must be running on http://localhost:8000
#   - test_dataset.json must exist with [{question, ground_truth}, ...]
#   - ANTHROPIC_API_KEY must be set in .env
#
# Run:
#   python3 evaluate.py
# ================================================================

import os
import json
import warnings
import requests
import pandas as pd
from dotenv import load_dotenv

# Suppress deprecation warnings (RAGAS 0.4.3 LangChain wrappers)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----------------------------------------------------------------
# RAGAS Imports
# ----------------------------------------------------------------

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


# ================================================================
# Configuration
# ================================================================

RAG_API_URL = "http://localhost:8000/chat"
TEST_DATASET_PATH = "./test_dataset.json"
JUDGE_MODEL = "claude-haiku-4-5-20251001"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in .env")


# ================================================================
# Helper Functions
# ================================================================

def load_test_dataset(path: str) -> list[dict]:
    """Load test questions and ground truths from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"✅ Loaded {len(dataset)} test questions from {path}")
    return dataset


def query_rag(question: str) -> dict:
    """Send question to /chat endpoint, return response data."""
    response = requests.post(
        RAG_API_URL,
        json={"question": question},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"RAG API returned {response.status_code}: {response.text}"
        )
    return response.json()


def collect_evaluation_data() -> list[dict]:
    """For each test question: query RAG, combine with ground truth."""
    dataset = load_test_dataset(TEST_DATASET_PATH)
    print(f"📡 Querying RAG endpoint for {len(dataset)} questions...")

    eval_records = []
    for item in dataset:
        rag_response = query_rag(item["question"])
        eval_records.append({
            "question": item["question"],
            "answer": rag_response["answer"],
            "contexts": rag_response["contexts"],
            "ground_truth": item["ground_truth"],
        })

    print(f"✅ Collected {len(eval_records)} evaluation records")
    return eval_records


def setup_ragas_evaluator():
    """Initialize Claude judge LLM and HuggingFace embeddings for RAGAS."""
    print("🧠 Setting up RAGAS judge LLM and embeddings...")

    claude_llm = ChatAnthropic(
        model=JUDGE_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,  # Deterministic scoring
        max_tokens=1024,
    )
    wrapped_llm = LangchainLLMWrapper(claude_llm)

    hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    wrapped_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    print(f"✅ Judge LLM ready: {JUDGE_MODEL}")
    return wrapped_llm, wrapped_embeddings


def run_faithfulness_evaluation(records, wrapped_llm, wrapped_embeddings):
    """Run RAGAS Faithfulness metric on collected records."""
    print("\n📦 Converting eval records to HuggingFace Dataset...")
    eval_dataset = Dataset.from_list(records)
    print(f"✅ Dataset created with {len(eval_dataset)} records")

    print("\n🧠 Running Faithfulness evaluation...")
    print("   Claude is extracting claims and verifying each one...")

    return evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness],
        llm=wrapped_llm,
        embeddings=wrapped_embeddings,
    )


def display_results(result) -> float:
    """Print overall + per-question Faithfulness scores. Returns mean score."""
    print("\n" + "=" * 60)
    print("📊 RAGAS Evaluation Results")
    print("=" * 60)

    results_df = result.to_pandas()

    # Smart column detection (handles RAGAS API naming variations)
    score_columns = [c for c in results_df.columns if "faithful" in c.lower()]
    score_column = score_columns[0] if score_columns else "faithfulness"

    question_candidates = ["question", "user_input", "query", "input"]
    question_col = next(
        (c for c in question_candidates if c in results_df.columns),
        None,
    )

    # Overall score
    overall_score = results_df[score_column].mean()
    print(f"\n🎯 Overall Faithfulness Score: {overall_score:.4f}")

    # Per-question breakdown
    print("\n📋 Per-question scores:")
    for i, row in results_df.iterrows():
        score = row[score_column]

        if pd.isna(score):
            emoji, score_str = "❓", "NaN"
        elif score >= 0.8:
            emoji, score_str = "✅", f"{score:.4f}"
        elif score >= 0.5:
            emoji, score_str = "⚠️ ", f"{score:.4f}"
        else:
            emoji, score_str = "❌", f"{score:.4f}"

        if question_col:
            question_text = str(row[question_col])[:60] + "..."
            print(f"   {emoji} [{i+1}] Score: {score_str} | {question_text}")
        else:
            print(f"   {emoji} [{i+1}] Score: {score_str}")

    return overall_score


# ================================================================
# Entry Point
# ================================================================

def main():
    """Main evaluation pipeline."""
    print("🚀 Starting RAGAS evaluation...\n")

    # Part 1: Setup + Data Collection
    wrapped_llm, wrapped_embeddings = setup_ragas_evaluator()
    records = collect_evaluation_data()

    # Part 2: Faithfulness Evaluation
    print("\n" + "=" * 60)
    print("🧪 Part 2: Running RAGAS Faithfulness Evaluation")
    print("=" * 60)

    result = run_faithfulness_evaluation(records, wrapped_llm, wrapped_embeddings)
    overall_score = display_results(result)

    # Final Summary
    print("\n" + "=" * 60)
    print("✅ Part 2 complete! RAGAS evaluation working.")
    print(f"🏆 First measured RAG quality: {overall_score:.4f}")
    print("⏭  Next: Add more metrics (Answer Relevancy, Context Precision)")
    print("=" * 60)


if __name__ == "__main__":
    main()