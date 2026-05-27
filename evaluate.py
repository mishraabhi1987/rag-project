# ================================================================
# evaluate.py — RAGAS Evaluation Runner (v3: Clean Console + KT-Ready)
# ================================================================
# Purpose: Evaluate the RAG pipeline (app.py) on all 4 core RAGAS metrics
#
# Metrics:
#   1. Faithfulness        — Is the answer grounded in retrieved context?
#   2. Answer Relevancy    — Does the answer address the question?
#   3. Context Precision   — Are relevant chunks ranked higher?
#   4. Context Recall      — Did retrieval capture all needed info?
#
# Flow:
#   1. Setup judge LLM (Claude) + embeddings (HuggingFace)
#   2. Load test_dataset.json
#   3. Hit /chat endpoint for each test question
#   4. Run all 4 RAGAS metrics in a single pass
#   5. Print aggregate scores + save full results to eval_results/
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
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm  # Progress bars for clean console output

# Hide noisy LangChain wrapper deprecation warnings from RAGAS 0.4.x
warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
# RAGAS expects its own BaseRagasLLM / BaseRagasEmbeddings interface.
# RAGAS adapter wrappers — translate LangChain's LLM/embedding objects into the interface RAGAS expects, 
# so any LangChain-supported model (Claude, OpenAI, Gemini, etc.) plugs into RAGAS without custom code.
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables (ANTHROPIC_API_KEY) from .env file
load_dotenv()


# ================================================================
# Configuration
# ================================================================
# All tunable knobs live here. To change models, thresholds, or
# file paths, edit only this section — no need to touch functions.
# ================================================================

RAG_API_URL = "http://localhost:8000/chat"
TEST_DATASET_PATH = "./test_dataset.json"
RESULTS_DIR = Path("./eval_results")
JUDGE_MODEL = "claude-haiku-4-5-20251001"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# All 4 RAGAS metrics run together in a single evaluation pass.
# Adding/removing a metric here automatically flows through the
# entire pipeline — no other code changes needed.
METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

# Score interpretation tiers. Used for emoji-based visual indicators
# in the console output (✅ excellent, 🟢 good, ⚠️ poor, ❌ very poor).
THRESHOLDS = {
    "excellent": 0.85,
    "good": 0.70,
    "poor": 0.50,
}

# Plain-English meaning of each metric — embedded in the JSON output
# so anyone reading the results file understands what each score means
# without referring back to RAGAS documentation.
METRIC_MEANINGS = {
    "faithfulness": "Is the answer grounded in retrieved context? (Higher = less hallucination)",
    "answer_relevancy": "Does the answer actually address the question? (Higher = on-topic)",
    "context_precision": "Are relevant chunks ranked higher in retrieval? (Higher = better ranking)",
    "context_recall": "Did retrieval capture all info needed for ground truth? (Higher = complete)",
}

# Fail fast if API key is missing — better than crashing mid-evaluation
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in .env")


# ================================================================
# Helper Functions
# ================================================================

def load_test_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"✅ Loaded {len(dataset)} test questions")

    # Honest disclosure about statistical confidence
    if len(dataset) < 10:
        print(f"⚠️  Small sample ({len(dataset)} questions) — scores indicative only.\n")

    return dataset


def query_rag(question: str) -> dict:
    """Send a single question to the RAG pipeline's /chat endpoint. Returns the JSON response containing:
        - answer    : the LLM-generated answer
        - contexts  : the retrieved chunks used to ground the answer
        
    Raises RuntimeError on any non-200 response so that evaluation fails loudly instead of producing misleading partial results."""

    response = requests.post(
        RAG_API_URL,
        json={"question": question},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"RAG API returned {response.status_code}: {response.text}")
    return response.json()


def collect_evaluation_data() -> list[dict]:
    """For each test question, query the RAG pipeline and assemble the 4-field record that RAGAS requires for evaluation:
        - question      : the input question
        - answer        : the RAG-generated answer
        - contexts      : the retrieved chunks
        - ground_truth  : the expected answer (from test dataset)

    Uses tqdm to show a clean progress bar instead of verbose per-question prints — the per-question detail still ends up in the JSON output."""

    dataset = load_test_dataset(TEST_DATASET_PATH)

    eval_records = []
    # tqdm wraps the iterable and renders a live progress bar
    for item in tqdm(dataset, desc="📡 Generating RAG responses", unit="q"):
        rag_response = query_rag(item["question"])
        eval_records.append({
            "question": item["question"],
            "answer": rag_response["answer"],
            "contexts": rag_response["contexts"],
            "ground_truth": item["ground_truth"],
        })

    return eval_records


def setup_ragas_evaluator():
    """ Initialize the two RAGAS dependencies:
        1. Judge LLM   : Claude (scores faithfulness, relevancy, precision, recall)
        2. Embeddings  : HuggingFace MiniLM (used for semantic similarity in Answer Relevancy metric)

    Both are wrapped with RAGAS LangChain wrappers so RAGAS can call them through its standard interface."""

    claude_llm = ChatAnthropic(
        model=JUDGE_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,        # Deterministic scoring — same input, same score
        max_tokens=1024,
    )
    wrapped_llm = LangchainLLMWrapper(claude_llm)

    hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    wrapped_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    print(f"✅ Judge LLM + embeddings ready ({JUDGE_MODEL})")
    return wrapped_llm, wrapped_embeddings


def run_full_evaluation(records, wrapped_llm, wrapped_embeddings):
    """ Run all 4 RAGAS metrics on the collected records in a single pass.
    RAGAS converts our list of dicts into a HuggingFace Dataset, then
    iterates over each record and calls the judge LLM to produce per-metric
    scores. RAGAS prints its own internal progress bar during this step.
    Returns the RAGAS Result object (pandas DataFrame underneath)."""

    eval_dataset = Dataset.from_list(records)

    print(f"\n🧪 Running RAGAS evaluation ({len(METRICS)} metrics)...")
    print(f"   ⏱  This may take 2-4 minutes...\n")

    return evaluate(
        dataset=eval_dataset,
        metrics=METRICS,
        llm=wrapped_llm,
        embeddings=wrapped_embeddings,
    )


def score_emoji(score: float) -> tuple[str, str]:
    """ Map a numeric score to a visual emoji indicator using THRESHOLDS.
    Returns a (emoji, formatted_score_string) tuple — used by the console
    aggregate summary to give an at-a-glance read on each metric's health.
    NaN scores are returned with ❓ to clearly signal a calculation failure
    instead of being silently treated as zero."""

    if pd.isna(score):
        return "❓", "NaN "
    if score >= THRESHOLDS["excellent"]:
        return "✅", f"{score:.4f}"
    if score >= THRESHOLDS["good"]:
        return "🟢", f"{score:.4f}"
    if score >= THRESHOLDS["poor"]:
        return "⚠️ ", f"{score:.4f}"
    return "❌", f"{score:.4f}"


def find_metric_columns(results_df: pd.DataFrame) -> dict[str, str]:
    """ Defensive column-name resolver — handles RAGAS API naming variations across versions.
    Problem this solves:
        RAGAS may return a column as "faithfulness" in one version, "Faithfulness" in another, or "faithfulness_score" in a third.
        Hardcoding any one name makes the script brittle.

    Strategy:
        1. Try exact match (fast path for normal case)
        2. Fall back to fuzzy match: normalize both sides (lowercase, strip underscores) and check for substring containment

    Returns a mapping: {standard_metric_name: actual_dataframe_column} that the rest of the pipeline uses to safely access scores."""

    metric_cols = {}
    for metric in METRICS:
        name = metric.name

        # Strategy 1: Exact match (most common case)
        if name in results_df.columns:
            metric_cols[name] = name
            continue

        # Strategy 2: Fuzzy match (handles version/naming drift)
        normalized = name.replace("_", "").lower()
        candidates = [
            c for c in results_df.columns
            if normalized in c.lower().replace("_", "")
        ]
        if candidates:
            metric_cols[name] = candidates[0]
    return metric_cols


def build_summary(result) -> dict:
    """ Transform the raw RAGAS result into a clean, structured summary dict ready for both console display and JSON persistence.
    This function is a PURE DATA BUILDER — it does NOT print anything. Display logic lives in print_aggregate_summary(); persistence logic
    lives in save_results(). This separation of concerns keeps each function single-purpose and easy to test/modify.

    Returns a dict with three keys:
        - overall       : mean score per metric across all questions
        - per_question  : list of per-question score dicts (for JSON detail)
        - sample_size   : number of questions evaluated (for context and confidence interpretation)"""

    results_df = result.to_pandas()
    metric_cols = find_metric_columns(results_df)

    # ---- Step 1: Compute overall (mean) score for each metric ----
    # Aggregating per-question scores into a single headline number per metric.
    overall_scores = {}
    for metric_name, col in metric_cols.items():
        mean_score = results_df[col].mean()
        overall_scores[metric_name] = (
            float(mean_score) if not pd.isna(mean_score) else None
        )

    # ---- Step 2: Collect per-question scores (for JSON detail only) ----
    # RAGAS sometimes names the question column "question", "user_input",
    # "query", or "input" depending on version — we probe for whichever exists.
    question_candidates = ["question", "user_input", "query", "input"]
    question_col = next(
        (c for c in question_candidates if c in results_df.columns),
        None,
    )

    per_question_records = []
    for i, row in results_df.iterrows():
        q_record = {
            "index": i + 1,
            "question": str(row[question_col]) if question_col else None,
        }
        # Add each metric's per-question score to the record
        for metric_name, col in metric_cols.items():
            score = row[col]
            q_record[metric_name] = (
                float(score) if not pd.isna(score) else None
            )
        per_question_records.append(q_record)

    # ---- Step 3: Pack everything into a single summary dictionary ----
    # This dict is the canonical handoff between evaluation and output stages.
    return {
        "overall": overall_scores,
        "per_question": per_question_records,
        "sample_size": len(results_df),
    }


def print_aggregate_summary(summary: dict):
    """ Print ONLY the aggregate (mean) scores to the console. Per-question detail is intentionally NOT printed here — it lives in
    the JSON output. This keeps the console clean and demo-friendly: one glance shows the overall health of each metric."""

    print("\n" + "=" * 60)
    print("📊 Aggregate Scores")
    print("=" * 60)
    for metric_name, score in summary["overall"].items():
        if score is None:
            emoji, score_str = "❓", "N/A"
        else:
            emoji, score_str = score_emoji(score)
        # :22s pads the metric name to 22 chars for vertical alignment
        print(f"   {emoji} {metric_name:22s} : {score_str}")
    print("=" * 60)


def save_results(summary: dict) -> Path:
    """ Persist the full evaluation results to a timestamped JSON file in eval_results/.

    The JSON contains everything needed for later analysis:
        - run metadata (timestamp, models used)
        - aggregate scores
        - per-question scores
        - threshold definitions
        - plain-English metric meanings
        - small-sample warning note (if applicable)

    Timestamped filenames mean every run produces a new artifact — nothing is overwritten, so trend tracking across runs is possible."""

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Assemble the full output payload — everything a future reader needs
    output = {
        "timestamp": timestamp,
        "judge_model": JUDGE_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "sample_size": summary["sample_size"],
        "overall_scores": summary["overall"],
        "per_question_scores": summary["per_question"],
        "thresholds": THRESHOLDS,
        "metric_meanings": METRIC_MEANINGS,
        "note": (
            "Small sample size — scores are indicative only"
            if summary["sample_size"] < 10
            else None
        ),
    }

    # Write JSON with indent=2 for human readability and
    # ensure_ascii=False to preserve any non-ASCII characters in questions.
    json_path = RESULTS_DIR / f"ragas_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved:")
    print(f"   📄 JSON: {json_path}")

    return json_path


# ================================================================
# Entry Point
# ================================================================

def main():
    """ Orchestrate the full RAGAS evaluation pipeline end-to-end.
    Pipeline stages:
        1. Setup    : Initialize judge LLM + embeddings
        2. Collect  : Query RAG endpoint for each test question
        3. Evaluate : Run all 4 RAGAS metrics in one pass
        4. Summarize: Build clean summary dict from raw results
        5. Display  : Print aggregate scores to console
        6. Persist  : Save full detail to timestamped JSON
    Each stage is delegated to a single-responsibility function, which makes the pipeline easy to read, test, and modify."""

    print("🚀 Starting RAGAS evaluation\n")

    # Stage 1: Setup judge LLM + embeddings
    wrapped_llm, wrapped_embeddings = setup_ragas_evaluator()

    # Stage 2: Collect (question, answer, contexts, ground_truth) records
    records = collect_evaluation_data()

    # Stage 3: Run all 4 RAGAS metrics in a single evaluation pass
    result = run_full_evaluation(records, wrapped_llm, wrapped_embeddings)

    # Stage 4: Transform raw RAGAS result into clean summary dict
    summary = build_summary(result)

    # Stage 5: Display aggregate scores on console (clean, headline view)
    print_aggregate_summary(summary)

    # Stage 6: Persist full detail (overall + per-question) to JSON
    save_results(summary)

    print("\n✅ Evaluation complete. See JSON file for per-question details.\n")


if __name__ == "__main__":
    main()