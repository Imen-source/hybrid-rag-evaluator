# Hybrid RAG Evaluator

Hybrid RAG Evaluator is a local-first evaluation pipeline for retrieval-augmented generation (RAG) systems. It combines lightweight LLM judging with deterministic scoring so the pipeline remains useful even on modest hardware using local models such as phi through Ollama.

## Features

- **Dataset Loading**: JSON dataset with question, context, and answer samples
- **Hybrid Evaluation**: Combines LLM judging with deterministic semantic similarity and keyword overlap
- **Metrics Computed**: correctness, relevance, groundedness, hallucination, explanation, semantic_similarity, keyword_overlap, confidence
- **Aggregate Metrics**: Average scores plus hallucination_rate
- **Caching**: LLM responses and embeddings cached for faster repeated runs
- **Provider Support**: Local Ollama (phi) and OpenAI API (optional)
- **Fallbacks**: Deterministic results if LLM fails or Ollama is unavailable

## Repository Layout

```
rag_evaluator/
├── cli.py               # Entry point
├── metaflow_pipeline.py # Pipeline runner (Metaflow + local fallback)
├── evaluator.py         # LLM evaluation, hybrid scoring, JSON handling
├── dataset_eval.py      # Dataset loading and aggregate metrics
├── prompts.py           # Evaluation prompt templates
├── schemas.py           # Pydantic schemas for structured output
├── similarity.py        # Semantic similarity & keyword overlap helpers
├── cache.py             # Clear cached artifacts
├── examples/            # Sample datasets
└── results.json         # Aggregated results
```

### Optional / Cleanup

- `metrics.py` → can be merged into dataset_eval.py (thin wrapper)
- Delete: `test_openai_key.py`, `debug.log`
- Ignore: `__pycache__`, `eval_cache/`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install ollama huggingface_hub sentence-transformers
```

## Usage

### Run with Ollama locally (phi)

```bash
python cli.py --provider ollama --model phi
```

**Optional flags:**

```bash
--use-hybrid true    # Combine LLM + semantic scoring
--num-runs 2         # Number of evaluation passes per sample
--verbose true       # Mirror debug info to console
```

### Run with OpenAI API

```bash
python cli.py --provider openai --model gpt-4o-mini
```

Set your API key:

```powershell
$env:OPENAI_API_KEY="your_openai_key_here"
```

## Hybrid Scoring Logic

Final score combines:

```
0.5 * llm_correctness + 0.3 * semantic_similarity + 0.2 * keyword_overlap
```

If LLM output is malformed or fails → deterministic scoring still returns a valid JSON payload

## Example Dataset

```json
[
  {
    "question": "What is the capital of France?",
    "context": "France's capital city is Paris.",
    "answer": "Paris is the capital of France."
  }
]
```

## Example Output

### Sample-level

```json
{
  "correctness": 0.9234,
  "relevance": 0.9235,
  "groundedness": 0.9234,
  "hallucination": false,
  "explanation": "Paris is the capital city of France.",
  "semantic_similarity": 0.9114,
  "keyword_overlap": 0.75,
  "confidence": 0.9645
}
```

### Aggregate metrics

```json
{
  "avg_correctness": 0.7877,
  "avg_relevance": 0.8552,
  "avg_groundedness": 0.8369,
  "avg_semantic_similarity": 0.8002,
  "avg_keyword_overlap": 0.6548,
  "avg_confidence": 0.9201,
  "hallucination_rate": 0.0,
  "sample_count": 3
}
```

## Logging & Debugging

- Raw LLM output, retries, and fallbacks logged to `debug.log`
- Use `--verbose true` to mirror logs to console
- Malformed output triggers retry with stricter prompt

## Caching

- **LLM results**: `eval_cache/`
- **Embeddings**: `eval_cache/embeddings/`

### Clear cache

```bash
python cache.py --cache-dir eval_cache
```

## Known Issues

- Small local models (e.g., phi) may produce inconsistent explanations
- Semantic similarity helps but is not a replacement for a gold reference
- Keyword overlap is lightweight; paraphrases can reduce accuracy
- Ollama must be running; otherwise deterministic fallback is used
- Windows users may see HuggingFace symlink warnings — non-blocking

## Future Improvements

- Support for larger local models if more system memory is available
- Improved fallback handling for multi-step or long-context questions
- Optional visualizations for metric distributions
- Better error reporting for hybrid mode inconsistencies
- Potential integration with cloud LLMs beyond OpenAI

## Sanity Checks

### Syntax check

```bash
python -m py_compile cli.py dataset_eval.py evaluator.py similarity.py schemas.py prompts.py metaflow_pipeline.py cache.py
```

### Reset cache

```bash
python cache.py
```

## Goals Achieved

- ✅ Fully functional local-first RAG evaluation pipeline
- ✅ Supports multiple LLM providers: OpenAI + Ollama (phi)
- ✅ Structured JSON output compatible with downstream metrics
- ✅ Optional hybrid semantic scoring
- ✅ Clean, modular, production-ready code