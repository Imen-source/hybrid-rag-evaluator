"""Prompt templates used by the evaluator."""

from __future__ import annotations


EVALUATION_SYSTEM_PROMPT = (
    "You are a strict evaluator for retrieval-augmented generation answers. "
    "Score the answer against the provided question and context. "
    "Return only valid JSON with no markdown, no prose, and no code fences."
)


def build_evaluation_prompt(question: str, context: str, answer: str) -> str:
    """Create the user prompt consumed by the judge model."""
    return f"""
Evaluate the answer using the context provided.

You must return ONLY a valid JSON object.
Do not add markdown.
Do not add code fences.
Do not add explanations before or after the JSON.
Do not omit any required key.

Return exactly this JSON schema:
{{
  "correctness": float between 0 and 1,
  "relevance": float between 0 and 1,
  "groundedness": float between 0 and 1,
  "hallucination": boolean,
  "explanation": string,
  "confidence": float between 0 and 1
}}

Scoring guidance:
- correctness measures factual accuracy relative to the question
- relevance measures how directly the answer addresses the question
- groundedness measures whether the answer is supported by the context
- hallucination is true when the answer contains unsupported or invented claims
- confidence measures how certain the evaluator is in its own judgment

Question: {question}
Context: {context}
Answer: {answer}
""".strip()
