"""
Metrics engine.
- Heuristic: keyword coverage, ROUGE-L, token count
- LLM-judge (Gemini): faithfulness, answer relevancy, context recall
"""
import re
import math
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─── Heuristic metrics ────────────────────────────────────────────────────────

def keyword_coverage(response: str, keywords: List[str]) -> float:
    """Fraction of expected keywords present in response (case-insensitive)."""
    if not keywords:
        return 1.0
    resp_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in resp_lower)
    return round(hits / len(keywords), 4)


def rouge_l_score(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 between hypothesis and reference."""
    if not hypothesis or not reference:
        return 0.0
    h_tokens = hypothesis.lower().split()
    r_tokens = reference.lower().split()
    lcs = _lcs_length(h_tokens, r_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(h_tokens)
    recall = lcs / len(r_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def _lcs_length(a: list, b: list) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(a), len(b)
    # Space-optimised DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, math.ceil(len(text) / 4))


# ─── Gemini LLM-judge metrics ─────────────────────────────────────────────────

def _gemini_score(prompt: str, api_key: str) -> float:
    """Call Gemini Flash and parse a 0-1 float from the response.
    Supports both legacy AIzaSy keys and new AQ. auth keys.
    """
    try:
        import requests as _requests
        # Detect key type
        if api_key.startswith("AQ."):
            # New OAuth2-style auth key — use Bearer token in header
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            r = _requests.post(url, headers=headers, json=payload, timeout=30)
        else:
            # Legacy AIzaSy key — use query param
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            r = _requests.post(url, headers=headers, json=payload, timeout=30)

        if r.status_code != 200:
            logger.error("Gemini API error %d: %s", r.status_code, r.text[:200])
            return 0.0

        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        match = re.search(r"\b([01](?:\.\d+)?|0\.\d+)\b", text)
        if match:
            return round(float(match.group(1)), 4)
        logger.warning("Gemini returned non-numeric: %s", text[:100])
        return 0.5
    except Exception as e:
        logger.error("Gemini call failed: %s", e)
        return 0.0


def faithfulness_score(
    response: str,
    contexts: List[str],
    api_key: Optional[str] = None
) -> float:
    """
    Are claims in the response grounded in the retrieved context?
    Returns 0-1. Falls back to keyword overlap if no API key.
    """
    if not contexts:
        return 0.0
    combined_context = "\n\n".join(contexts)

    if not api_key:
        # Heuristic fallback: word overlap ratio
        resp_words = set(response.lower().split())
        ctx_words = set(combined_context.lower().split())
        overlap = len(resp_words & ctx_words)
        return round(overlap / max(len(resp_words), 1), 4)

    prompt = f"""You are an evaluation judge. Score the FAITHFULNESS of the answer.

Retrieved context:
\"\"\"
{combined_context[:2000]}
\"\"\"

Answer:
\"\"\"
{response[:1000]}
\"\"\"

Faithfulness measures whether every claim in the answer is supported by the context.
Respond with ONLY a single decimal number between 0.0 and 1.0.
1.0 = fully grounded, 0.0 = completely hallucinated."""
    return _gemini_score(prompt, api_key)


def answer_relevancy_score(
    response: str,
    question: str,
    api_key: Optional[str] = None
) -> float:
    """Does the answer actually address the question?"""
    if not api_key:
        # Heuristic: word overlap between question and answer
        q_words = set(question.lower().split())
        a_words = set(response.lower().split())
        overlap = len(q_words & a_words)
        return round(overlap / max(len(q_words), 1), 4)

    prompt = f"""You are an evaluation judge. Score ANSWER RELEVANCY.

Question: {question}

Answer: {response[:1000]}

Does the answer directly and completely address the question?
Respond with ONLY a single decimal between 0.0 and 1.0.
1.0 = perfectly relevant, 0.0 = completely irrelevant."""
    return _gemini_score(prompt, api_key)


def context_recall_score(
    contexts: List[str],
    expected_answer: str,
    api_key: Optional[str] = None
) -> float:
    """Did retrieval surface the context needed to answer correctly?"""
    if not contexts:
        return 0.0
    combined_context = "\n\n".join(contexts)

    if not api_key:
        # Heuristic: how much of the expected answer is present in context
        exp_words = set(expected_answer.lower().split())
        ctx_words = set(combined_context.lower().split())
        overlap = len(exp_words & ctx_words)
        return round(overlap / max(len(exp_words), 1), 4)

    prompt = f"""You are an evaluation judge. Score CONTEXT RECALL.

Expected answer: {expected_answer[:500]}

Retrieved context:
\"\"\"
{combined_context[:2000]}
\"\"\"

Does the retrieved context contain enough information to derive the expected answer?
Respond with ONLY a single decimal between 0.0 and 1.0.
1.0 = all needed info present, 0.0 = no relevant info."""
    return _gemini_score(prompt, api_key)


def compute_overall(scores: dict) -> float:
    """Weighted average of all available scores."""
    weights = {
        "keyword_coverage": 0.15,
        "rouge_l": 0.15,
        "faithfulness": 0.30,
        "answer_relevancy": 0.25,
        "context_recall": 0.15,
    }
    total, weight_sum = 0.0, 0.0
    for key, w in weights.items():
        val = scores.get(key)
        if val is not None:
            total += val * w
            weight_sum += w
    return round(total / weight_sum, 4) if weight_sum else 0.0
