---
title: AgentEval Studio
emoji: 🧪
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Evaluation & observability dashboard for RAG apps and AI agents
---

# 🧪 AgentEval Studio

> Open-source evaluation and observability dashboard for AI agents and RAG systems.

Benchmark your RAG app or AI agent across **groundedness, retrieval quality, latency, token usage, and prompt versions** using golden datasets and automated metrics.

## Features (V1 + V2)

- **Upload golden datasets** (CSV or JSON)
- **Run evaluations** against any HTTP endpoint
- **6 metrics**: Faithfulness, Answer Relevancy, Context Recall, ROUGE-L, Keyword Coverage, Latency
- **LLM-as-judge** via Gemini Flash (optional, bring your own key)
- **Version leaderboard**: compare prompt/model versions side-by-side
- **Per-case drill-down**: inspect each input/response/context
- **Export results** as CSV

## Target API Contract

Your RAG app should expose a POST endpoint:

```json
POST /your/endpoint
{ "question": "...", "app_version": "v1" }

→ { "answer": "...", "contexts": ["chunk1", "chunk2"], "token_count": 123 }
```

`contexts` and `token_count` are optional but enable richer metrics.

## Stack

- **FastAPI** — evaluation backend
- **Streamlit** — dashboard UI
- **SQLite** — result storage
- **Gemini Flash** — LLM-judge (optional)
- **Custom metrics** — ROUGE-L, keyword coverage, heuristic fallbacks

## Local Development

```bash
pip install -r requirements.txt

# Terminal 1 — Backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — UI
streamlit run ui/streamlit_app.py
```

## Adding Your RAG Debate Arena

1. Deploy your RAG app and note the `/query` endpoint URL
2. Upload `datasets/rag_debate_sample.json` as a starting dataset
3. Run an eval pointing at your endpoint
4. Iterate on prompts and compare versions in the Leaderboard
