---
title: AgentEval Studio
emoji: 🧪
colorFrom: indigo
colorTo: purple
sdk: docker
sdk_version: "3.10"
pinned: false
license: mit
short_description: Eval dashboard for RAG apps and AI agents
---

# 🧪 AgentEval Studio

> Open-source evaluation and observability dashboard for RAG apps and AI agents.

[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/BalajiBaluP/AgentEval-Studio)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B)](https://streamlit.io)

---

## What it does

Stop guessing if your AI app works. Start measuring it.

AgentEval Studio sends test questions to your AI app, scores the responses automatically, and shows you exactly where it fails across 5 metrics, per question, with version comparison.

| Metric | What it measures |
|---|---|
| **Faithfulness** | Did the AI use real info or hallucinate? |
| **Answer Relevancy** | Did it actually answer the question? |
| **Context Recall** | Did it retrieve the right material? |
| **ROUGE-L** | Word overlap with expected answer |
| **Keyword Coverage** | Key terms present in response |

---

## Live Results — RAG Debate Arena

| Metric | Score |
|---|---|
| Pass Rate | 100% |
| Faithfulness | 86% |
| Answer Relevancy | 88% |
| Context Recall | 88% |
| Avg Latency | 37s |

---

## Stack

- **FastAPI** — evaluation backend
- **Streamlit** — dashboard UI
- **SQLite** — run history storage
- **Groq (llama-3.1-8b)** — free LLM-as-judge
- **Docker** — HuggingFace Spaces deployment

---

## Quickstart

```bash
pip install -r requirements.txt

# Terminal 1
uvicorn app.main:app --reload --port 8000

# Terminal 2
streamlit run ui/streamlit_app.py
```

Set your free Groq key from [console.groq.com](https://console.groq.com):
```bash
export GROQ_API_KEY=gsk_...
```

---

## ⚠️ HF Free Tier Note

The demo evaluates [RAG Debate Arena](https://huggingface.co/spaces/BalajiBaluP/RAG_Debate) hosted on HF Free tier which **sleeps after inactivity**. If you see errors on first run, wait 30 seconds and re-run — the Space wakes automatically. Scores on retry will be normal.

## Your app needs one endpoint

```json
POST /your-endpoint
{ "question": "..." }

→ { "answer": "...", "contexts": ["chunk1", "chunk2"] }
```

---

## How to use

1. **Datasets** — Paste test questions as JSON
2. **New Run** — Enter your app URL, start evaluation
3. **Results** — Scores per question, inspect failures
4. **Leaderboard** — Compare prompt versions side by side

---

## Score guide

| Score | Meaning |
|---|---|
| >70% | Working well |
| 40-70% | Needs prompt tuning |
| <40% | Something broken |

---

## Structure

```
agenteval-studio/
├── app/
│   ├── api/          ← dataset + run endpoints
│   ├── core/         ← metrics, runner, evaluator
│   └── models/       ← DB + schemas
├── ui/
│   └── streamlit_app.py
├── datasets/
├── Dockerfile
└── start.sh
```

---

## Related

- [RAG Debate Arena](https://huggingface.co/spaces/BalajiBaluP/RAG_Debate) — AI app evaluated using this studio

*Built by [Balaji Parlapalli](https://github.com/BalajiParlapalli)*
