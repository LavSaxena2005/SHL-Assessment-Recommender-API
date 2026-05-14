# SHL Assessment Recommender

A production-ready conversational AI system for discovering SHL Individual Test Solutions. Built with FastAPI, RAG architecture (BM25 + FAISS), and GPT-4o-mini / Gemini Flash.

---

## Architecture Overview

```
User Query
    │
    ▼
FastAPI /chat (stateless)
    │
    ├─► Hybrid Retriever ──────────────────────────────┐
    │     ├─ BM25 (keyword match)                      │
    │     └─ FAISS + Sentence-Transformers (semantic)  │
    │              [all-MiniLM-L6-v2]                  │
    │                                                   │
    ├─◄── Retrieved Assessments (top-10) ◄─────────────┘
    │
    ├─► LLM (GPT-4o-mini / Gemini Flash)
    │     ├─ System prompt with catalog context
    │     ├─ Full conversation history
    │     └─ Intent: clarify | recommend | refine | compare | refuse
    │
    ├─► Hallucination Validator
    │     └─ Filters output against catalog.json
    │
    └─► Structured JSON Response
          ├─ reply (string)
          ├─ recommendations (validated list)
          └─ end_of_conversation (bool)
```

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- OpenAI API key **or** Google Gemini API key

### 1. Clone and Install

```bash
git clone <repo-url>
cd shl-recommender
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API key:
# OPENAI_API_KEY=sk-...
# or
# GEMINI_API_KEY=... and LLM_PROVIDER=gemini
```

### 3. Build the Catalog and Indexes

```bash
# Step 1: Scrape SHL catalog (or use pre-built curated data)
python -m app.scraper.scraper

# Step 2: Build retrieval indexes (BM25 + FAISS)
python -c "from app.retriever.hybrid_retriever import get_retriever; get_retriever().build_index()"
```

### 4. Start the Server

```bash
# Development (with auto-reload)
ENV=development python -m app.main

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`  
Swagger docs: `http://localhost:8000/docs`

---

## API Reference

### GET /health

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{"status": "ok"}
```

---

### POST /chat

Conversational endpoint. Send full conversation history with every request (stateless API).

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hiring Java developer"}
    ]
  }'
```

**Response:**
```json
{
  "reply": "What seniority level is this role? (Entry, Mid-level, or Senior)",
  "recommendations": [],
  "end_of_conversation": false
}
```

**Multi-turn example:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hiring Java developer"},
      {"role": "assistant", "content": "What seniority level is this role?"},
      {"role": "user", "content": "Mid-level, 3+ years experience"}
    ]
  }'
```

**Response:**
```json
{
  "reply": "Based on your requirements, here are the most relevant SHL assessments for a mid-level Java developer...",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/assessments/skills-and-knowledge/java-8/",
      "test_type": "K"
    },
    {
      "name": "Automata Pro",
      "url": "https://www.shl.com/solutions/products/assessments/skills-and-knowledge/automata-pro/",
      "test_type": "K"
    },
    {
      "name": "Verify Numerical Reasoning",
      "url": "https://www.shl.com/solutions/products/assessments/cognitive-assessments/verify-numerical-reasoning/",
      "test_type": "A"
    }
  ],
  "end_of_conversation": true
}
```

---

### Test Type Reference

| Code | Meaning |
|------|---------|
| `A`  | Ability / Cognitive |
| `P`  | Personality |
| `K`  | Knowledge / Skills |
| `S`  | Situational Judgement |
| `B`  | Biodata |
| `W`  | Work Sample |

---

## Evaluation

### Run Retrieval Evaluation (no API key needed)

```bash
python -m app.evaluation.eval_suite
```

### Run Full Evaluation (requires running API)

```bash
# Terminal 1: Start API
uvicorn app.main:app --port 8000

# Terminal 2: Run evals
python -m app.evaluation.eval_suite http://localhost:8000
```

**Metrics:**
- **Recall@10** — Are expected assessments in top 10 results?
- **Groundedness** — Are all returned assessments in the catalog?
- **Hallucination Rate** — Are any invented assessments returned?
- **Behavioral Probes** — Correct intent routing (clarify/refuse/recommend)?

---

## Deployment

### Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  shl-recommender
```

### Render

1. Push code to GitHub
2. Connect repo in Render dashboard
3. Select `render.yaml` configuration
4. Add `OPENAI_API_KEY` as environment variable
5. Deploy

---

## Design Decisions

### Stateless API
The API accepts full conversation history with every request. No server-side session storage. This enables horizontal scaling and simplifies deployment.

### Hybrid Retrieval (RAG)
- **BM25** catches exact keyword matches (e.g., "Java 8", "OPQ32")
- **Semantic search** (FAISS + all-MiniLM-L6-v2) catches conceptual matches (e.g., "numerical skills" → Verify Numerical Reasoning)
- **RRF fusion** (60% semantic, 40% BM25) balances precision and recall

### Hallucination Prevention
Every LLM-generated recommendation name is validated against:
1. Retrieved catalog (top-10 results from hybrid search)
2. Full catalog (fallback fuzzy match)

Unrecognized names are silently dropped. The system will never return a URL or assessment that doesn't exist in `catalog.json`.

### Intent Detection
The LLM classifies every turn into one of 5 intents:
- `clarify` — vague query; ask one focused question
- `recommend` — specific enough; return top assessments  
- `refine` — add/change constraints on previous recommendation
- `compare` — explain differences between named assessments
- `refuse` — out-of-scope request (competitors, salary advice, etc.)

### 8-Turn Limit
After 8 user turns, the system always provides recommendations and ends the conversation to prevent infinite clarification loops.

---

## Project Structure

```
shl-recommender/
├── app/
│   ├── main.py              # FastAPI application
│   ├── routes/
│   │   └── api.py           # /health and /chat endpoints
│   ├── services/
│   │   └── conversation.py  # Conversation orchestration
│   ├── retriever/
│   │   └── hybrid_retriever.py  # BM25 + FAISS hybrid search
│   ├── prompts/
│   │   └── templates.py     # LLM prompt templates
│   ├── scraper/
│   │   └── scraper.py       # SHL catalog scraper
│   ├── evaluation/
│   │   └── eval_suite.py    # Recall@10, groundedness, behavioral tests
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   ├── catalog/
│   │   ├── catalog.json     # Scraped catalog data
│   │   ├── faiss.index      # FAISS vector index (generated)
│   │   ├── embeddings.npy   # Embeddings cache (generated)
│   │   └── bm25.pkl         # BM25 index (generated)
│   └── utils/
│       └── catalog_utils.py # Catalog helpers
├── requirements.txt
├── Dockerfile
├── render.yaml
├── .env.example
└── README.md
```
Deploy Link https://shl-assessment-recommender-api-1.onrender.com/
