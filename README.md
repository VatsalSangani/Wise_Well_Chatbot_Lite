# 🏥 WiseWell Medical Chatbot - Lite

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/react-18+-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)

> A medical-information assistant that answers health questions using
> retrieval-augmented generation (RAG) over a PubMed abstract corpus — dense
> retrieval (Pinecone) + Claude Haiku synthesis, wrapped in safety guardrails so
> it explains evidence in plain language but never acts like a doctor.

> 📐 **Design rationale & evaluation methodology → [DESIGN.md](DESIGN.md)** — the
> *why* behind every architectural and eval decision (Pinecone vs FAISS, the 0.58
> evidence fork, the conservative multi-turn resolver, the RAG-eval triad, Langfuse
> observability), grounded in the actual code, with Mermaid diagrams.

---
[![Live Demo](https://img.shields.io/badge/AWS%20EC2-Live%20Demo-orange)](https://46kclo66ry2ptmxgwckb6ben4a0mdfsq.lambda-url.eu-west-2.on.aws/?p=wiswell)

> **Try it live** → [Click here](https://46kclo66ry2ptmxgwckb6ben4a0mdfsq.lambda-url.eu-west-2.on.aws/?p=wiswell)

![Demo UI using React](https://github.com/VatsalSangani/Wise_Well_Chatbot_Lite/blob/main/WiseWell%20Screenshot.png)

---

## ✨ Features

- **Dense RAG over PubMed** — Pinecone vector search (`all-MiniLM-L6-v2`, 384-dim,
  ~619k abstract chunks); abstract text rehydrated from a local SQLite store.
- **Claude Haiku synthesis** — direct Anthropic API (`claude-haiku-4-5`), with
  inline PMID citations in RAG mode.
- **Structured answers** — every answer is JSON-shaped: `summary`, `key_points`,
  `explanation`, `when_to_see_doctor`, `citations`, `follow_up_suggestion` —
  assembled into clean markdown.
- **Six safety decisions** — `ANSWER` · `ABSTAIN` · `REFUSE` · `ESCALATE` ·
  `CHITCHAT` · `CLARIFY`. Emergencies, refusals, and crisis support are
  **deterministic code-inserted strings**, never LLM-generated.
- **Multi-turn** — the frontend sends the last few turns; a **conservative
  reference-only resolver** rewrites follow-ups ("how is it treated" → "how is
  rheumatoid arthritis treated") without injecting unrelated context.
- **Evidence-confidence fork** — strong retrieval → cited RAG answer; weak
  retrieval → general-knowledge answer with a disclaimer (no fabricated citations).
- **RAG evaluation** — an independent GPT-4o-mini judge scores faithfulness,
  answer-relevancy, context-relevancy (+ hallucination-rate + readability),
  per-query, **off the request path**.
- **Observability** — Langfuse v4 traces (retrieval, synthesis, tokens/cost, eval
  scores) for every query.

---

## 🏗️ Architecture (live request path)

```
User query (+ optional recent turns)
    ↓
Query resolver ── self-contained? pass through · genuine reference? Haiku substitution-only rewrite
    ↓
Red-flag detector ──(active emergency / self-harm)──▶ ESCALATE (deterministic 999 / Samaritans)
    ↓ (none)
Front-door router ──(greeting/meta)──▶ CHITCHAT (deterministic friendly reply)
    ↓ (medical)
Safety intent ──(individual Dx / Rx / dosing)──▶ REFUSE (deterministic decline)
    ↓ (ok)
Pinecone dense retrieval (+ lab-value query rewrite) → SQLite text rehydration
    ↓
Evidence-confidence fork:  top_score ≥ 0.58  AND  ≥ 3 distinct PMIDs  AND  evidence_gate=pass
    ├─ confident ─▶ RAG mode     → synthesize from evidence + citations
    └─ weak      ─▶ general mode  → synthesize from model knowledge + disclaimer
    ↓
Structured JSON synthesis → markdown + code-inserted trailers (disclaimer / soft-defer / follow-up)
    ↓
Response  ·  then (off request path) BackgroundTask → RAG eval scores attached to the Langfuse trace
```

> Full labeled Mermaid diagrams (request flow + eval flow) and the reasoning behind
> each node are in **[DESIGN.md](DESIGN.md)**.

The live path is `backend/routes/query.py → orchestration/service.py →
scripts/qa_check.py::answer_query`. (A LangGraph definition exists in
`orchestration/graph.py`/`nodes.py` from an earlier design but is **not** on the
live request path.)

---

## 🧩 Stateless server, client-supplied history

The server is **stateless per request** — every `/query` is independent and keeps
no per-user state, so it scales horizontally behind a load balancer with no session
affinity. What lives in RAM is shared, read-only, and identical across instances
(the Pinecone client, the lazily-loaded MiniLM embedder, the read-only SQLite text
store, the guardrails config).

**Multi-turn works without server state:** the client sends the last few turns with
each request (`QueryRequest.history`, `backend/schemas.py`), and the resolver
(`orchestration/query_resolver.py`) collapses a follow-up into a standalone query
*before* the guardrails run — so safety checks still evaluate a single
self-contained message. The resolver is deliberately conservative: self-contained
queries pass through unchanged (`is coffee bad for me` stays itself), and only
genuine references/fragments are resolved (`how is it treated` → the prior topic).

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ · Node.js 18+
- A Pinecone index (`wisewell-abstracts`) and a local SQLite text store
  (`py scripts/build_text_store.py`)

### Backend environment (`.env`, loaded via python-dotenv)
```env
# Required
ANTHROPIC_API_KEY=sk-ant-...          # Claude Haiku synthesis (direct API)
PINECONE_API_KEY=pcsk_...             # dense retrieval
PINECONE_INDEX_NAME=wisewell-abstracts

# Optional — evaluation & observability (degrade gracefully if absent)
OPENAI_API_KEY=sk-...                 # GPT-4o-mini RAG-eval judge
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# Server / flags
WISEWELL_PORT=8502
WISEWELL_HOST=0.0.0.0
WISEWELL_ALLOWED_ORIGINS=http://localhost:5173   # add prod origin in prod
ENABLE_LLM_SYNTHESIS=true
EVAL_SAMPLE_RATE=1.0                  # fraction of ANSWER queries evaluated
```

If the optional keys are missing, tracing/eval silently no-op — the bot still works.

### Run
```bash
# Backend (port 8502)
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8502

# Frontend (dev server, served under /wiswell-ui/)
cd frontend && npm install && npm run dev
```

**Local dev:** frontend `http://localhost:5173/wiswell-ui/` → backend
`http://localhost:8502` (set in `frontend/.env` via `VITE_API_URL`).

---

## 🧪 Testing & validation

```bash
# Deterministic guardrail / routing checks (no LLM needed for decisions)
py scripts/validate_router.py            # red-flag tiers, router, 50/50, 0.58 fork, safety regression
py scripts/validate_multiturn.py         # multi-turn resolution + safety on follow-ups
py scripts/validate_structured_output.py # structured JSON, citation validation, fallbacks
py scripts/validate_threshold_split.py   # data-driven check of the 0.58 fork bar

# API smoke test
curl -X POST http://localhost:8502/query -H "Content-Type: application/json" \
  -d '{"query": "What is C-reactive protein?"}'
curl http://localhost:8502/health
```

---

## 🎯 Example behaviors

| Query | Decision | Behavior |
|---|---|---|
| "What is rheumatoid arthritis?" | ANSWER (rag) | Cited, structured answer from PubMed evidence |
| "Is coffee bad for me?" | ANSWER (general) | Educational answer + general-knowledge disclaimer |
| "Should I stop my blood pressure medication?" | REFUSE | Warm decline — individual treatment decisions need a doctor |
| "I think I'm having a heart attack" | ESCALATE | Urgent "call 999" message (deterministic) |
| "hi" / "thanks" | CHITCHAT | Friendly reply, no retrieval |
| "Why is this number high?" | CLARIFY | Asks what biomarker / context |

---

## 📁 Project Structure (current)

```
wisewell-backend-production/
├── config.py                         ← constants: model id, 0.58 fork, eval/judge, ports
├── requirements.txt
├── DESIGN.md                         ← design rationale & eval methodology
├── backend/
│   ├── main.py                       ← FastAPI entry (load_dotenv, CORS, lifespan)
│   ├── schemas.py                    ← QueryRequest (query + history) / QueryResponse
│   ├── deps.py                       ← PineconeRetriever singleton
│   └── routes/  health.py · query.py (POST /query) · admin.py
├── orchestration/
│   ├── service.py                    ← run_wisewell_query() wrapper
│   ├── query_resolver.py             ← conservative reference-only multi-turn resolver
│   ├── llm_syntheses.py              ← Claude Haiku structured-JSON synthesis + fallbacks
│   ├── observability.py             ← Langfuse v4 tracing + off-path GPT-4o-mini RAG eval
│   ├── audit_logger.py · bootstrap.py · utils.py · state.py
│   └── graph.py · nodes.py           ← legacy LangGraph (NOT on live path)
├── retrieval/
│   ├── pinecone_retriever.py         ← dense retrieval (drop-in evidence schema)
│   ├── embedder.py                   ← lazy MiniLM seam (all-MiniLM-L6-v2, 384-dim)
│   ├── text_store.py                 ← read-only SQLite text rehydration
│   ├── query_rewrite.py             ← lab-value → concept query rewrite
│   └── hybrid_retriever.py           ← legacy BM25+FAISS (dormant rollback path)
├── guardrails/
│   ├── red_flags.py · router.py · safety_intent.py · responses.py
│   ├── evidence_gate.py · topic_consistency.py · overlap/mechanism gates
│   └── citation_verifier.py · composer_extractive.py · config_loader.py
├── config/guardrails.yaml            ← safety thresholds & retrieval config
├── scripts/
│   ├── qa_check.py                    ← the live pipeline (answer_query)
│   ├── build_text_store.py           ← builds the SQLite text store
│   ├── pinecone_smoke_test.py · validate_*.py
│   └── build_hybrid_indexes.py       ← legacy index builder (rollback path)
├── frontend/                         ← React + Vite + Tailwind (base /wiswell-ui/)
└── kb/                               ← text_store.sqlite + legacy indexes (gitignored, not pushed)
```

---

## 🛠️ Tech Stack

**Backend:** Python 3.9+, FastAPI, sentence-transformers + CPU PyTorch (MiniLM embeddings)
**Retrieval:** Pinecone (dense vectors) + SQLite (text store)
**LLM:** Anthropic API — `claude-haiku-4-5`
**Evaluation:** OpenAI `gpt-4o-mini` (independent RAG-eval judge) · `textstat` (readability)
**Observability:** Langfuse v4 (LLM traces + eval scores); operational metrics via an
external Prometheus/Grafana stack (docker-compose, not part of the request path)
**Frontend:** React 18+, TypeScript, Vite, Tailwind CSS
**Deployment:** AWS EC2 · backend port 8502

> `requirements.txt` active stack: `anthropic`, `pinecone`, `sentence-transformers`
> + CPU `torch`, `langfuse`, `openai`, `textstat`, `fastapi`/`uvicorn`. Legacy
> `faiss-cpu`, `rank-bm25`, `boto3`, `langgraph`/`langsmith` are retained for the
> rollback path / earlier design and are not on the live request path.

---

## 🔒 Medical Disclaimer

> ⚠️ **For informational purposes only.**
> Always consult a qualified healthcare professional for medical advice, diagnosis,
> or treatment. This tool does not provide personal medical recommendations.

---

## 📝 License

MIT License — see [LICENSE](LICENSE).

---

## 📞 Contact

**GitHub:** [@VatsalSangani](https://github.com/VatsalSangani)
**Repository:** [Wise_Well_Chatbot_Lite](https://github.com/VatsalSangani/Wise_Well_Chatbot_Lite)

---

## 🙏 Acknowledgments

- Medical literature from PubMed
- Embeddings via sentence-transformers (`all-MiniLM-L6-v2`)
- Dense vector search by Pinecone
- LLM synthesis by Anthropic Claude Haiku
- Evaluation & observability via OpenAI (judge) and Langfuse

---

<div align="center">

**Made with ❤️ for evidence-based medical information**

⭐ Star this repo if you find it useful!

</div>
