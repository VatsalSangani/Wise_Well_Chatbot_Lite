# 🏥 WiseWell Medical Chatbot - Lite

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/react-18+-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/accuracy-87.5%25-brightgreen.svg)]()

> An advanced medical information retrieval system combining hybrid search (BM25 + FAISS) with AWS Bedrock Claude Sonnet synthesis, featuring comprehensive safety guardrails and natural language responses.

---
[![Live Demo](https://img.shields.io/badge/AWS%20EC2-Live%20Demo-orange)](http://13.134.107.196:8090/wiswell)

> **Try it live** → [http://13.134.107.196:8090/wiswell](http://13.134.107.196:8090/wiswell)
![Demo UI using React](https://github.com/VatsalSangani/Wise_Well_Chatbot_Lite/blob/main/WiseWell%20Screenshot.png)

---

## ✨ Features

### 🎯 **Core Capabilities**
- **87.5% Accuracy** on medical Q&A evaluation
- **8-Stage Safety Guardrails** with LangGraph orchestration
- **Hybrid Retrieval** - BM25 sparse + FAISS dense search
- **AWS Bedrock Claude Sonnet** for natural language synthesis (IAM role auth)
- **Full Citations** - Every claim backed by PubMed IDs

### 🛡️ **Safety First**
- ✅ **Refuses** personal medical advice
- ✅ **Abstains** when evidence insufficient
- ✅ **Answers** only with strong evidence support
- ✅ **Transparent** decision-making (ANSWER/ABSTAIN/REFUSE)

---
## 🏗️ Architecture

```
User Query
    ↓
8-Stage Guardrail Pipeline (LangGraph)
    ├─ Safety Intent → Block harmful
    ├─ Query Specificity → Clear questions
    ├─ Topic Consistency → Filter off-topic
    ├─ Hybrid Retrieval → BM25 + FAISS
    ├─ Overlap Veto → Diverse sources
    ├─ Mechanism Gate → Evidence quality
    ├─ Evidence Gate → Sufficient support
    └─ Citation Check → Validate PMIDs
    ↓
Decision: ANSWER / ABSTAIN / REFUSE
    ↓
AWS Bedrock Claude Sonnet Synthesis (if ANSWER)
    ↓
Natural Language Response + Citations
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- AWS IAM role with `bedrock:InvokeModel` permission (no API key required)
- Knowledge base indexes at `WISEWELL_INDEXES_ROOT`

### Installation

```bash
# 1. Clone repository
git clone https://github.com/VatsalSangani/Wise_Well_Chatbot_Lite.git
cd Wise_Well_Chatbot_Lite

# 2. Backend setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Frontend setup  (frontend/.env already set to EC2 URL)
cd frontend && npm install

# 4. Run application
# Terminal 1 — backend (port 8502):
python backend/main.py

# Terminal 2 — frontend dev server:
cd frontend && npm run dev
```

**Production (EC2):** `http://13.134.107.196:8504`  
**Local dev:** `http://localhost:5173`

---

## 🧪 Testing

```bash
# Run test suite
python test_backend.py

# Test API (local)
curl -X POST http://localhost:8502/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is CRP?"}'

# Health check
curl http://localhost:8502/health
```

---

## 🎯 Example Queries

### ✅ ANSWER - Strong evidence found

**Query:** "What do IL-6 inhibitors do in rheumatoid arthritis?"

**Response:**
```
IL-6 inhibitors are medications that block interleukin-6, a key 
inflammatory molecule that drives joint damage in rheumatoid 
arthritis [PMID: 39050325]. These drugs work by preventing IL-6 
from binding to its receptors, reducing inflammation and pain...

[8 sources from PubMed]
```

### 🟡 ABSTAIN - Needs clarification

**Query:** "Why is this number high?"

**Response:**
```
I need more details to answer accurately. Could you specify:
• What measurement or biomarker?
• In what clinical context?
```

### 🔴 REFUSE - Personal medical advice

**Query:** "Should I take aspirin?"

**Response:**
```
⚠️ I cannot provide personal medical advice.
Please consult with a qualified healthcare professional.
```

---

## 📁 Project Structure

```
Wise_Well_Chatbot_Lite/
├── config.py                        ← all constants (port, model IDs, paths, defaults)
├── requirements.txt                 ← CPU-only torch, boto3, no openai
├── test_backend.py
├── backend/
│   ├── main.py                      ← slim entry point (≤50 lines, port 8502)
│   ├── schemas.py                   ← Pydantic request/response models
│   ├── deps.py                      ← HybridRetriever singleton (lru_cache)
│   └── routes/
│       ├── health.py                ← GET /, /health, /health/ready, /health/live
│       ├── query.py                 ← POST /query (guardrails + Bedrock synthesis)
│       └── admin.py                 ← GET /admin/config, /admin/stats
├── retrieval/
│   ├── __init__.py
│   └── hybrid_retriever.py          ← BM25 + FAISS with RRF fusion (single copy)
├── orchestration/
│   ├── service.py                   ← run_wisewell_query() wrapper
│   ├── llm_syntheses.py             ← AWS Bedrock Claude Sonnet (IAM role auth)
│   ├── graph.py                     ← LangGraph StateGraph definition
│   ├── state.py                     ← QAState dataclass
│   ├── nodes.py                     ← LangGraph node implementations
│   ├── bootstrap.py                 ← sys.path initialisation
│   └── utils.py                     ← timing decorators
├── guardrails/                      ← 8-stage safety pipeline
│   ├── input_validation.py
│   ├── safety_intent.py
│   ├── query_specificity.py
│   ├── topic_consistency.py
│   ├── evidence_gate.py
│   ├── composer_extractive.py
│   ├── citation_verifier.py
│   ├── config_loader.py
│   └── validate_config.py
├── config/
│   └── guardrails.yaml              ← safety thresholds and retrieval config
├── scripts/                         ← data ingestion and evaluation utilities
│   ├── qa_check.py
│   ├── eval_runner.py
│   ├── eval_quick_test.py
│   └── build_hybrid_indexes.py
├── frontend/
│   ├── .env                         ← VITE_API_URL=http://13.134.107.196:8504
│   └── src/
│       ├── App.tsx                  ← health check → EC2 :8502/health
│       ├── components/
│       └── services/api.ts
└── kb/indexes/                      ← FAISS + BM25 indexes (not committed)
    └── year=2023/ year=2024/
```

---

## ⚙️ Configuration

### Backend environment variables
```env
# AWS — credentials come from the EC2 IAM role (no key needed)
AWS_DEFAULT_REGION=eu-west-1

# Knowledge base
WISEWELL_INDEXES_ROOT=/home/ubuntu/projects/wisewell/kb/indexes
WISEWELL_YEARS=2023,2024

# API server
WISEWELL_PORT=8502
WISEWELL_HOST=0.0.0.0

# Feature flags
ENABLE_LLM_SYNTHESIS=true
ENABLE_ADMIN_ENDPOINTS=false
DEBUG=false
```

### Frontend (`frontend/.env`)
```env
VITE_API_URL=http://13.134.107.196:8504
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 87.5% |
| Response Time | 1.5s avg |
| Cost per Query | $0.0003 |
| Throughput | 15-20 q/s |

---

## 🔒 Medical Disclaimer

> ⚠️ **For informational purposes only.**  
> Always consult with a qualified healthcare professional for medical advice, diagnosis, or treatment.  
> This tool does not provide personal medical recommendations.

---

## 🛠️ Tech Stack

**Backend:** Python 3.9+, FastAPI, LangGraph, FAISS (CPU), sentence-transformers  
**Frontend:** React 18+, TypeScript, Vite, Tailwind CSS  
**LLM:** AWS Bedrock — `eu.anthropic.claude-3-sonnet-20240229-v1:0` (IAM role auth)  
**Deployment:** AWS EC2 · port 8502  
**Knowledge base:** PubMed 2023–2024 (hybrid BM25 + FAISS indexes)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 📞 Contact

**GitHub:** [@VatsalSangani](https://github.com/VatsalSangani)  
**Repository:** [Wise_Well_Chatbot_Lite](https://github.com/VatsalSangani/Wise_Well_Chatbot_Lite)

---

## 🙏 Acknowledgments

- Medical literature from PubMed
- Embeddings from sentence-transformers
- Vector search by FAISS
- LLM synthesis by AWS Bedrock Claude Sonnet
- Orchestration by LangGraph

---

<div align="center">

**Made with ❤️ for evidence-based medical information**

⭐ Star this repo if you find it useful!

</div>
