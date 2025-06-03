# 🤖 Wise Well Chatbot Lite

**Wise Well Chatbot Lite** is a lightweight version of an AI-powered medical assistant built using `FastAPI`, `Sentence-BERT` for semantic retrieval, and `DistilGPT2` fine-tuned on medical data for contextual generation.

This project demonstrates full-stack capabilities (frontend + backend integration) while keeping compute usage minimal for public deployment.

---

## 🚀 Demo

**Lite Chatbot UI:**  
🌐 [https://wisewellchatbot.vatsalsangani.in](https://wisewellchatbot.vatsalsangani.in)

> 📝 _This is a general-purpose model with an improved RAG pipeline but not trained specifically on medical corpora. It is built to showcase backend and frontend engineering skills._  
> _The main BioBART v2 version was too resource-intensive to deploy on Hugging Face Spaces._

---

## 🧠 Features

- 🔍 **Semantic Retrieval** using Sentence-BERT
- 📄 **Medical Corpus Embedding** with FAISS
- 💬 **DistilGPT2 LoRA model** for context-aware generation
- 🧵 **FastAPI backend** with RAG logic
- 🌐 **React-based frontend** with consent control and dynamic loading
- 🐳 **Dockerized** and Hugging Face Spaces-compatible

---

## 🗂️ Project Structure

```
Wise_Well_Chatbot_Lite/
├── backend/                    # FastAPI app and logic
├── corpus/                    # Medical facts dataset (CSV)
├── distilgpt2-medical-lora/   # LoRA fine-tuned GPT2 model
├── Wise_Well_chatbot_frontend/  # React frontend (see below)
├── Dockerfile
├── requirements.txt
```

---

## 💡 Setup Instructions

### 🔧 Backend (FastAPI)

1. Clone the repo:
```bash
git clone https://github.com/VatsalSangani/Wise_Well_Chatbot_Lite.git
cd Wise_Well_Chatbot_Lite
```

2. Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the backend:
```bash
uvicorn backend.wise_well_app:app --reload
```

---

### 🔧 Backend LLM (DistillGPT2)

⚠️ Due to GitHub LFS limits, the model is hosted externally.

**Download frontend folder**:  
📥 [Download from Hugging Face Libraries](https://huggingface.co/brendvat/Wise_Well_Chatbot_Lite)

### 🌐 Frontend (React)

1. Extract into the project root:
```
Wise_Well_Chatbot_Lite/
└── Wise_Well_chatbot_frontend/
```

2. Install dependencies and run:
```bash
cd Wise_Well_chatbot_frontend
npm install
npm run dev
```

---

## 🐳 Docker (Optional)

Build and run everything via Docker:

```bash
docker build -t wisewell-lite .
docker run -p 7860:7860 wisewell-lite
```

---

## 📦 Hugging Face Space

A live deployment is available on Hugging Face Spaces:  
🔗 [https://huggingface.co/spaces/brendvat/Wise-well-chatbot-lite](https://huggingface.co/spaces/brendvat/Wise-well-chatbot-lite)

---

## 🧠 Main Chatbot (Heavy Model)

The original version uses **BioBART v2** (Transformer model for biomedical domain), but it was too resource-heavy for public hosting. The Lite version uses a distilled general model for demo purposes This model would sometimes give some irrelevant responses .

---

## ⚠️ NOTE

This model gives irrelevant results to users queries the sole purpose of this project is to Show Full Stack Capabilities.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

**Vatsal Sangani**  
🔗 [GitHub](https://github.com/VatsalSangani) | [Portfolio](https://vatsalsangani.in)
