# HackRx - LLM Powered Insurance Doc QnA API

## 🔥 Features
- Fast answers using Groq + Cohere
- Semantic + keyword hybrid search with Pinecone
- PDF OCR support (scanned docs)
- FastAPI backend ready for HackRx

## ✅ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Environment
Copy `.env.example` → `.env` and fill in your keys.

### 3. Run the App
```bash
uvicorn main:app --reload
```

### 4. Test Endpoint
POST `/hackrx/run`
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": ["What is the waiting period?", "Is maternity covered?"]
}
```
Header: `Authorization: Bearer hackrx-secure-token`

## 🔐 .env Format
```
COHERE_API_KEY=...
GROQ_API_KEY=...
PINECONE_API_KEY=...
BEARER_TOKEN=hackrx-secure-token
```

## 💡 Tips
- Uses `mixtral` on Groq for super-fast generation
- Use `cohere-embed-v3.0` for accurate embeddings
- Index deleted per session (auto cleanup)

---
Built for HackRx 6.0 🚀