from dotenv import load_dotenv
import os

#print("🔍 .env check: CO_API_KEY =", os.getenv("CO_API_KEY"))


import uuid
import asyncio
import requests
from io import BytesIO
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from pdf2image import convert_from_bytes
import pytesseract

from app.utils import chunk_text, generate_embeddings, extract_text_from_pdf_stream
from app.llm import get_llm_answer
from app.vector import init_pinecone, upsert_chunks_to_pinecone, delete_vectors, embed_questions

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

print("✅ CO_API_KEY =", os.getenv("CO_API_KEY"))
print("🔑 GROQ_API_KEY =", os.getenv("GROQ_API_KEY"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

security = HTTPBearer()
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "hackrx-secure-token")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-index")
pinecone_index = init_pinecone(PINECONE_INDEX_NAME)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=RunResponse)
async def run_query(request: RunRequest, _: str = Depends(verify_token)):
    session_id = str(uuid.uuid4())
    try:
        response = requests.get(request.documents)
        response.raise_for_status()
        text = extract_text_from_pdf_stream(BytesIO(response.content))
        chunks = chunk_text(text)
        upsert_chunks_to_pinecone(pinecone_index, chunks, session_id)

        question_embeddings = embed_questions(request.questions)
        tasks = [get_llm_answer(q, question_embeddings[i], chunks, pinecone_index) for i, q in enumerate(request.questions)]
        answers = await asyncio.gather(*tasks)

        delete_vectors(pinecone_index, session_id)
        return RunResponse(answers=answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))