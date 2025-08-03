import os
import requests
import cohere
from io import BytesIO
from app.utils import extract_text_from_pdf_stream, chunk_text, generate_embeddings, hybrid_query
from app.llm_handler import get_llm_answer
from app.models import QueryRequest, QueryResponse
from pinecone import Pinecone, ServerlessSpec

cohere_client = cohere.Client(os.getenv("CO_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

class QueryEngine:
    def __init__(self):
        self.document_cache = {}

    async def process_query_request(self, request: QueryRequest) -> QueryResponse:
        doc_url = request.documents

        # Caching check
        if doc_url in self.document_cache:
            text_chunks, chunk_embeddings = self.document_cache[doc_url]
        else:
            # 1. Download document
            response = requests.get(doc_url)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch document from URL: {doc_url}")

            # 2. Extract & chunk text
            text = extract_text_from_pdf_stream(BytesIO(response.content))
            text_chunks = chunk_text(text)

            # 3. Generate embeddings using Cohere
            chunk_embeddings = generate_embeddings(text_chunks, cohere_client)

            # 4. Store in cache
            self.document_cache[doc_url] = (text_chunks, chunk_embeddings)

        # 5. Process each question
        answers = []
        for question in request.questions:
            # Generate query embedding
            query_embedding = cohere_client.embed(
                texts=[question],
                model="embed-english-v3.0",
                input_type="search_query"
            ).embeddings[0]

            # 6. Call LLM
            answer = await get_llm_answer(
                question,
                query_embedding,
                text_chunks,
                chunk_embeddings
            )
            answers.append(answer)

        return QueryResponse(answers=answers)

    async def health_check(self):
        return {"status": "healthy", "llm": "Groq", "embedding": "Cohere"}
