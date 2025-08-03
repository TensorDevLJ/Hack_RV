import os
from app.utils import hybrid_query
from groq import Groq

async def get_llm_answer(question, query_embedding, all_chunks, chunk_embeddings):
    # ✅ Initialize Groq client inside the function (after .env is loaded in main)
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Use hybrid search to get top relevant context
    top_chunks = hybrid_query(
        question,
        chunks=all_chunks,
        chunk_embeddings=chunk_embeddings,
        query_embedding=query_embedding,
        top_k=4
    )

    # Combine context chunks
    context_str = "\n---\n".join([chunk for chunk, _ in top_chunks])

    # Build prompt
    prompt = f"""Use ONLY the context below to answer the question.

Context:
---
{context_str}
---
Question: {question}
Answer:"""

    # Call Groq LLM (e.g., LLaMA3 model)
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()
