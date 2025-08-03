import os
from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
def generate_answer(question: str, context: str) -> str:
    system_prompt = "You are a precise insurance assistant. Only answer based on the provided context. If the answer is missing, reply: 'The answer is not available in the provided document.'"
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content.strip()