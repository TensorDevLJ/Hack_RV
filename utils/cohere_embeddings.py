import os
import cohere
from dotenv import load_dotenv
load_dotenv()
co = cohere.Client(os.getenv("CO_API_KEY"))
def get_embeddings(texts: list) -> list:
    resp = co.embed(texts=texts, model="embed-english-v3.0")
    return resp.embeddings