import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=768, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENVIRONMENT")))
pinecone_index = pc.Index(index_name)
def upsert_embeddings(embeddings, texts, session_id):
    vectors = [{"id": f"{session_id}-{i}", "values": emb, "metadata": {"text": txt, "session_id": session_id}}
               for i, (emb, txt) in enumerate(zip(embeddings, texts))]
    pinecone_index.upsert(vectors=vectors)
def semantic_search(embedding, session_id, top_k=5):
    results = pinecone_index.query(vector=embedding, top_k=top_k, include_metadata=True, filter={"session_id": session_id})
    return [match.metadata['text'] for match in results.matches]