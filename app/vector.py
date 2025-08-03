import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from app.utils import chunk_text
from dotenv import load_dotenv

# Load .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

# Load API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
print(f"🔑 PINECONE_API_KEY loaded = {PINECONE_API_KEY}")

# Initialize sentence transformer model (768-dim)
embedding_model = SentenceTransformer("paraphrase-mpnet-base-v2")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def init_pinecone(index_name):
    print(f"🔍 Checking Pinecone index: {index_name}")
    if index_name not in pc.list_indexes().names():
        print(f"📦 Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    # Wait for index to become ready
    while True:
        description = pc.describe_index(index_name)
        if description.status['ready']:
            print(f"✅ Pinecone index '{index_name}' is ready.")
            break

    # ✅ FIX: Use Pinecone().Index(name=...) to get the index object
    return pc.Index(name=index_name)


def embed_questions(questions):
    return embedding_model.encode(questions).tolist()

def embed_chunks(chunks):
    return embedding_model.encode(chunks).tolist()

def upsert_chunks_to_pinecone(index, chunks, doc_id):
    embeddings = embed_chunks(chunks)

    # 🔍 Add debug logs
    print("🔍 Embedding Sample:", embeddings[0])
    print("✅ Embedding Type:", type(embeddings[0]), " Length:", len(embeddings[0]))

    vectors = [
        {
            "id": f"{doc_id}_chunk_{i}",
            "values": emb,
            "metadata": {"text": txt, "doc_id": doc_id}
        }
        for i, (txt, emb) in enumerate(zip(chunks, embeddings))
    ]
    index.upsert(vectors=vectors)

    print(f"📤 Upserted {len(vectors)} vectors to Pinecone.")

def query_pinecone(index, embedding, top_k):
    return index.query(vector=list(embedding), top_k=top_k, include_metadata=True).matches

def hybrid_query(index, chunks, question, q_embed, top_k):
    pinecone_results = query_pinecone(index, q_embed, top_k)
    pinecone_texts = [match.metadata['text'] for match in pinecone_results]

    keywords = set(question.lower().split()[:10])  # Limit keyword noise
    keyword_matches = [c for c in chunks if any(k in c.lower() for k in keywords)]

    return list(dict.fromkeys(pinecone_texts + keyword_matches))  # Remove duplicates

def delete_vectors(index, doc_id):
    index.delete(filter={"doc_id": doc_id})
    print(f"🗑️ Deleted vectors with doc_id: {doc_id}")
