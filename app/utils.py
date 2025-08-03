from typing import List, Tuple
from pdf2image import convert_from_bytes
import pytesseract
from sentence_transformers import util
import torch

# Set the path to Tesseract-OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set the Poppler path (used for PDF to image conversion)
POPPLER_PATH = r"C:\Users\HP\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

def extract_text_from_pdf_stream(file_stream):
    """
    Extracts text from a PDF file stream using OCR (Tesseract).
    """
    try:
        images = convert_from_bytes(file_stream.read(), poppler_path=POPPLER_PATH)
        text = "".join(pytesseract.image_to_string(img) for img in images)
        return text
    except Exception as e:
        raise RuntimeError(f"Error during OCR extraction: {str(e)}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks for embedding or processing.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

def generate_embeddings(chunks: List[str], cohere_client) -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks using Cohere embed model.
    """
    response = cohere_client.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return response.embeddings

def hybrid_query(query: str, chunks: List[str], chunk_embeddings: List[List[float]], query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Returns top-k matching chunks for a given query using cosine similarity.
    """
    query_tensor = torch.tensor([query_embedding])
    chunk_tensors = torch.tensor(chunk_embeddings)

    cosine_scores = util.cos_sim(query_tensor, chunk_tensors)[0]
    top_results = torch.topk(cosine_scores, k=top_k)

    results = []
    for score_idx in top_results.indices:
        chunk = chunks[score_idx]
        score = cosine_scores[score_idx].item()
        results.append((chunk, score))

    return results
