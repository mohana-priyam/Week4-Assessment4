
from app.rag import load_and_split_pdf
from app.vector_store import build_index, retrieve


def index_manual():
    chunks = load_and_split_pdf()
    total = build_index(chunks)

    return {
        "message": "Manual indexed successfully",
        "total_chunks": total
    }


def search_manual(question):
    return retrieve(question, top_k=3)

