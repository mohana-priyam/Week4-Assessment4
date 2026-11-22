import re
import math

from fastapi import FastAPI
from pydantic import BaseModel

from app.pipeline import search_manual


app = FastAPI(
    title="RAG Assistant API",
    description="Product manual question-answering API",
    version="1.0"
)


class AskRequest(BaseModel):
    question: str


STOP_WORDS = {
    "what", "is", "the", "a", "an", "how", "do", "does",
    "should", "i", "we", "you", "can", "could", "please",
    "tell", "me", "about", "when", "where", "why", "who",
    "to", "of", "for", "in", "on", "and", "or", "my"
}


def get_answer(question, chunks):
    question_words = {
        word.lower()
        for word in re.findall(r"\b[a-zA-Z0-9]+\b", question)
        if word.lower() not in STOP_WORDS
    }

    if not question_words:
        return "I don't know.", []

    candidates = []

    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk["text"])

        for sentence in sentences:
            sentence_words = {
                word.lower()
                for word in re.findall(r"\b[a-zA-Z0-9]+\b", sentence)
            }

            matches = question_words.intersection(sentence_words)

            if matches:
                coverage = len(matches) / len(question_words)
                candidates.append({
                    "sentence": sentence.strip(),
                    "page": chunk["page"],
                    "matches": len(matches),
                    "coverage": coverage
                })

    if not candidates:
        return "I don't know.", []

    candidates.sort(
        key=lambda item: (item["coverage"], item["matches"]),
        reverse=True
    )

    best = candidates[0]
    minimum_matches = math.ceil(len(question_words) * 0.5)

    if best["matches"] < minimum_matches:
        return "I don't know.", []

    return best["sentence"], [{"page": best["page"]}]


@app.get("/")
def home():
    return {"message": "RAG Assistant API is running"}


@app.post("/ask")
def ask_question(request: AskRequest):
    chunks = search_manual(request.question)

    answer, sources = get_answer(request.question, chunks)

    return {
        "answer": answer,
        "sources": sources
    }
