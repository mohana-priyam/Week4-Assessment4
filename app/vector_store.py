
import sqlite3
import re
import math
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "rag_index.db"


def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def build_index(chunks):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS documents")

        conn.execute("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                page INTEGER NOT NULL
            )
        """)

        for chunk in chunks:
            conn.execute(
                "INSERT INTO documents (content, page) VALUES (?, ?)",
                (chunk["text"], chunk["page"])
            )

    return len(chunks)


def retrieve(question, top_k=3):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT content, page FROM documents"
        ).fetchall()

    if not rows:
        return []

    query_words = tokenize(question)
    documents = [tokenize(row[0]) for row in rows]

    document_frequency = Counter()

    for words in documents:
        document_frequency.update(set(words))

    total_docs = len(documents)
    vocabulary_size = len(document_frequency)

    def vectorize(words):
        counts = Counter(words)
        total_words = max(len(words), 1)

        vector = {}

        for word, count in counts.items():
            tf = count / total_words
            idf = math.log(
                (total_docs + 1) /
                (document_frequency[word] + 1)
            ) + 1

            vector[word] = tf * idf

        return vector

    query_vector = vectorize(query_words)
    results = []

    for row, words in zip(rows, documents):
        document_vector = vectorize(words)

        dot_product = sum(
            query_vector.get(word, 0) * document_vector.get(word, 0)
            for word in query_vector
        )

        query_norm = math.sqrt(
            sum(value ** 2 for value in query_vector.values())
        )

        document_norm = math.sqrt(
            sum(value ** 2 for value in document_vector.values())
        )

        denominator = query_norm * document_norm

        score = dot_product / denominator if denominator else 0

        results.append({
            "text": row[0],
            "page": row[1],
            "score": score
        })

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]