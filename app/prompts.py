
RAG_PROMPT = """
You are a product manual assistant.
Answer using ONLY the supplied context.
Never invent information.
If the answer is unavailable, respond exactly: I don't know.

Context:
{context}

Question:
{question}

Answer:
"""

CLASSIFICATION_PROMPT = """
Classify into exactly one category:
product_information
troubleshooting
usage_instructions
unanswerable

Choose unanswerable when the context does not contain the answer.

Question: {question}
Context: {context}
Category:
"""

CHAT_PROMPT = """
Answer product questions using manual context only.
Use tools for customer or order information.
Never invent information.
If unavailable, say: I don't know.

Context: {context}
Question: {question}
"""