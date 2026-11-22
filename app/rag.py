
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent
MANUAL_PATH = BASE_DIR / "data" / "product_manual.txt"


def load_pdf():
    if not MANUAL_PATH.exists():
        raise FileNotFoundError(
            f"Product manual not found: {MANUAL_PATH}"
        )

    text = MANUAL_PATH.read_text(encoding="utf-8")

    return [{
        "text": text,
        "page": 1
    }]


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    chunks = []

    for document in documents:
        page_chunks = splitter.split_text(document["text"])

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": document["page"]
            })

    return chunks


def load_and_split_pdf():
    documents = load_pdf()
    return split_documents(documents)