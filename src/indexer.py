import os
from pathlib import Path
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- Config ----------
DOCS_DIR = Path("documents")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "locallens"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def load_text_from_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    try:
        if suffix in {".txt", ".md"}:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif suffix in {".docx"}:
            from docx import Document
            doc = Document(str(file_path))
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            print(f"Skipping unsupported file: {file_path}")
            return ""
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def chunk_text(text: str, source: str) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return [
        {
            "id": f"{source}__chunk_{i}",
            "text": chunk,
            "metadata": {"source": source, "chunk_index": i}
        }
        for i, chunk in enumerate(chunks)
    ]

def build_or_update_index(force_reindex: bool = False):
    print("Loading embedding model (first time will download ~80 MB)...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    supported = {".txt", ".md", ".pdf", ".docx"}
    files = [f for f in DOCS_DIR.rglob("*") if f.suffix.lower() in supported and f.is_file()]

    if not files:
        print("No documents found in ./documents. Add some files and try again.")
        return

    print(f"Found {len(files)} files. Processing...")

    all_ids, all_texts, all_metadatas = [], [], []

    for file_path in files:
        rel_source = str(file_path.relative_to(DOCS_DIR))
        text = load_text_from_file(file_path)
        if not text.strip():
            continue

        chunks = chunk_text(text, rel_source)
        for c in chunks:
            all_ids.append(c["id"])
            all_texts.append(c["text"])
            all_metadatas.append(c["metadata"])

    if not all_ids:
        print("No usable text extracted.")
        return

    print(f"Creating embeddings for {len(all_ids)} chunks...")
    embeddings = embedder.encode(all_texts, show_progress_bar=True).tolist()

    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_texts,
        metadatas=all_metadatas
    )

    print(f"Index ready. Total chunks in collection: {collection.count()}")

if __name__ == "__main__":
    build_or_update_index()