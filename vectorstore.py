import os
from langchain_community.vectorstores import FAISS
from ingest import load_documents
from embeddings import get_embedding_model

FAISS_INDEX_PATH = "faiss_index"

def build_index():
    print("Loading documents...")
    docs = load_documents("nq_rag_documents.json")
    print(f"Loaded {len(docs)} documents")

    print("Loading embedding model...")
    embeddings = get_embedding_model()

    print("Building FAISS index (this may take ~1-2 min)...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"✅ Index saved to '{FAISS_INDEX_PATH}/'")
    return vectorstore

def load_index():
    embeddings = get_embedding_model()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Index loaded from disk")
    return vectorstore

def get_or_build_index():
    """Load index if it exists, otherwise build it. Safe for Streamlit Cloud."""
    if os.path.exists(FAISS_INDEX_PATH):
        return load_index()
    else:
        print("No index found — building now (first-run, takes ~5 min)...")
        return build_index()

