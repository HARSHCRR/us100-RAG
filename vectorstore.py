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

if __name__ == "__main__":
    if os.path.exists(FAISS_INDEX_PATH):
        print("Index already exists, loading...")
        vs = load_index()
    else:
        vs = build_index()

    # Test a similarity search
    results = vs.similarity_search("NQ reaction to hot CPI print", k=3)
    print(f"\nTop 3 results for test query:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Date: {doc.metadata['date']}")
        print(f"Bias: {doc.metadata['overall_bias']}")
        print(f"CPI Event: {doc.metadata['has_cpi_event']}")
        print(f"Text preview: {doc.page_content[:150]}...")
