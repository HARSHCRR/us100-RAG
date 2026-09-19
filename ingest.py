import json
from langchain_core.documents import Document

def load_documents(filepath):
    with open(filepath) as f:
        records = json.load(f)

    documents = []
    for record in records:
        doc = Document(
            page_content=record["text"],   # ← already clean NL text ✅
            metadata={
                "id": record["id"],
                "date": record["date"],
                "year": record["year"],
                "quarter": record["quarter"],
                "overall_bias": record["price_action"]["overall_bias"],
                "has_cpi_event": record["metadata"]["has_cpi_event"],
                "has_nfp_event": record["metadata"]["has_nfp_event"],
                "has_fed_event": record["metadata"]["has_fed_event"],
                "event_categories": str(record["metadata"]["event_categories"]),
                "surprise_signals": str(record["metadata"]["surprise_signals"]),
                "net_change_pct": record["price_action"]["net_change_pct"],
            }
        )
        documents.append(doc)

    return documents


docs = load_documents("nq_rag_documents.json")
print(f"Total documents: {len(docs)}")
print(docs[0].page_content)
print(docs[0].metadata)
