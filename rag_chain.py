import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from vectorstore import get_or_build_index

load_dotenv()

SYSTEM_PROMPT = """You are a US100/NQ Macro Intelligence Analyst.
Answer the user's question based ONLY on the retrieved historical NQ trading data below.
If the data doesn't contain enough information, say "Not enough data in my knowledge base."

DATASET FACTS (always accurate — use these for any date/range questions):
- First record : 2016-09-14 (Wednesday, BULLISH)
- Last record  : 2026-08-14 (Friday, BEARISH — NQ opened 30241.75, closed 30144.5, range 252 pts)
- Total sessions: 2,517
- Events covered: CPI, Core CPI, NFP, FOMC, Jobless Claims, PCE, Unemployment Rate
If asked about the "most recent", "last", or "latest" data point, the answer is 2026-08-14.

Retrieved Context (sorted by date, most recent first):
{context}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])


def _retrieve_sorted(query: str, vectorstore, k: int = 100):
    """Fetch k docs via FAISS, re-sort by date descending, return top 5."""
    temporal = ["last", "latest", "recent", "newest", "current", "most recent"]
    k = 100 if any(w in query.lower() for w in temporal) else 5
    docs = vectorstore.similarity_search(query, k=k)
    docs.sort(key=lambda d: d.metadata.get("date", ""), reverse=True)
    return docs[:5]


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    """Returns a callable: query (str) → {result, source_documents}"""
    vectorstore = get_or_build_index()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )

    def run(inputs: dict) -> dict:
        query = inputs.get("query", "")
        docs = _retrieve_sorted(query, vectorstore)
        context = _format_docs(docs)
        prompt_value = PROMPT.invoke({"context": context, "question": query})
        response = llm.invoke(prompt_value)

        # Newer Gemini models return content as a list of blocks, not a plain string
        content = response.content
        if isinstance(content, list):
            result_text = "".join(
                part.get("text", "") if isinstance(part, dict) else getattr(part, "text", str(part))
                for part in content
            )
        else:
            result_text = content

        return {
            "result": result_text,
            "source_documents": docs,
        }

    return run


if __name__ == "__main__":
    chain = build_rag_chain()
    query = "How did NQ react to hot CPI prints historically?"
    print(f"\nQuery: {query}\n")
    result = chain({"query": query})
    print("=== ANSWER ===")
    print(result["result"])
    print("\n=== SOURCE DOCUMENTS ===")
    for doc in result["source_documents"]:
        print(f"  - {doc.metadata['date']} | {doc.metadata['overall_bias']} | CPI: {doc.metadata['has_cpi_event']}")
