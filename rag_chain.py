import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pydantic import Field
from vectorstore import load_index

load_dotenv()

PROMPT_TEMPLATE = """You are a US100/NQ Macro Intelligence Analyst.
Answer the user's question based ONLY on the retrieved historical NQ trading data below.
If the data doesn't contain enough information, say "Not enough data in my knowledge base."

IMPORTANT: Your knowledge base covers NQ trading data from September 2016 to September 2026.
If asked about "latest", "last", or "most recent" data, refer to the most recent date in the retrieved context.

Retrieved Context (sorted by date, most recent first):
{context}

Question: {question}

Answer (be specific, cite dates and price reactions):"""


class DateSortedRetriever(BaseRetriever):
    """Retrieves docs by semantic similarity, then re-sorts by date (most recent first)."""
    base_retriever: object = Field(description="Base FAISS retriever")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.get_relevant_documents(query)
        # Sort by date descending → most recent first
        docs.sort(key=lambda d: d.metadata.get("date", ""), reverse=True)
        return docs[:5]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


def build_rag_chain():
    vectorstore = load_index()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
    retriever = DateSortedRetriever(base_retriever=base_retriever)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain


if __name__ == "__main__":
    print("Building RAG chain...")
    chain = build_rag_chain()

    query = "what's the last cpi data?"
    print(f"\nQuery: {query}\n")

    result = chain.invoke({"query": query})
    print("=== ANSWER ===")
    print(result["result"])
    print("\n=== SOURCE DOCUMENTS (sorted by date, most recent first) ===")
    for doc in result["source_documents"]:
        print(f"  - {doc.metadata['date']} | {doc.metadata['overall_bias']} | CPI: {doc.metadata['has_cpi_event']}")
