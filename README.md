# 🧠 US100 Macro Intelligence RAG Assistant

A lightweight **Retrieval-Augmented Generation (RAG)** system built on 10 years of NQ/US100 (NASDAQ-100) historical trading data.

Ask natural language questions about macroeconomic events (CPI, NFP, FOMC, Jobless Claims) and get grounded answers backed by real historical price data.

---

## 🏗️ Architecture

```
JSON Dataset → Preprocessing → FastEmbed Embeddings → FAISS Index → Retrieval → Gemini LLM → Answer
```

| Module | File | Purpose |
|--------|------|---------|
| Preprocessing | `ingest.py` | Load JSON → LangChain Documents |
| Embeddings | `embeddings.py` | FastEmbed `BAAI/bge-small-en-v1.5` (384-dim) |
| Vector Store | `vectorstore.py` | Build & save FAISS in-memory index |
| RAG Chain | `rag_chain.py` | DateSortedRetriever + Gemini LLM |

---

## 📦 Dataset

- **Instrument:** NQ (NASDAQ-100 E-mini Futures)
- **Period:** September 2016 – September 2026
- **Records:** 2,517 trading day documents
- **Events:** CPI, Core CPI, PCE, NFP, Unemployment Rate, Initial Jobless Claims, FOMC

---

## 🚀 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key
Create a `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
Get a free key at [aistudio.google.com](https://aistudio.google.com)

### 3. Build the FAISS index (first time only, ~5-10 min)
```bash
python vectorstore.py
```

### 4. Run the RAG chain
```bash
python rag_chain.py
```

---

## 💬 Example Queries

- *"How did NQ react to hot CPI prints historically?"*
- *"What happened on FOMC days in 2022?"*
- *"Show NFP days where NQ was bearish"*
- *"What's the latest CPI data in the dataset?"*
- *"Average NQ range on jobless claims beat days?"*

---

## 🛠️ Tech Stack

- **LangChain** — RAG orchestration
- **FastEmbed** (`BAAI/bge-small-en-v1.5`) — Lightweight CPU embeddings
- **FAISS** — In-memory vector similarity search
- **Google Gemini** (`gemini-3.6-flash`) — LLM for grounded answers
- **Python** 3.9+
