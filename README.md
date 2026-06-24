---
title: Lecture RAG Assistant
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# Lecture Notes Assistant — RAG-based Q&A System

An AI assistant that answers questions from your own lecture notes and recordings.

## Tech Stack
- **Whisper AI** — transcribes audio/video lectures to text
- **LangChain** — chunking, retrieval chain, prompt management
- **HuggingFace Embeddings** — all-MiniLM-L6-v2 for local embeddings
- **ChromaDB** — local vector database for semantic search
- **Gemini 2.5 Flash** — LLM for grounded answer generation
- **Streamlit** — interactive web UI

## How It Works
1. Upload a PDF or audio file
2. Text is extracted (PyMuPDF for PDF, Whisper for audio)
3. LangChain splits text into 500-word chunks (50-word overlap)
4. HuggingFace model embeds each chunk into a 384-dim vector
5. Vectors stored in ChromaDB on disk
6. At query time: question is embedded, top-k chunks retrieved
7. Chunks + question sent to Gemini via LangChain RetrievalQA
8. Answer is grounded strictly in your notes

## Design Tradeoffs
| Decision | Choice | Why |
|---|---|---|
| Embedding model | all-MiniLM-L6-v2 | Free, local, fast. Trade-off: lower quality than OpenAI |
| Chunk size | 500 words, 50 overlap | Balances context completeness vs retrieval noise |
| Whisper model | base | Fast enough for demos. Large model = better accuracy |
| Vector DB | ChromaDB | Local, no signup, perfect for portfolio. Pinecone for scale |
| LLM | Gemini 1.5 Flash | Free tier, fast. Mistral for fully local option |

## Run Locally
pip install -r requirements.txt
Add GEMINI_API_KEY to .env
streamlit run app.py