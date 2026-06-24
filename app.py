import os
import tempfile
import streamlit as st
import hashlib
from dotenv import load_dotenv

import whisper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import fitz  # PyMuPDF

load_dotenv()

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Lecture Notes Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Lecture Notes Assistant")
st.caption("Upload lecture PDFs or audio recordings — then ask anything from them")

# ── Load models (cached so they don't reload on every interaction) ─
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource
def load_vectorstore():
    return Chroma(
        collection_name="lecture_notes",
        embedding_function=load_embedding_model(),
        persist_directory="./chroma_db"
    )

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2   # low = more factual, less creative
    )

embedding_model = load_embedding_model()
whisper_model   = load_whisper()
vectorstore     = load_vectorstore()
llm             = load_llm()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)


# ── Helper: get list of unique source filenames in the DB ───────
def get_ingested_filenames():
    try:
        data = vectorstore._collection.get(include=["metadatas"])
        sources = {m["source"] for m in data["metadatas"] if "source" in m}
        return sorted(sources)
    except Exception:
        return []


# ── Helper: ingest any uploaded file ────────────────────────────
def ingest_uploaded_file(uploaded_file):
    filename = uploaded_file.name
    ext = filename.split(".")[-1].lower()

    # Save to a temp file so libraries can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Extract text
    if ext == "pdf":
        doc = fitz.open(tmp_path)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
    elif ext in ["mp3", "mp4", "wav", "m4a", "webm"]:
        with st.spinner(f"Transcribing {filename} with Whisper..."):
            result = whisper_model.transcribe(tmp_path)
            text = result["text"]
    else:
        st.error(f"Unsupported file type: {ext}")
        return 0

    os.unlink(tmp_path)  # clean up temp file

    # Chunk and store
    chunks = splitter.split_text(text)
    metadatas = [{"source": filename, "chunk_index": i, "type": ext}
                 for i in range(len(chunks))]
    ids = [
        hashlib.md5(f"{filename}-{i}-{chunk}".encode()).hexdigest()
        for i, chunk in enumerate(chunks)
    ]
    vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)


# ── Sidebar: upload files ────────────────────────────────────────
with st.sidebar:
    st.header("Upload Content")
    uploaded_files = st.file_uploader(
        "PDF or Audio files",
        type=["pdf", "mp3", "mp4", "wav", "m4a", "webm"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Ingest Files", type="primary"):
            for f in uploaded_files:
                with st.spinner(f"Processing {f.name}..."):
                    n = ingest_uploaded_file(f)
                    st.success(f"{f.name} → {n} chunks stored")

    st.divider()
    st.markdown("**How it works**")
    st.markdown("""
1. Upload PDFs or recordings
2. They get chunked + embedded
3. Stored in ChromaDB locally
4. Ask questions below
5. Top matching chunks sent to Gemini
6. Answer grounded in your content
    """)

    # Show what's in the DB
    st.divider()
    count = vectorstore._collection.count()
    st.metric("Chunks in database", count)

    st.divider()
    st.markdown("**Files in database:**")
    available_files = get_ingested_filenames()
    if available_files:
        for f in available_files:
            st.text(f"• {f}")
    else:
        st.text("No files yet")


# ── Main: Q&A Interface ──────────────────────────────────────────
st.subheader("Ask a Question")

question = st.text_input(
    "Your question:",
    placeholder="What is the main idea of chapter 3?"
)

col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.selectbox("Chunks to retrieve", [2, 3, 5], index=1)
with col2:
    file_filter = st.selectbox(
        "Search in:",
        ["All files"] + get_ingested_filenames()
    )

if st.button("Ask", type="primary") and question:
    if vectorstore._collection.count() == 0:
        st.warning("No content in database yet. Upload some files first.")
    else:
        with st.spinner("Searching your notes and generating answer..."):

            search_kwargs = {"k": top_k}
            if file_filter != "All files":
                search_kwargs["filter"] = {"source": file_filter}

            # Build LangChain RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",       # stuff = put all chunks in one prompt
                retriever=vectorstore.as_retriever(
                    search_kwargs=search_kwargs
                ),
                return_source_documents=True
            )

            result = qa_chain.invoke({"query": question})

        # Display answer
        st.markdown("### Answer")
        st.write(result["result"])

        # Display sources
        with st.expander("📄 Retrieved chunks (what Gemini used)"):
            for i, doc in enumerate(result["source_documents"]):
                src = doc.metadata.get("source", "unknown")
                idx = doc.metadata.get("chunk_index", "?")
                st.markdown(f"**Chunk {i+1}** — `{src}` (chunk #{idx})")
                st.text(doc.page_content[:400] + "...")
                st.divider()