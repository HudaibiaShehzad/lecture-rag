import os
import fitz                          # PyMuPDF — reads PDFs
import whisper                       # Whisper — transcribes audio
import hashlib
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ── Models ──────────────────────────────────────────────────────
whisper_model = whisper.load_model("base")   # base = fast, good quality
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── Text splitter (LangChain) ────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # ~500 words per chunk
    chunk_overlap=50,      # 50-word overlap so context isn't cut mid-sentence
    separators=["\n\n", "\n", ".", " "]
)


# ── PDF extraction ───────────────────────────────────────────────
def extract_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()
    print(f"  Extracted {len(full_text)} characters from PDF")
    return full_text


# ── Audio/Video transcription ────────────────────────────────────
def extract_from_audio(audio_path):
    print(f"  Transcribing audio (this takes a minute)...")
    result = whisper_model.transcribe(audio_path)
    print(f"  Transcribed {len(result['text'])} characters")
    return result["text"]


# ── Main ingest function ─────────────────────────────────────────
def ingest_file(file_path):
    filename = os.path.basename(file_path)
    ext = filename.split(".")[-1].lower()

    print(f"\nIngesting: {filename}")

    # Step 1: Extract text
    if ext == "pdf":
        text = extract_from_pdf(file_path)
    elif ext in ["mp3", "mp4", "wav", "m4a", "webm"]:
        text = extract_from_audio(file_path)
    else:
        print(f"  Skipping unsupported file type: {ext}")
        return

    # Step 2: Chunk with LangChain splitter
    chunks = splitter.split_text(text)
    print(f"  Split into {len(chunks)} chunks")

    # Step 3: Add metadata to each chunk
    metadatas = [{"source": filename, "chunk_index": i, "type": ext}
                 for i in range(len(chunks))]

    # Step 3b: Create a unique, deterministic ID per chunk (prevents duplicates on re-ingest)
    ids = [
        hashlib.md5(f"{filename}-{i}-{chunk}".encode()).hexdigest()
        for i, chunk in enumerate(chunks)
    ]

    # Step 4: Store in ChromaDB (LangChain handles embedding automatically)
    vectorstore = Chroma(
        collection_name="lecture_notes",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )
    vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
    print(f"  Stored in ChromaDB successfully (duplicates automatically skipped)")


# ── Run on all files in data/ ────────────────────────────────────
if __name__ == "__main__":
    for folder in ["./data/pdfs", "./data/audio"]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        for fname in os.listdir(folder):
            ingest_file(os.path.join(folder, fname))
    print("\nAll files ingested. ChromaDB is ready.")