# eval.py
"""
Simple retrieval evaluation script.
Run this AFTER you've ingested your lecture content into ChromaDB.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="lecture_notes",
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
)

# EDIT THIS: add your own (question, expected_source_filename) pairs
# based on content you've actually uploaded
test_cases = [
    {"question": "What is supervised learning?", "expected_source": "lecture1.pdf"},
    {"question": "What is the difference between classification and regression?", "expected_source": "lecture1.pdf"},
    # add 10-15 more based on your real uploaded files
]

def evaluate():
    correct = 0
    for case in test_cases:
        results = vectorstore.similarity_search(case["question"], k=3)
        retrieved_sources = [r.metadata.get("source") for r in results]

        hit = case["expected_source"] in retrieved_sources
        correct += hit

        status = "✅" if hit else "❌"
        print(f"{status} Q: {case['question']}")
        print(f"   Expected: {case['expected_source']} | Retrieved: {retrieved_sources}\n")

    accuracy = correct / len(test_cases)
    print(f"\nRetrieval accuracy: {correct}/{len(test_cases)} ({accuracy:.0%})")

if __name__ == "__main__":
    evaluate()