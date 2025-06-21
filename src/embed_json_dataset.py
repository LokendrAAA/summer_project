import json
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# === CONFIG ===
INPUT_PATH = Path("data\empathetic_dialogues_prepared.jsonl")
PERSIST_DIR = "chroma_db"
BATCH_SIZE = 250
PROGRESS_FILE = Path("embedding_progress.txt")

# === Load Progress ===
def get_last_index():
    if PROGRESS_FILE.exists():
        return int(PROGRESS_FILE.read_text().strip())
    return 0

def save_progress(index):
    PROGRESS_FILE.write_text(str(index))

# === Load All Data ===
with INPUT_PATH.open("r", encoding="utf-8") as f:
    raw_lines = f.readlines()

print(f"🔍 Total documents: {len(raw_lines)}")

# === Setup Embedding and Vector DB ===
embedding = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding
)

start_idx = get_last_index()
print(f"⏩ Resuming from index {start_idx}...")

# === Batch Processing ===
for i in range(start_idx, len(raw_lines), BATCH_SIZE):
    batch = raw_lines[i:i+BATCH_SIZE]
    documents = []

    for line in batch:
        item = json.loads(line)
        context = item.get("Context", "")
        response = item.get("Response", "")
        full_text = f"Context: {context}\nResponse: {response}"
        documents.append(Document(page_content=full_text))

    print(f"🔄 Embedding docs {i} → {i + len(documents)}...")
    vectorstore.add_documents(documents)
    vectorstore.persist()

    save_progress(i + len(documents))
    print(f"✅ Batch {i // BATCH_SIZE + 1} complete.")

print("🎉 All documents embedded and stored.")
