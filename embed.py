from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import json, os
from tqdm import tqdm

# === Setup ===
DATASET_PATH = "augesc_train.json"  # JSON file
CHROMA_DIR = "chroma_augesc_store"
COLLECTION_NAME = "augesc"
BATCH_SIZE = 256
PROGRESS_FILE = "progress.txt"

# === Load dataset ===

data = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))


# === ChromaDB and Embedding Setup ===
model = SentenceTransformer("all-MiniLM-L6-v2")
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# === Progress Resume ===
start_index = 0
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        start_index = int(f.read().strip())

# === Utility: extract usr and sys messages ===
def extract_usr_sys_pairs(convo_json_str):
    try:
        convo = json.loads(convo_json_str)
        user_lines = []
        last_sys = None
        for i in range(len(convo)):
            role, text = convo[i]
            if role == "usr":
                user_lines.append(text.strip())
            elif role == "sys":
                last_sys = text.strip()  # Keep latest assistant response
        return " ".join(user_lines), last_sys
    except Exception as e:
        print(f"⚠️ Error parsing conversation: {e}")
        return "", ""

# === Embed and Save ===
for i in tqdm(range(start_index, len(data), BATCH_SIZE), desc="Processing"):
    batch = data[i:i + BATCH_SIZE]
    ids, docs, metas = [], [], []

    for idx, item in enumerate(batch, start=i):
        user_text, assistant_text = extract_usr_sys_pairs(item["text"])
        if user_text and assistant_text:
            ids.append(f"conv_{idx}")
            docs.append(user_text)
            metas.append({"response": assistant_text})

    if not docs:
        continue

    embeddings = model.encode(docs, convert_to_tensor=False).tolist()
    
    try:
        collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metas
        )
    except Exception as e:
        print(f"❌ Error at batch starting index {i}: {e}")
        break

    with open(PROGRESS_FILE, "w") as f:
        f.write(str(i + BATCH_SIZE))

    print(f"✅ Saved up to {i + len(docs)}")

print("🎉 Done.")
