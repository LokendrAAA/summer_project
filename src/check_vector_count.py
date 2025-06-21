from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

# Path to the local DB where you stored the embeddings
PERSIST_DIR = "chroma_db_empathy"

# Reuse same embedding function
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Load existing vectorstore
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding
)

# Check number of stored vectors
collection = vectorstore._collection
doc_count = collection.count()

print(f"🧠 Documents currently stored in ChromaDB: {doc_count}")
