from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Create embedding model (use the one you pulled via Ollama)
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Sample text documents (like coping strategies)
docs = [
    Document(page_content="Try 5-4-3-2-1 grounding technique for anxiety."),
    Document(page_content="Practice deep breathing when feeling overwhelmed."),
    Document(page_content="Write your thoughts in a journal to reduce stress.")
]

# Create a Chroma vector store and persist it locally
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory="chroma_db")
print("✅ Documents embedded and stored!")
