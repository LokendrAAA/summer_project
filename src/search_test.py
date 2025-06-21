from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

embedding = OllamaEmbeddings(model="nomic-embed-text")

# Load the existing Chroma DB
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)

query = "I'm feeling anxious and overwhelmed."
results = vectorstore.similarity_search(query, k=2)

for r in results:
    print("🔎 Match:", r.page_content)
