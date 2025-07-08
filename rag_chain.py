from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
import os

# === Load Embeddings and LLM ===
embedding = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(
    model="llama3:instruct",
    temperature=0.3,
    num_predict=256,
    stream=True
)

# === Connect to AUGESC ChromaDB ===
vectorstore_augsec = Chroma(
    persist_directory="chroma_augesc_store",
    embedding_function=embedding
)

# === Load Prompt ===
prompt_path = os.path.join(os.path.dirname(__file__), "templates\empathetic_prompt.txt")
with open(prompt_path, "r", encoding="utf-8") as f:
    template_text = f.read()

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template_text
)

# === Combined Retrieval + Response Function ===
def combined_qa_run(query, k_each=3):
    docs_augsec = vectorstore_augsec.similarity_search(query, k=k_each)
    combined_context = "\n\n".join(doc.page_content for doc in docs_augsec)
    final_prompt = prompt.format(context=combined_context, question=query)
    return llm.invoke(final_prompt).content

__all__ = ["combined_qa_run", "vectorstore_augsec"]