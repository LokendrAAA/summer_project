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

# === Connect to BOTH ChromaDBs ===
vectorstore_counsel = Chroma(
    persist_directory="chroma_db",  # CounselChat dataset
    embedding_function=embedding
)

vectorstore_empathy = Chroma(
    persist_directory="chroma_db_empathy",  # EmpatheticDialogues dataset
    embedding_function=embedding
)

# === Load Prompt ===
with open("templates/empathetic_prompt.txt") as f:
    template_text = f.read()

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template_text
)

# === Combined Retrieval + Response Function ===
def combined_qa_run(query, k_each=1):
    docs_counsel = vectorstore_counsel.similarity_search(query, k=k_each)
    docs_empathy = vectorstore_empathy.similarity_search(query, k=k_each)

    combined_context = "\n\n".join(doc.page_content for doc in (docs_counsel + docs_empathy))
    final_prompt = prompt.format(context=combined_context, question=query)
    return llm.invoke(final_prompt).content

__all__ = ["combined_qa_run", "vectorstore_counsel", "vectorstore_empathy"]
