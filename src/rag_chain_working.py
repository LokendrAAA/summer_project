from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# Load embeddings and language model
embedding = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(
    model="llama3:instruct",
    temperature=0.3,
    num_predict=256,
    stream=True
)

# Connect to ChromaDB
vectorstore = Chroma(
    persist_directory="chroma_db_empathy",
    embedding_function=embedding
)

# Load prompt
with open("templates/empathetic_prompt.txt") as f:
    template_text = f.read()

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template_text
)

# Setup RAG QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context"
    }
)
__all__ = ["qa_chain", "vectorstore"]

