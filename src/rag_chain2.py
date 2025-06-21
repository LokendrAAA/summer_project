from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
# Vector store will be plugged in later

# Load model
llm = ChatOllama(model="llama3")

# Placeholder template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=Path("templates/empathetic_prompt.txt").read_text()
)

# Setup chain (mock retriever for now)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=None,  # <-- add vectorstore.as_retriever() later
    chain_type_kwargs={"prompt": prompt}
)
