import streamlit as st
import datetime
from pymongo import MongoClient
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOllama
from rag_chain import combined_qa_run, vectorstore_counsel, vectorstore_empathy

# === MongoDB Setup ===
client = MongoClient("mongodb://localhost:27017/")
db = client["mental_health_bot"]
summary_collection = db["user_summaries"]
feedback_collection = db["feedback"]

# === Summarization Setup ===
llm = ChatOllama(model="llama3:instruct")
summarize_chain = load_summarize_chain(llm, chain_type="stuff")

CRISIS_KEYWORDS = ["suicide", "kill myself", "self harm", "end my life", "want to die", "hurting myself", "cutting", "hopeless", "no reason to live"]

def check_crisis(text):
    return any(keyword in text.lower() for keyword in CRISIS_KEYWORDS)

def generate_and_store_summary(user_id, conversation):
    conversation_text = "\n".join([msg["content"] for msg in conversation if msg["role"] == "user"])
    if not conversation_text.strip(): return
    docs = [Document(page_content=conversation_text)]
    summary = summarize_chain.run(docs)
    summary_collection.update_one({"user_id": user_id}, {"$set": {"summary": summary, "last_updated": datetime.datetime.utcnow()}}, upsert=True)

def is_response_blocked(user_id, question):
    return feedback_collection.count_documents({"user": user_id, "question": question, "feedback": "👎"}) >= 10

# === Streamlit UI Setup ===
st.set_page_config(page_title="🧠 Mental Health Coping Companion", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f9f9f9;}
        .stButton>button {padding: 0.25rem 0.75rem; margin: 0.2rem;}
        .emoji-feedback {display: flex; gap: 0.5rem; margin-top: 0.2rem;}
        .stChatMessage[data-testid="user"] {background-color: #e6f7f7;}
        .stChatMessage[data-testid="assistant"] {background-color: #f7e6f9;}
        .timestamp {font-size:0.7rem; color:gray; margin-top:-0.4rem;}
    </style>
""", unsafe_allow_html=True)

st.title("💬 Mental Health Coping Companion")

if not st.session_state.get("logged_in"):
    st.warning("⚠️ Please log in from the Home page.")
    st.stop()

user_id = st.session_state["username"]
stored_summary_doc = summary_collection.find_one({"user_id": user_id}) or {}
stored_summary = stored_summary_doc.get("summary", "")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_store" not in st.session_state:
    st.session_state.feedback_store = {}
if "last_input" not in st.session_state:
    st.session_state.last_input = None
    st.session_state.last_retrieved_docs_counsel = []
    st.session_state.last_retrieved_docs_empathy = []

if "pending_user_input" in st.session_state:
    pending_input = st.session_state.pop("pending_user_input")
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({"role": "user", "content": pending_input, "time": timestamp})

    if check_crisis(pending_input):
        st.session_state.messages.append({"role": "assistant", "content": "**🚨 Crisis Detected:** Please contact a professional.", "time": timestamp})
    elif is_response_blocked(user_id, pending_input):
        st.session_state.messages.append({"role": "assistant", "content": "⚠️ This question has been flagged multiple times. Please rephrase.", "time": timestamp})
    else:
        with st.spinner("Thinking..."):
            query = f"User's Summary:\n{stored_summary}\n\nUser's Question:\n{pending_input}"
            response = combined_qa_run(query)
            st.session_state.messages.append({"role": "assistant", "content": response, "time": timestamp})
            st.session_state.feedback_store[pending_input] = {"response": response, "submitted": False}
            st.session_state.last_input = pending_input
            st.session_state.last_retrieved_docs_counsel = vectorstore_counsel.similarity_search(pending_input, k=2)
            st.session_state.last_retrieved_docs_empathy = vectorstore_empathy.similarity_search(pending_input, k=2)

# === Display Chat History ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        st.markdown(f"<div class='timestamp'>{msg['time']}</div>", unsafe_allow_html=True)

# === Feedback Section per Response (with compact emoji buttons) ===
for question, data in st.session_state.feedback_store.items():
    if not data["submitted"]:
        
        cols = st.columns(2)
        with cols[0]:
            if st.button("👍", key=f"good_{question}"):
                feedback_collection.insert_one({
                    "user": user_id, "question": question, "response": data["response"],
                    "feedback": "👍", "timestamp": datetime.datetime.utcnow()
                })
                st.session_state.feedback_store[question]["submitted"] = True
                st.success("Thank you for your feedback!")
        with cols[1]:
            if st.button("👎", key=f"bad_{question}"):
                feedback_collection.insert_one({
                    "user": user_id, "question": question, "response": data["response"],
                    "feedback": "👎", "timestamp": datetime.datetime.utcnow()
                })
                st.session_state.feedback_store[question]["submitted"] = True
                st.success("Feedback recorded. Thanks!")

# === Chat Input ===
user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.pending_user_input = user_input
    st.rerun()

# === Show Retrieved Documents ===
if st.session_state.last_input:
    with st.expander(f"🔎 Retrieved for: '{st.session_state.last_input}'", expanded=False):
        st.subheader("🗂 Counsel Dataset:")
        for i, doc in enumerate(st.session_state.last_retrieved_docs_counsel, 1):
            st.markdown(f"**Doc {i}:** {doc.page_content}")
        st.subheader("🗂 Empathetic Dataset:")
        for i, doc in enumerate(st.session_state.last_retrieved_docs_empathy, 1):
            st.markdown(f"**Doc {i}:** {doc.page_content}")

# === Sidebar Controls ===
with st.sidebar:
    st.header("⚙️ Session Options")
    if st.button("✅ Save Summary"):
        generate_and_store_summary(user_id, st.session_state.messages)
        st.success("✅ Summary saved.")
        st.session_state.clear()
        st.rerun()
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()
