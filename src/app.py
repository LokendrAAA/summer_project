import streamlit as st
import datetime
import os
from rag_chain import qa_chain, vectorstore

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "self harm", "end my life", "want to die",
    "hurting myself", "cutting", "hopeless", "no reason to live"
]
JOURNAL_DIR = "journal_entries"
os.makedirs(JOURNAL_DIR, exist_ok=True)

def check_crisis(text):
    text = text.lower()
    return any(keyword in text for keyword in CRISIS_KEYWORDS)

st.set_page_config(page_title="🧠 Coping Companion", layout="wide")
st.title("Mental Health Coping Companion")

# Session-only conversation storage
if "messages" not in st.session_state:
    st.session_state.messages = []

def process_user_input():
    user_msg = st.session_state.user_input.strip()
    if not user_msg:
        return

    if check_crisis(user_msg):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "**🚨 Crisis Detected:** Please contact a mental health professional or a crisis hotline immediately."
        })
    else:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.spinner("Thinking..."):
            response = qa_chain.run(user_msg)
        st.session_state.messages.append({"role": "assistant", "content": response})

    st.session_state.user_input = ""  # Clear input after use

tab1, tab2 = st.tabs(["💬 Support Chat", "📓 Journaling"])

with tab1:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        else:
            st.chat_message("assistant").markdown(message["content"])

    st.text_input("Your message:", key="user_input", on_change=process_user_input)

with tab2:
    st.header("📓 Daily Journal")
    today = datetime.date.today().isoformat()
    journal_text = st.text_area("Write about your day:", height=250)

    if st.button("Save Journal Entry"):
        if journal_text.strip():
            filename = f"{JOURNAL_DIR}/{today}.txt"
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now().isoformat()}\n{journal_text}\n\n")
            st.success(f"Journal saved for {today}.")
        else:
            st.warning("Please write something before saving.")

    if st.checkbox("Show Previous Entries"):
        entries = sorted(os.listdir(JOURNAL_DIR))
        if not entries:
            st.info("No journal entries found yet.")
        else:
            for entry in entries:
                with open(os.path.join(JOURNAL_DIR, entry), encoding="utf-8") as f:
                    st.subheader(entry.replace(".txt", ""))
                    st.text(f.read())
