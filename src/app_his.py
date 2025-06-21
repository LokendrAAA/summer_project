import streamlit as st
import datetime
import os
from rag_chain import combined_qa_run, vectorstore_counsel, vectorstore_empathy

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

if "messages" not in st.session_state:
    st.session_state.messages = []

tab1, tab2 = st.tabs(["💬 Support Chat", "📓 Journaling"])

with tab1:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        else:
            st.chat_message("assistant").markdown(message["content"])

    user_input = st.chat_input("Your message:")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        if check_crisis(user_input):
            crisis_msg = "**🚨 Crisis Detected:** Please contact a mental health professional or a crisis hotline immediately."
            st.session_state.messages.append({"role": "assistant", "content": crisis_msg})
        else:
            with st.spinner("Thinking..."):
                response = combined_qa_run(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})

        st.rerun()


    with st.expander("🔎 Retrieved Documents (Debug Info)"):
        retrieved_docs_counsel = vectorstore_counsel.similarity_search(user_input or "", k=1)
        retrieved_docs_empathy = vectorstore_empathy.similarity_search(user_input or "", k=1)

        st.subheader("From Counsel Chat Dataset:")
        if retrieved_docs_counsel:
            st.text(retrieved_docs_counsel[0].page_content)
        else:
            st.text("No relevant document found in Counsel Chat.")

        st.subheader("From Empathetic Dialogues Dataset:")
        if retrieved_docs_empathy:
            st.text(retrieved_docs_empathy[0].page_content)
        else:
            st.text("No relevant document found in Empathy DB.")

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
