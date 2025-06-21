import streamlit as st
import datetime
import os
from rag_chain import qa_chain, vectorstore  # Make sure to import vectorstore

# === CONFIG ===
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "self harm", "end my life", "want to die",
    "hurting myself", "cutting", "hopeless", "no reason to live"
]
JOURNAL_DIR = "journal_entries"
os.makedirs(JOURNAL_DIR, exist_ok=True)

def check_crisis(text):
    text = text.lower()
    return any(keyword in text for keyword in CRISIS_KEYWORDS)

# === STREAMLIT PAGE ===
st.set_page_config(page_title="🧠 Coping Companion", layout="wide")
st.title("Mental Health Coping Companion")

tab1, tab2 = st.tabs(["💬 Support Chat", "📓 Journaling"])

# === SUPPORT CHAT TAB ===
with tab1:
    user_input = st.text_area("How are you feeling today?", height=150)

    if st.button("Get Support"):
        if user_input.strip():
            if check_crisis(user_input):
                st.markdown("### 🚨 Crisis Detected")
                st.error("Your message suggests you're going through a very difficult time.\n\n"
                         "**Please contact a mental health professional or a crisis hotline immediately.**\n\n"
                         "_This chatbot cannot provide urgent or life-saving support._")
            else:
                with st.spinner("Thinking..."):
                    result = qa_chain.run(user_input)
                    st.markdown("### 🤖 Response")
                    st.success(result)

                # OPTIONAL: Debug - Show retrieved document
                with st.expander("🔍 Retrieved Document (Debug Info)"):
                    retrieved_docs = vectorstore.similarity_search(user_input, k=1)
                    if retrieved_docs:
                        st.write(retrieved_docs[0].page_content)
                    else:
                        st.write("No relevant document found.")
        else:
            st.warning("Please enter something.")

# === JOURNALING TAB ===
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



# import streamlit as st
# from rag_chain import qa_chain, vectorstore  # Make sure to import vectorstore

# st.set_page_config(page_title="🧠 Coping Companion", layout="wide")
# st.title("Mental Health Coping Companion")

# user_input = st.text_area("How are you feeling today?", height=150)
# CRISIS_KEYWORDS = [
#     "suicide", "kill myself", "self harm", "end my life", "want to die",
#     "hurting myself", "cutting", "hopeless", "no reason to live"
# ]

# def check_crisis(text):
#     text = text.lower()
#     return any(keyword in text for keyword in CRISIS_KEYWORDS)

# if st.button("Get Support"):
#     if user_input.strip():
#         if check_crisis(user_input):
#             st.markdown("### 🚨 Crisis Detected")
#             st.error("Your message suggests you're going through a very difficult time.\n\n"
#                      "**Please contact a mental health professional or a crisis hotline immediately.**\n\n"
#                      "_This chatbot cannot provide urgent or life-saving support._")
#         else:
#             with st.spinner("Thinking..."):
#                 result = qa_chain.run(user_input)
#                 st.markdown("### 🤖 Response")
#                 st.success(result)
#     else:
#         st.warning("Please enter something.")













# if st.button("Get Support"):
#     if user_input.strip():
#         # 🔎 Debug block: Show the most relevant document retrieved
#         with st.expander("🔍 Retrieved Document (Debug Info)"):
#             retrieved_docs = vectorstore.similarity_search(user_input, k=1)
#             if retrieved_docs:
#                 st.write(retrieved_docs[0].page_content)
#             else:
#                 st.write("No relevant document found.")

#         # 🤖 Call the QA chain
#         with st.spinner("Thinking..."):
#             result = qa_chain.run(user_input)
#             st.markdown("### 🤖 Response")
#             st.success(result)
#     else:
#         st.warning("Please enter something.")
