import streamlit as st

st.set_page_config(page_title="Mental Health Coping Companion", layout="wide")
st.title("🧠 Mental Health Coping Companion (Offline RAG)")

user_input = st.text_area("How are you feeling today?", height=150)

if st.button("Get Support"):
    if not user_input.strip():
        st.warning("Please enter something.")
    else:
        st.info("🤖 Thinking... (LLM response will appear here)")
        # TODO: add RAG call here
