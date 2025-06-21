import streamlit as st
import datetime
from pymongo import MongoClient
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOllama
from rag_chain import combined_qa_run, vectorstore_counsel, vectorstore_empathy
import time

# === MongoDB Setup ===
client = MongoClient("mongodb://localhost:27017/")
db = client["mental_health_bot"]
summary_collection = db["user_summaries"]
feedback_collection = db["feedback"]
blocked_patterns_collection = db["blocked_patterns"]  # New collection for learning from bad feedback

# === Enhanced Summarization Setup ===
llm = ChatOllama(
    model="llama3:instruct",
    temperature=0.7,  # Reduced for faster, more consistent responses
    num_ctx=2048     # Reduced context window for speed
)
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

def get_blocked_patterns():
    """Get patterns from questions that received 10+ bad feedbacks"""
    pipeline = [
        {"$match": {"feedback": "👎"}},
        {"$group": {"_id": "$question", "bad_count": {"$sum": 1}}},
        {"$match": {"bad_count": {"$gte": 10}}},
        {"$project": {"question": "$_id", "bad_count": 1}}
    ]
    return list(feedback_collection.aggregate(pipeline))

def analyze_and_store_bad_patterns():
    """Analyze badly rated questions and store patterns for LLaMA to learn from"""
    bad_patterns = get_blocked_patterns()
    
    for pattern in bad_patterns:
        question = pattern["question"]
        bad_responses = list(feedback_collection.find(
            {"question": question, "feedback": "👎"}
        ).limit(5))  # Get sample of bad responses
        
        # Create learning pattern
        learning_data = {
            "question_pattern": question,
            "bad_response_examples": [resp["response"] for resp in bad_responses],
            "guidance": f"When users ask similar questions to '{question}', avoid responses that are too generic, clinical, or dismissive. Focus on empathetic, personalized responses.",
            "created_at": datetime.datetime.utcnow()
        }
        
        blocked_patterns_collection.update_one(
            {"question_pattern": question},
            {"$set": learning_data},
            upsert=True
        )

def get_response_guidance(question):
    """Get guidance for LLaMA on how NOT to respond based on past bad feedback"""
    blocked_patterns = list(blocked_patterns_collection.find())
    
    guidance = ""
    for pattern in blocked_patterns:
        if any(word in question.lower() for word in pattern["question_pattern"].lower().split()):
            guidance += f"\n⚠️ Guidance: {pattern['guidance']}\n"
            guidance += f"❌ Avoid responses like: {pattern['bad_response_examples'][0][:100]}...\n"
    
    return guidance

# === Enhanced Combined QA Function ===
def enhanced_qa_run(query, user_question):
    """Enhanced QA with guidance"""
    # Get guidance from past bad feedback
    guidance = get_response_guidance(user_question)
    
    # Enhanced query with guidance
    enhanced_query = f"""
    {query}
    
    {guidance}
    
    Instructions: Provide an empathetic, personalized response. Avoid generic advice.
    """
    
    start_time = time.time()
    response = combined_qa_run(enhanced_query)
    end_time = time.time()
    
    return response

# === Streamlit UI Setup ===
st.set_page_config(page_title="🧠 Mental Health Coping Companion", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f9f9f9;}
        .stButton>button {
            padding: 0.25rem 0.75rem; 
            margin: 0.1rem;
            min-width: 60px;
        }
        .feedback-container {
            display: flex;
            gap: 10px;
            align-items: center;
            margin: 10px 0;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 8px;
        }
        .feedback-buttons {
            display: flex;
            gap: 5px;
        }
        .stChatMessage[data-testid="user"] {background-color: #e6f7f7;}
        .stChatMessage[data-testid="assistant"] {background-color: #f7e6f9;}
        .timestamp {font-size:0.7rem; color:gray; margin-top:-0.4rem;}
        .thinking-box {
            background-color: #e8f4f8;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #2196F3;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💬 Mental Health Coping Companion")

if not st.session_state.get("logged_in"):
    st.warning("⚠️ Please log in from the Home page.")
    st.stop()

user_id = st.session_state["username"]
stored_summary_doc = summary_collection.find_one({"user_id": user_id}) or {}
stored_summary = stored_summary_doc.get("summary", "")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_store" not in st.session_state:
    st.session_state.feedback_store = {}
if "last_input" not in st.session_state:
    st.session_state.last_input = None
    st.session_state.last_retrieved_docs_counsel = []
    st.session_state.last_retrieved_docs_empathy = []

# Handle pending user input
if "pending_user_input" in st.session_state:
    pending_input = st.session_state.pop("pending_user_input")
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({"role": "user", "content": pending_input, "time": timestamp})

    if check_crisis(pending_input):
        st.session_state.messages.append({"role": "assistant", "content": "**🚨 Crisis Detected:** Please contact a professional.", "time": timestamp})
    elif is_response_blocked(user_id, pending_input):
        st.session_state.messages.append({"role": "assistant", "content": "⚠️ This question has been flagged multiple times. Please rephrase.", "time": timestamp})
    else:
        with st.spinner("🧠 Generating response..."):
            query = f"User's Summary:\n{stored_summary}\n\nUser's Question:\n{pending_input}"
            
            # Use enhanced QA function
            response = enhanced_qa_run(query, pending_input)
        
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

# === Enhanced Feedback Section ===
pending_feedback = {k: v for k, v in st.session_state.feedback_store.items() if not v["submitted"]}

if pending_feedback:
    st.markdown("### 💬 Rate the last response:")
    
    for question, data in pending_feedback.items():
        st.markdown(f"**Question:** {question[:100]}{'...' if len(question) > 100 else ''}")
        
        # Create side-by-side feedback buttons
        col1, col2, col3 = st.columns([1, 1, 8])
        
        with col1:
            if st.button("👍 Good", key=f"good_{hash(question)}", help="This response was helpful"):
                feedback_collection.insert_one({
                    "user": user_id, "question": question, "response": data["response"],
                    "feedback": "👍", "timestamp": datetime.datetime.utcnow()
                })
                st.session_state.feedback_store[question]["submitted"] = True
                st.success("✅ Thank you for your feedback!")
                st.rerun()
        
        with col2:
            if st.button("👎 Bad", key=f"bad_{hash(question)}", help="This response needs improvement"):
                feedback_collection.insert_one({
                    "user": user_id, "question": question, "response": data["response"],
                    "feedback": "👎", "timestamp": datetime.datetime.utcnow()
                })
                st.session_state.feedback_store[question]["submitted"] = True
                
                # Check if this reaches 10 bad feedbacks and update learning patterns
                bad_count = feedback_collection.count_documents({"question": question, "feedback": "👎"})
                if bad_count >= 10:
                    analyze_and_store_bad_patterns()
                    st.warning(f"⚠️ This question pattern has received {bad_count} negative feedbacks. The AI will learn from this.")
                else:
                    st.info(f"📝 Feedback recorded. ({bad_count}/10 negative feedbacks for this pattern)")
                st.rerun()
        
        st.markdown("---")

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

with st.sidebar:
    st.header("⚙️ Session Options")
    
    # Performance stats
    bad_patterns_count = blocked_patterns_collection.count_documents({})
    st.metric("📚 Learned Patterns", bad_patterns_count)
    
    if st.button("🔄 Refresh Learning Patterns"):
        analyze_and_store_bad_patterns()
        st.success("✅ Learning patterns updated!")
    
    if st.button("✅ Save Summary"):
        generate_and_store_summary(user_id, st.session_state.messages)
        st.success("✅ Summary saved.")
        st.session_state.clear()
        st.rerun()
        
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()
        
    # Debug section
    with st.expander("🔧 Debug Info"):
        st.write("Session State Keys:", list(st.session_state.keys()))