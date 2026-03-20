import streamlit as st
import json
import os

from backend.ai_engine import get_answer

st.set_page_config(
    page_title="AI Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# DARK STYLE
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #121212;
}
.chat-user {
    background-color: #505050;
    padding: 10px;
    border-radius: 12px;
    margin: 5px 0;
    text-align: right;
    color: white;
}
.chat-ai {
    padding: 10px;
    margin: 5px 0;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.title("Menu")

    with st.expander("Downloads"):
        st.write("Download subjects here")

    with st.expander("Chats"):
        st.write("Chat history")

# -------------------------------
# LOAD CHAT
# -------------------------------
CHAT_PATH = "chat.json"

if os.path.exists(CHAT_PATH):
    with open(CHAT_PATH, "r") as f:
        chat_data = json.load(f)
else:
    chat_data = {"messages": []}

# -------------------------------
# HEADER
# -------------------------------
st.title("Chat Bot - Assistant")

# -------------------------------
# DISPLAY CHAT
# -------------------------------
chat_container = st.container()

with chat_container:
    for msg in chat_data["messages"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">{msg["text"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-ai">{msg["text"]}</div>',
                unsafe_allow_html=True
            )

# -------------------------------
# INPUT
# -------------------------------
user_input = st.chat_input("Ask me anything...")

if user_input:

    # Add user message
    chat_data["messages"].append({
        "role": "user",
        "text": user_input
    })

    # AI processing
    answer = get_answer(user_input)

    # Add AI response
    chat_data["messages"].append({
        "role": "ai",
        "text": answer
    })

    # Save JSON
    with open(CHAT_PATH, "w") as f:
        json.dump(chat_data, f, indent=2)

    st.rerun()
