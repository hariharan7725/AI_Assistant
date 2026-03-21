# 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

import os
import re
import time
import json
import pickle
import faiss
import numpy as np
import streamlit as st

from huggingface_hub import InferenceClient
from openai import OpenAI

st.write("HF token loaded:", bool(st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))))
# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MEMORY_FOLDER = "memory"
CHAT_DIR = "chats"
CHAT_FILE = os.path.join(CHAT_DIR, "chat1.json")

TOP_K = 2
SUBJECT_THRESHOLD = 0.35

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

os.makedirs(CHAT_DIR, exist_ok=True)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Assist",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# THEME / STYLING
# --------------------------------------------------

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #111111;
    color: #f1f1f1;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stAppViewContainer"] > .main {
    background: #111111;
}

.block-container {
    max-width: 900px;
    padding-top: 1.2rem;
    padding-bottom: 6rem;
}

.main-title {
    font-size: 2rem;
    font-weight: 700;
    color: #f5f5f5;
    margin-bottom: 0.25rem;
}

.sub-title {
    color: #a3a3a3;
    margin-bottom: 1.2rem;
}

[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 0.2rem 0.4rem;
}

[data-testid="stChatMessageContent"] {
    border-radius: 16px;
    padding: 0.85rem 1rem;
    font-size: 1rem;
    line-height: 1.6;
}

section[data-testid="stSidebar"] {
    background: #1a1a1a;
}

div[data-testid="stChatInput"] {
    background: #111111;
}

div[data-testid="stChatInput"] textarea {
    background: #262626 !important;
    color: #f5f5f5 !important;
    border-radius: 14px !important;
}

.stAlert {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TOKEN
# --------------------------------------------------

HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def load_chat():
    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"chat_id": "chat1", "messages": []}


def save_chat(chat_data):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)


def l2_normalize(vec):
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def cosine_similarity_single_to_many(query_vec, matrix):
    query_vec = np.array(query_vec, dtype=np.float32)
    matrix = np.array(matrix, dtype=np.float32)

    query_norm = np.linalg.norm(query_vec)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = (query_norm * matrix_norms) + 1e-12
    sims = np.dot(matrix, query_vec) / denom
    return sims


def detect_word_limit(question):
    match = re.search(r'(\d+)\s*words?', question.lower())
    words = int(match.group(1)) if match else 120
    tokens = int(words * 1.4)
    return words, tokens

# --------------------------------------------------
# API CLIENTS
# --------------------------------------------------

@st.cache_resource
def get_embed_client():
    if not HF_TOKEN:
        return None
    return InferenceClient(
        provider="hf-inference",
        api_key=HF_TOKEN,
    )


@st.cache_resource
def get_gen_client():
    if not HF_TOKEN:
        return None
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )

# --------------------------------------------------
# SUBJECT KEYWORDS
# --------------------------------------------------

@st.cache_resource
def load_subject_keywords():
    subject_keywords = {}

    if not os.path.exists(MEMORY_FOLDER):
        return subject_keywords

    for subject in os.listdir(MEMORY_FOLDER):
        subject_path = os.path.join(MEMORY_FOLDER, subject)

        if not os.path.isdir(subject_path):
            continue

        value_file = os.path.join(subject_path, "values.txt")
        if not os.path.exists(value_file):
            continue

        with open(value_file, "r", encoding="utf8") as f:
            content = f.read()
            keywords = re.findall(r'"(.*?)"', content)

        if keywords:
            subject_keywords[subject] = keywords

    return subject_keywords

# --------------------------------------------------
# EMBEDDING HELPERS
# --------------------------------------------------

def embed_text(text):
    embed_client = get_embed_client()
    if embed_client is None:
        raise RuntimeError("HF_TOKEN is missing. Add it in environment variable or Streamlit secrets.")

    result = embed_client.feature_extraction(
        text,
        model=EMBED_MODEL_ID,
    )
    return l2_normalize(result)


def embed_texts(texts):
    embed_client = get_embed_client()
    if embed_client is None:
        raise RuntimeError("HF_TOKEN is missing. Add it in environment variable or Streamlit secrets.")

    vectors = embed_client.feature_extraction(
        texts,
        model=EMBED_MODEL_ID,
    )
    return np.array([l2_normalize(v) for v in vectors], dtype=np.float32)

# --------------------------------------------------
# BUILD EMBEDDINGS LAZILY
# --------------------------------------------------

@st.cache_resource
def build_subject_keyword_embeddings():
    subject_keywords = load_subject_keywords()
    subject_keyword_embeddings = {}

    for subject, keywords in subject_keywords.items():
        if keywords:
            emb = embed_texts(keywords)
            subject_keyword_embeddings[subject] = emb

    return subject_keyword_embeddings

# --------------------------------------------------
# DETECT SUBJECT
# --------------------------------------------------

def detect_subject(question):
    subject_keyword_embeddings = build_subject_keyword_embeddings()
    q_embed = embed_text(question)

    best_subject = None
    best_score = 0.0

    for subject, emb in subject_keyword_embeddings.items():
        scores = cosine_similarity_single_to_many(q_embed, emb)
        score = float(np.max(scores))

        if score > best_score:
            best_score = score
            best_subject = subject

    if best_score < SUBJECT_THRESHOLD:
        return None, best_score

    return best_subject, best_score

# --------------------------------------------------
# MAIN ANSWERING FUNCTION
# --------------------------------------------------

def get_answer(question):
    start_time = time.time()

    try:
        gen_client = get_gen_client()
        if gen_client is None:
            return "HF_TOKEN is missing. Set it first."

        subject, score = detect_subject(question)

        if subject is None:
            return "Question not related to available subjects."

        vector_path = os.path.join(MEMORY_FOLDER, subject, "vector_db.faiss")
        chunk_path = os.path.join(MEMORY_FOLDER, subject, "chunks.pkl")

        if not os.path.exists(vector_path):
            return f"Missing vector DB for subject: {subject}"

        if not os.path.exists(chunk_path):
            return f"Missing chunks for subject: {subject}"

        index = faiss.read_index(vector_path)

        with open(chunk_path, "rb") as f:
            texts = pickle.load(f)

        query_embedding = np.array([embed_text(question)], dtype=np.float32)
        scores, ids = index.search(query_embedding, TOP_K)

        context_parts = []
        for score_item, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            if score_item < 0.45:
                continue
            context_parts.append(texts[idx][:350])

        context = "\n".join(context_parts).strip()

        if not context:
            return "I could not find enough relevant context."

        words, _ = detect_word_limit(question)

        system_prompt = (
            f"You are a tutor for the subject: {subject}. "
            f"Use the provided context only. "
            f"Answer clearly in about {words} words."
        )

        user_prompt = f"""Context:
{context}

Question:
{question}
"""

        completion = gen_client.chat.completions.create(
            model=GEN_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=int(words * 1.4),
            top_p=0.9,
        )

        answer = completion.choices[0].message.content.strip()

        end_time = time.time()
        print("Subject:", subject)
        print("Similarity Score:", round(score, 3))
        print("Time taken:", round(end_time - start_time, 2), "seconds")

        return answer

    except Exception as e:
        return f"API error: {str(e)}"

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("### AI Assist")
    st.caption("Grey chat UI")
    if st.button("Clear chat"):
        st.session_state.chat_data = {"chat_id": "chat1", "messages": []}
        save_chat(st.session_state.chat_data)
        st.rerun()

    st.markdown("### Subjects")
    if os.path.exists(MEMORY_FOLDER):
        for item in os.listdir(MEMORY_FOLDER):
            if os.path.isdir(os.path.join(MEMORY_FOLDER, item)):
                st.write(f"- {item}")

# --------------------------------------------------
# SESSION
# --------------------------------------------------

if "chat_data" not in st.session_state:
    st.session_state.chat_data = load_chat()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown('<div class="main-title">AI Assist</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask questions from your subject memory in a chat-style interface.</div>', unsafe_allow_html=True)

if not HF_TOKEN:
    st.warning("HF_TOKEN is not set. Add it locally with environment variable or .streamlit/secrets.toml")

# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for msg in st.session_state.chat_data["messages"]:
    with st.chat_message("user"):
        st.markdown(msg["question"])

    with st.chat_message("assistant"):
        st.markdown(msg["answer"])

# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

prompt = st.chat_input("Ask your question here...")

if prompt:
    st.session_state.chat_data["messages"].append({
        "question": prompt,
        "answer": ""
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_answer(prompt)
            st.markdown(answer)

    st.session_state.chat_data["messages"][-1]["answer"] = answer
    save_chat(st.session_state.chat_data)
    st.rerun()
