import os
import streamlit as st
from dotenv import load_dotenv
from ingest import process_and_index_codebase
from query import ask_codebase

load_dotenv()

# Sync Streamlit Community Cloud st.secrets into os.environ if present
for key in ["GOOGLE_API_KEY", "WEAVIATE_URL", "WEAVIATE_API_KEY"]:
    if not os.getenv(key) and key in st.secrets:
        os.environ[key] = st.secrets[key]

st.set_page_config(page_title="GitHub Codebase Chatter", page_icon="💻", layout="wide")
st.title("💻 GitHub Codebase Chatter (Code-Aware RAG)")
st.caption("AST-aware syntax chunking with LangChain, Weaviate Vector Search, and Google Gemini")

missing_keys = [
    key
    for key in ["GOOGLE_API_KEY", "WEAVIATE_URL", "WEAVIATE_API_KEY"]
    if not os.getenv(key)
]

if missing_keys:
    st.error(
        f"Missing required API keys: **{', '.join(missing_keys)}**. "
        "Add them in Streamlit Community Cloud **App settings > Secrets** (or your local `.env` file)."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("1. Ingest GitHub Repository")
    repo_url = st.text_input("Public GitHub Repo URL", "https://github.com/pallets/flask")
    collection_name = st.text_input("Weaviate Collection Name", "FlaskCodebase")

    if st.button("Clone & Index Repository", type="primary", use_container_width=True):
        if not repo_url.strip() or not collection_name.strip():
            st.warning("Please provide both a GitHub URL and a collection name.")
        else:
            try:
                with st.spinner("Cloning repo, splitting AST boundaries, and indexing into Weaviate..."):
                    chunk_count = process_and_index_codebase(
                        repo_url.strip(), index_name=collection_name.strip()
                    )
                    st.success(f"Indexed **{chunk_count}** syntax-aware code chunks into `{collection_name}`!")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.header("2. Ask Architectural & Code Questions")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View Retrieved Code Chunks & File Paths"):
                for src in msg["sources"]:
                    st.markdown(f"**`{src['file_path']}`** ({src['language']})")
                    st.code(src["snippet"], language=src["language"])

if prompt := st.chat_input("Ask a question (e.g., Where is request routing handled?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching Weaviate vectors & analyzing code..."):
                answer, sources = ask_codebase(prompt, index_name=collection_name.strip())
                st.markdown(answer)
                with st.expander("View Retrieved Code Chunks & File Paths"):
                    for src in sources:
                        st.markdown(f"**`{src['file_path']}`** ({src['language']})")
                        st.code(src["snippet"], language=src["language"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
        except Exception as e:
            st.error(f"Query failed: {e}")
