import streamlit as st
from llm import init_llm
from rag import init_rag_settings, get_index
from document import save_uploaded_file, load_documents

st.set_page_config(page_title="Smart Doc Chatbot", layout="centered")

# Callback to handle automatic indexing
def process_new_file():
    if st.session_state.file_uploader_key is not None:
        # Initialize backend
        init_llm()
        init_rag_settings()
        
        # Save and Load
        save_uploaded_file(st.session_state.file_uploader_key)
        docs = load_documents()
        
        # Index (No spinner here, so it happens silently in the background)
        st.session_state.index = get_index(docs)

# Initialize Session States
if "index" not in st.session_state:
    st.session_state.index = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Document Setup")
    # key and on_change handle the automatic trigger
    st.file_uploader(
        "Upload Document", 
        type=["pdf", "docx"], 
        key="file_uploader_key", 
        on_change=process_new_file
    )
    
    if st.session_state.index:
        st.success("Document active.")

# Main Interface
st.title("📄 Smart Document Chatbot")

# Disable chat until index exists
chat_disabled = st.session_state.index is None

if chat_disabled:
    st.info("Please upload a document in the sidebar to start chatting.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question...", disabled=chat_disabled):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # We keep the spinner only for the LLM response so the user knows the AI is 'thinking'
        from rag import query_rag
        with st.spinner("Searching..."):
            response = query_rag(st.session_state.index, prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})