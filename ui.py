import streamlit as st
import tempfile
import os
import re
from pathlib import Path
from fpdf import FPDF
from docx import Document as DocxDocument
from llama_index.core import VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.astra_db import AstraDBVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()

# --- 1. RAG Configuration ---
Settings.llm = Gemini(
    model="models/gemini-2.5-flash-lite", 
    api_key=os.environ.get("GOOGLE_API_KEY"),
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)
Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

# --- 2. Conversion Helpers ---
def convert_to_pdf(input_path, extension):
    """Converts .txt and .docx to a temporary PDF file."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    output_pdf_path = input_path.replace(extension, ".pdf")
    
    if extension == ".txt":
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                pdf.multi_cell(0, 10, txt=line.encode('latin-1', 'replace').decode('latin-1'))
    
    elif extension == ".docx":
        doc = DocxDocument(input_path)
        for para in doc.paragraphs:
            if para.text.strip():
                pdf.multi_cell(0, 10, txt=para.text.encode('latin-1', 'replace').decode('latin-1'))
    
    pdf.output(output_pdf_path)
    return output_pdf_path

def sanitize_collection_name(name):
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name).lower()
    return f"doc_{clean_name}"[:40]

# --- 3. UI Configuration ---
st.set_page_config(page_title="Smart Document Chatbot", page_icon="📄", layout="centered")
st.title("📄 Smart Document Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload a PDF, DOCX, or TXT file to begin!"}]
if "index" not in st.session_state:
    st.session_state.index = None

# --- 4. Sidebar: File Upload ---
with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("Select Document", type=["pdf", "docx", "txt"])
    
    # Automatic Processing Logic
    if uploaded_file:
        # Check if this specific file has already been indexed in this session
        # We use the file name as a simple unique identifier
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            
            dynamic_coll_name = sanitize_collection_name(uploaded_file.name)
            
            with st.status(f"Indexing {uploaded_file.name}...", expanded=True) as status:
                st.write("Preparing file...")
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    raw_path = tmp_file.name

                st.write("Converting to searchable format...")
                if suffix in [".docx", ".txt"]:
                    processing_path = convert_to_pdf(raw_path, suffix)
                else:
                    processing_path = raw_path

                st.write("Connecting to Astra DB...")
                astra_db_store = AstraDBVectorStore(
                    token=os.environ.get("ASTRA_DB_TOKEN"),
                    api_endpoint=os.environ.get("ASTRA_DB_ENDPOINT"),
                    collection_name=dynamic_coll_name,
                    embedding_dimension=384,
                )
                
                storage_context = StorageContext.from_defaults(vector_store=astra_db_store)
                documents = SimpleDirectoryReader(input_files=[processing_path]).load_data()
                
                st.session_state.index = VectorStoreIndex.from_documents(
                    documents, storage_context=storage_context
                )
                
                # Cleanup files
                if os.path.exists(raw_path): os.remove(raw_path)
                if processing_path != raw_path and os.path.exists(processing_path): os.remove(processing_path)
                
                # Update state so we don't re-run this until a NEW file is uploaded
                st.session_state.last_uploaded_file = uploaded_file.name
                status.update(label="Document Ready!", state="complete", expanded=False)
    else:
        # Reset index if file is removed
        st.session_state.index = None
        st.session_state.last_uploaded_file = None

# --- 5. Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is in this document?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.index:
        query_engine = st.session_state.index.as_query_engine()
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = query_engine.query(prompt)
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
    else:
        st.warning("Please upload a file first!")