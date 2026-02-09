from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.astra_db import AstraDBVectorStore
from llama_index.core.node_parser import SentenceSplitter
import os
from dotenv import load_dotenv

load_dotenv()

def init_rag_settings():
    # Initialize Embedding Model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def get_index(documents):
    # Astra DB Configuration
    token = os.environ.get("ASTRA_DB_TOKEN")
    api_endpoint = os.environ.get("ASTRA_DB_ENDPOINT")
    collection = "smart_doc_collection"
    
    astra_db_store = AstraDBVectorStore(
        token=token,
        api_endpoint=api_endpoint,
        collection_name=collection,
        embedding_dimension=384,
    )
    
    try:
        # Access the underlying AstraDB client to clear the collection
        astra_db_store._astra_db.collection(collection).delete_many({})
    except Exception as e:
        print(f"Note: Could not clear collection: {e}")

    storage_context = StorageContext.from_defaults(vector_store=astra_db_store)
    
    # 3. Create fresh index
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=False
    )
    return index

def query_rag(index, query_text):
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return response.response