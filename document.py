import os
from llama_index.core import SimpleDirectoryReader

UPLOAD_DIR = "./temp_docs"

def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    # Clear directory to ensure only current doc is indexed
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        os.unlink(file_path)
        
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_documents():
    if not os.path.exists(UPLOAD_DIR) or not os.listdir(UPLOAD_DIR):
        return None
    reader = SimpleDirectoryReader(input_dir=UPLOAD_DIR)
    return reader.load_data()