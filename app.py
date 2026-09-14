import os
import re
import io
import pickle
import hashlib
import requests

import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer

# ==================================================
# PAGE SETTINGS
# ==================================================
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📑",
    layout="wide"
)

st.title("📑 AI Document Assistant")
st.write("Upload your documents (PDF, DOCX, TXT) or paste a Google Drive link to ask questions!")

# ==================================================
# MODEL CACHING
# ==================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedding_model()

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def extract_text_from_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_text_from_docx(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text_from_txt(file_bytes):
    return file_bytes.decode("utf-8", errors="ignore")

def extract_gdrive_file_id(url):
    """Google Drive URL se File ID nikalta hai"""
    match = re.search(r'(?:/d/|id=)([\w-]+)', url)
    if match:
        return match.group(1)
    return None

def download_from_gdrive(file_id):
    """Google Drive Direct Export URL se file download karta hai"""
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(download_url)
    
    # Large files confirmation warning handle karne ke liye
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            download_url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(download_url)
            break
            
    return response.content

def process_file_content(file_bytes, file_name):
    file_name = file_name.lower()
    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif file_name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif file_name.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    else:
        # Default try PDF
        try:
            return extract_text_from_pdf(file_bytes)
        except Exception:
            return extract_text_from_txt(file_bytes)

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def build_vector_store(all_chunks):
    embeddings = embedder.encode(all_chunks, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    
    st.session_state["chunks"] = all_chunks
    st.session_state["index"] = index

# ==================================================
# SIDEBAR - FILE UPLOAD & GDRIVE LINK
# ==================================================
st.sidebar.header("📁 Document Source")

# Source Selection
source_option = st.sidebar.radio(
    "Choose Input Method:",
    ("Direct Upload", "Google Drive Link")
)

all_chunks = []

if source_option == "Direct Upload":
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    if uploaded_files and st.sidebar.button("Process Uploaded Files"):
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                text = process_file_content(file_bytes, uploaded_file.name)
                if text:
                    all_chunks.extend(chunk_text(text))
            
            if all_chunks:
                build_vector_store(all_chunks)
                st.sidebar.success(f"Processed {len(all_chunks)} chunks successfully!")
            else:
                st.sidebar.error("Could not extract text from files.")

elif source_option == "Google Drive Link":
    gdrive_url = st.sidebar.text_input("Paste Google Drive Sharing Link:")
    file_type = st.sidebar.selectbox("Select File Extension:", [".pdf", ".docx", ".txt"])
    
    if gdrive_url and st.sidebar.button("Fetch & Process Drive File"):
        file_id = extract_gdrive_file_id(gdrive_url)
        if not file_id:
            st.sidebar.error("Invalid Google Drive Link format!")
        else:
            with st.spinner("Downloading and processing Google Drive file..."):
                try:
                    file_bytes = download_from_gdrive(file_id)
                    text = process_file_content(file_bytes, f"file{file_type}")
                    if text:
                        all_chunks = chunk_text(text)
                        build_vector_store(all_chunks)
                        st.sidebar.success(f"Processed {len(all_chunks)} chunks from Google Drive!")
                    else:
                        st.sidebar.error("Failed to extract text. Make sure file access is 'Anyone with the link'.")
                except Exception as e:
                    st.sidebar.error(f"Error fetching file: {str(e)}")

# ==================================================
# MAIN CHAT & SEARCH INTERFACE
# ==================================================
query = st.text_input("💬 Ask a question based on uploaded documents:")

if query:
    if "index" in st.session_state and "chunks" in st.session_state:
        query_vector = embedder.encode([query]).astype("float32")
        k = 3  # Top 3 matching passages
        distances, indices = st.session_state["index"].search(query_vector, k)
        
        st.subheader("🔍 Top Matching Contexts:")
        for idx in indices[0]:
            if idx < len(st.session_state["chunks"]):
                st.info(st.session_state["chunks"][idx])
    else:
        st.warning("Please process a document or Google Drive link first!")
