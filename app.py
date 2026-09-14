import os
import re
import pickle
import hashlib

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
st.write("Upload your documents (PDF, DOCX, TXT) and ask questions directly!")

# ==================================================
# MODEL CACHING
# ==================================================
@st.cache_resource
def load_embedding_model():
    # Model ko Streamlit cache mein load karte hain taaki baar baar load na ho
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedding_model()

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def process_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif file_name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif file_name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file format!")
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# ==================================================
# SIDEBAR - FILE UPLOAD & PROCESSING
# ==================================================
st.sidebar.header("📁 Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    if "chunks" not in st.session_state:
        all_chunks = []
        with st.spinner("Processing documents and building vector store..."):
            for uploaded_file in uploaded_files:
                text = process_file(uploaded_file)
                if text:
                    file_chunks = chunk_text(text)
                    all_chunks.extend(file_chunks)
            
            if all_chunks:
                embeddings = embedder.encode(all_chunks, show_progress_bar=False)
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(np.array(embeddings).astype("float32"))
                
                st.session_state["chunks"] = all_chunks
                st.session_state["index"] = index
                st.sidebar.success(f"Processed {len(all_chunks)} text chunks successfully!")
            else:
                st.sidebar.warning("No readable text found in uploaded documents.")

# ==================================================
# MAIN CHAT & SEARCH INTERFACE
# ==================================================
query = st.text_input("💬 Ask a question based on uploaded documents:")

if query:
    if "index" in st.session_state and "chunks" in st.session_state:
        query_vector = embedder.encode([query]).astype("float32")
        k = 3  # Top 3 relevant chunks
        distances, indices = st.session_state["index"].search(query_vector, k)
        
        st.subheader("🔍 Top Matching Contexts:")
        for idx in indices[0]:
            if idx < len(st.session_state["chunks"]):
                st.info(st.session_state["chunks"][idx])
    else:
        st.warning("Please upload at least one document first!")
