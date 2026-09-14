python
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


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Assistant")
st.write(
    "Upload a PDF, DOCX, TXT, or MD document and ask questions "
    "about its content."
)


# =========================================================
# SETTINGS
# =========================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5

STORAGE_DIR = "document_store"

os.makedirs(STORAGE_DIR, exist_ok=True)


# =========================================================
# LOAD SENTENCE TRANSFORMER
# =========================================================

@st.cache_resource
def load_embedding_model():
    """
    Load the Sentence Transformer model once.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf(file):
    """
    Extract text from every PDF page.

    Returns:
        List of dictionaries containing text,
        file name and page number.
    """

    reader = PdfReader(file)

    extracted_data = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            extracted_data.append({
                "text": text,
                "file_name": file.name,
                "page": page_number
            })

    return extracted_data


# =========================================================
# DOCX EXTRACTION
# =========================================================

def extract_docx(file):
    """
    Extract text from a DOCX document.

    DOCX does not provide reliable page numbers through
    python-docx, so page is stored as None.
    """

    document = Document(file)

    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text)

    text = "\n".join(paragraphs)

    return [{
        "text": text,
        "file_name": file.name,
        "page": None
    }]


# =========================================================
# TXT EXTRACTION
# =========================================================

def extract_txt(file):
    """
    Extract text from a TXT file.
    """

    text = file.read().decode("utf-8", errors="ignore")

    return [{
        "text": text,
        "file_name": file.name,
        "page": None
    }]


# =========================================================
# MARKDOWN EXTRACTION
# =========================================================

def extract_md(file):
    """
    Extract text from a Markdown file.
    """

    text = file.read().decode("utf-8", errors="ignore")

    return [{
        "text": text,
        "file_name": file.name,
        "page": None
    }]


# =========================================================
# DOCUMENT EXTRACTION ROUTER
# =========================================================

def extract_document(file):
    """
    Select the correct extraction function
    according to the uploaded file type.
    """

    extension = file.name.lower().split(".")[-1]

    if extension == "pdf":
        return extract_pdf(file)

    elif extension == "docx":
        return extract_docx(file)

    elif extension == "txt":
        return extract_txt(file)

    elif extension == "md":
        return extract_md(file)

    else:
        return []


# =========================================================
# TEXT CHUNKING
# =========================================================

def create_chunks(extracted_data):
    """
    Split document text into overlapping chunks.

    Every chunk keeps:
    - file name
    - page number
    - chunk text
    """

    chunks = []

    for item in extracted_data:

        text = item["text"]

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "file_name": item["file_name"],
                    "page": item["page"]
                })

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# =========================================================
# CREATE DOCUMENT ID
# =========================================================

def create_document_id(file_bytes):
    """
    Create a unique ID from the document content.
    """

    return hashlib.md5(file_bytes).hexdigest()


# =========================================================
# SAVE EMBEDDINGS + METADATA
# =========================================================

def save_document_data(document_id, embeddings, chunks):
    """
    Save embeddings and chunk metadata so they can
    be reused later.
    """

    index = faiss.IndexFlatIP(embeddings.shape[1])

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    index.add(embeddings)

    index_path = os.path.join(
        STORAGE_DIR,
        f"{document_id}.index"
    )

    metadata_path = os.path.join(
        STORAGE_DIR,
        f"{document_id}.pkl"
    )

    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks


# =========================================================
# LOAD SAVED DATA
# =========================================================

def load_document_data(document_id):
    """
    Load previously created FAISS index and metadata.
    """

    index_path = os.path.join(
        STORAGE_DIR,
        f"{document_id}.index"
    )

    metadata_path = os.path.join(
        STORAGE_DIR,
        f"{document_id}.pkl"
    )

    if os.path.exists(index_path) and os.path.exists(metadata_path):

        index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            chunks = pickle.load(f)

        return index, chunks

    return None, None


# =========================================================
# KEYWORD SEARCH
# =========================================================

def get_keywords(question):
    """
    Extract important words from the question.

    Very common words are removed.
    """

    stop_words = {
        "what", "is", "are", "the", "a", "an",
        "of", "to", "in", "on", "for", "and",
        "or", "how", "why", "when", "where",
        "who", "does", "do", "this", "that",
        "with", "about", "from"
    }

    words = re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())

    keywords = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]

    return keywords


def keyword_score(question, text):
    """
    Calculate a simple keyword matching score.
    """

    keywords = get_keywords(question)

    if not keywords:
        return 0.0

    text_lower = text.lower()

    matches = 0

    for keyword in keywords:
        if keyword in text_lower:
            matches += 1

    return matches / len(keywords)


# =========================================================
# HYBRID SEARCH
# =========================================================

def hybrid_search(question, index, chunks, top_k=TOP_K):
    """
    Combine:

    1. FAISS semantic search
    2. Keyword search

    Final score:

        70% semantic score
        30% keyword score
    """

    if not chunks:
        return []

    # -----------------------------------------------------
    # Semantic search
    # -----------------------------------------------------

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(question_embedding)

    search_k = min(top_k * 3, len(chunks))

    semantic_scores, semantic_indexes = index.search(
        question_embedding,
        search_k
    )

    results = []

    for score, index_number in zip(
        semantic_scores[0],
        semantic_indexes[0]
    ):

        if index_number == -1:
            continue

        chunk = chunks[index_number]

        semantic_score = float(score)

        keyword_match = keyword_score(
            question,
            chunk["text"]
        )

        # -------------------------------------------------
        # Hybrid score
        # -------------------------------------------------

        hybrid_score = (
            0.70 * semantic_score
            + 0.30 * keyword_match
        )

        results.append({
            "text": chunk["text"],
            "file_name": chunk["file_name"],
            "page": chunk["page"],
            "semantic_score": semantic_score,
            "keyword_score": keyword_match,
            "hybrid_score": hybrid_score
        })

    # -----------------------------------------------------
    # Sort by hybrid score
    # -----------------------------------------------------

    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# SESSION STATE
# =========================================================

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "document_id" not in st.session_state:
    st.session_state.document_id = None


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "docx", "txt", "md"]
)


# =========================================================
# PROCESS DOCUMENT
# =========================================================

if uploaded_file:

    file_bytes = uploaded_file.getvalue()

    document_id = create_document_id(file_bytes)

    # Check whether document was processed before
    saved_index, saved_chunks = load_document_data(
        document_id
    )

    if saved_index is not None:

        st.session_state.index = saved_index
        st.session_state.chunks = saved_chunks
        st.session_state.document_id = document_id

        st.success(
            "This document was already processed. "
            "Saved embeddings were reused."
        )

    else:

        with st.spinner("Extracting document text..."):

            extracted_data = extract_document(
                uploaded_file
            )

        if not extracted_data:

            st.error(
                "No text could be extracted from this document."
            )
            st.stop()

        # -------------------------------------------------
        # Document information
        # -------------------------------------------------

        st.subheader("📋 Document Information")

        total_characters = sum(
            len(item["text"])
            for item in extracted_data
        )

        pages = [
            item["page"]
            for item in extracted_data
            if item["page"] is not None
        ]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**File Name:**")
            st.write(uploaded_file.name)

        with col2:
            st.write("**Extracted Characters:**")
            st.write(total_characters)

        with col3:
            if pages:
                st.write("**Pages:**")
                st.write(len(pages))
            else:
                st.write("**Pages:**")
                st.write("Not available")

        # -------------------------------------------------
        # Show extracted text
        # -------------------------------------------------

        with st.expander("View extracted text"):

            for item in extracted_data:

                if item["page"]:
                    st.markdown(
                        f"**Page {item['page']}**"
                    )

                st.write(item["text"])

        # -------------------------------------------------
        # Create chunks
        # -------------------------------------------------

        with st.spinner("Creating text chunks..."):

            chunks = create_chunks(
                extracted_data
            )

        st.subheader("✂️ Chunk Information")

        st.write(
            f"Created **{len(chunks)} chunks** "
            f"from this document."
        )

        # -------------------------------------------------
        # Create embeddings
        # -------------------------------------------------

        with st.spinner(
            "Creating embeddings with Sentence Transformer..."
        ):

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype("float32")

        # -------------------------------------------------
        # Save data
        # -------------------------------------------------

        with st.spinner(
            "Saving embeddings and metadata..."
        ):

            index, saved_chunks = save_document_data(
                document_id,
                embeddings,
                chunks
            )

        st.session_state.index = index
        st.session_state.chunks = saved_chunks
        st.session_state.document_id = document_id

        st.success(
            "Document processed successfully! "
            "Embeddings and metadata have been saved."
        )


# =========================================================
# QUESTION ANSWERING / SEARCH
# =========================================================

if (
    st.session_state.index is not None
    and st.session_state.chunks
):

    st.divider()

    st.subheader("🔎 Ask Questions About Your Document")

    question = st.text_input(
        "Enter your question:"
    )

    if question:

        with st.spinner(
            "Searching document..."
        ):

            results = hybrid_search(
                question,
                st.session_state.index,
                st.session_state.chunks
            )

        if results:

            st.subheader(
                "📌 Most Relevant Document Chunks"
            )

            for number, result in enumerate(
                results,
                start=1
            ):

                st.markdown(
                    f"### Result {number}"
                )

                st.write(
                    f"**File:** {result['file_name']}"
                )

                if result["page"] is not None:
                    st.write(
                        f"**Page:** {result['page']}"
                    )

                st.write(
                    f"**Semantic Score:** "
                    f"{result['semantic_score']:.3f}"
                )

                st.write(
                    f"**Keyword Score:** "
                    f"{result['keyword_score']:.3f}"
                )

                st.write(
                    f"**Hybrid Score:** "
                    f"{result['hybrid_score']:.3f}"
                )

                st.info(
                    result["text"]
                )

        else:

            st.warning(
                "No relevant chunks were found."
            )
else:

    st.info(
        "Upload a PDF, DOCX, TXT, or MD file to begin."
    )
```
