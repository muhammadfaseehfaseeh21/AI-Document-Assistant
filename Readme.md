```markdown
# 📄 AI Document Assistant

A simple Streamlit AI Document Assistant that can read:

- PDF
- DOCX
- TXT
- Markdown (MD)

The application extracts document text, creates overlapping chunks,
generates Sentence Transformer embeddings, stores them in FAISS,
and searches the document using both semantic and keyword search.

---

## 🚀 Features

### 1. Multiple Document Formats

The application supports:

- `.pdf`
- `.docx`
- `.txt`
- `.md`

Each format has its own extraction function.

---

### 2. Document Information

After uploading a document, the application shows:

- File name
- Extracted character count
- Number of pages when available
- Extracted text

PDF files keep page numbers.

DOCX, TXT and MD files do not have reliable page numbers,
so their page metadata is stored as `None`.

---

### 3. Text Chunking

Large documents are divided into smaller chunks.

The application uses:

```text
Chunk size = 800 characters
Overlap = 150 characters
```

The overlap helps preserve context between neighboring chunks.

Every chunk contains:

```text
text
file_name
page
```

The application also displays the total number of chunks created.

---

### 4. Sentence Transformer Embeddings

The application uses:

```text
all-MiniLM-L6-v2
```

from Sentence Transformers.

The document chunks are converted into numerical vectors called
embeddings.

These embeddings allow the application to perform semantic search.

---

### 5. Embedding Reuse

The application creates a unique ID from the uploaded document.

If the same document is uploaded again, the application checks:

```text
document_store/
```

If embeddings already exist, they are loaded instead of being
created again.

This saves processing time.

---

### 6. FAISS Semantic Search

FAISS is used to search the document embeddings.

When the user asks a question:

```text
Question
   ↓
Question Embedding
   ↓
FAISS Search
   ↓
Relevant Chunks
```

---

### 7. Keyword Search

The application also performs a simple keyword search.

Important words are extracted from the question.

The application checks how many of those words appear in each
document chunk.

---

### 8. Hybrid Search

The application combines semantic and keyword search.

The final score is:

```text
Hybrid Score =
    70% Semantic Score
    +
    30% Keyword Score
```

The chunks are then sorted according to their hybrid score.

The top 5 results are displayed.

---

## 📁 Project Structure

```text
ai-document-assistant/
│
├── app.py
├── requirements.txt
├── README.md
│
└── document_store/
```

The `document_store` folder is automatically created by the app.

It contains saved FAISS indexes and chunk metadata.

---

## ▶️ Run the Application

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

---

## 🔄 Application Workflow

```text
Upload Document
       ↓
Extract Text
       ↓
Show Document Information
       ↓
Create Overlapping Chunks
       ↓
Create Sentence Transformer Embeddings
       ↓
Save Embeddings + Metadata
       ↓
Create FAISS Index
       ↓
User Asks Question
       ↓
Create Question Embedding
       ↓
FAISS Semantic Search
       ↓
Keyword Search
       ↓
Combine Scores
       ↓
Hybrid Ranking
       ↓
Return Most Relevant Chunks
```

---

## 🧠 Technologies Used

- Python
- Streamlit
- PyPDF
- python-docx
- Sentence Transformers
- FAISS
- NumPy

---

## ⚠️ Important Note

This version is a document retrieval/search assistant.

It retrieves the most relevant document chunks but does not yet
use an LLM to generate a natural-language answer from those chunks.

An LLM such as Groq can be connected later:

```text
User Question
      ↓
Hybrid Search
      ↓
Relevant Chunks
      ↓
LLM
      ↓
Final Answer
```
```
