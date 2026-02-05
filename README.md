# 📄 Document RAG Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDF documents and ask natural-language questions about their content.

The system extracts text from PDFs, performs preprocessing and analysis, builds semantic embeddings using **Sentence Transformers**, indexes them with **FAISS**, and generates accurate answers using a **large language model (Mistral-7B)** through a clean web-based interface.

---

## 🚀 Features

- 📂 Upload multiple PDF documents
- 🔍 Semantic search using FAISS vector database
- 🧠 Context-aware question answering (RAG pipeline)
- 🏷 Optional Named Entity Recognition (NER)
- 📊 Document statistics and confidence scoring
- 💬 Interactive chat interface
- ⚡ FastAPI backend with REST APIs
- 🌐 HTML, CSS, and JavaScript frontend

---

## 🧠 Architecture Overview

Frontend (HTML / CSS / JS)
</br>
↓
</br>
FastAPI Backend
</br>
↓
</br>
PDF Text Extraction (pdfplumber)
</br>
↓
</br>
Text Cleaning & EDA
</br>
↓
</br>
Sentence Chunking
</br>
↓
</br>
Sentence Embeddings (MiniLM)
</br>
↓
</br>
FAISS Vector Index
</br>
↓
</br>
LLM Answer Generation (Mistral-7B)
</br>

---

## 🛠 Tech Stack

### Backend
- FastAPI
- Sentence-Transformers (`all-MiniLM-L6-v2`)
- FAISS
- HuggingFace Inference API
- Transformers (NER)
- NLTK
- pdfplumber
- Pandas / NumPy

### Frontend
- HTML5
- CSS3
- Vanilla JavaScript
- Font Awesome

---

## 📦 Project Structure

├── backend/
</br>
│ ├── rag_pipeline.py
</br>
│ ├── uploads/
</br>
│ └── app.py
</br>
│
</br>
├── frontend/
</br>
│ ├── index.html
</br>
│ ├── style.css
</br>
│ └── script.js
</br>
│
</br>
└── README.md
</br>


---

## ⚙️ Installation & Setup

### Clone Repository
```bash
git clone https://github.com/your-username/document-rag-assistant.git
cd document-rag-assistant
