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
↓
FastAPI Backend
↓
PDF Text Extraction (pdfplumber)
↓
Text Cleaning & EDA
↓
Sentence Chunking
↓
Sentence Embeddings (MiniLM)
↓
FAISS Vector Index
↓
LLM Answer Generation (Mistral-7B)


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
│ ├── rag_pipeline.py
│ ├── uploads/
│ └── app.py
│
├── frontend/
│ ├── index.html
│ ├── style.css
│ └── script.js
│
└── README.md


---

## ⚙️ Installation & Setup

### Clone Repository
```bash
git clone https://github.com/your-username/document-rag-assistant.git
cd document-rag-assistant
