
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline import process_uploaded_pdfs, answer_query
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    file_paths = []

    for file in files:
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_paths.append(path)

    process_uploaded_pdfs(file_paths)

    return {"message": "Documents uploaded & processed successfully"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "RAG API is running"}

@app.get("/stats")
async def get_stats():
    """Return current system statistics"""
    try:
        from rag_pipeline import VECTOR_STORE, VECTOR_INDEX
        stats = {
            "documents_loaded": 0,
            "sentences_indexed": 0,
            "index_ready": VECTOR_INDEX is not None,
            "embedding_dimension": 0
        }
        
        if VECTOR_STORE:
            stats["documents_loaded"] = len(set(m["file"] for m in VECTOR_STORE.get("metadata", [])))
            stats["sentences_indexed"] = len(VECTOR_STORE.get("sentences", []))
            
        if VECTOR_INDEX:
            stats["embedding_dimension"] = VECTOR_INDEX.d
            
        return stats
    except:
        return {"error": "Statistics not available"}

@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question")
    result = answer_query(question)
    return result
