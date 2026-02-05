# rag_pipeline.py
import cv2
import pytesseract
from pdf2image import convert_from_path
import pdfplumber
import fitz
from PIL import Image
import io
import os
import re
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import nltk
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Initialize global variables
VECTOR_INDEX = None
VECTOR_STORE = None
EMBEDDER = None
NER_PIPELINE = None
LLM_CLIENT = None


try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# top of rag_pipeline.py (global scope)
EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

NER_PIPELINE = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
LLM_CLIENT = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)

def process_uploaded_pdfs(file_paths: list[str]):
    global VECTOR_INDEX, VECTOR_STORE, EMBEDDER, NER_PIPELINE, LLM_CLIENT
    
    print(f"Processing {len(file_paths)} PDF files...")
    document = {}
    
    for file in file_paths:
        text = ""
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            document[file] = text
        except Exception as e:
            print(f"Error processing {file}: {e}")
            document[file] = ""
    
    # ---- CREATE DATAFRAME ----
    df = pd.DataFrame({
        "files": list(document.keys()),
        "text": list(document.values())
    })
    
    # ---- BASIC METRICS ----
    df['char_count'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    
    # ---- NOISE METRICS ----
    def noise_metric(text):
        return {
            "digit_ratio": len(re.findall(r"\d", text)) / max(len(text), 1),
            "special_char_ratio": len(re.findall(r"[^\w\s]", text)) / max(len(text), 1),
            "newline_ratio": len(re.findall(r"\n", text)) / max(len(text), 1)
        }
    
    noise_df = df['text'].apply(noise_metric).apply(pd.Series)
    noise_df = noise_df.add_prefix("noise_")
    df = pd.concat([df, noise_df], axis=1)
    
    # ---- PAGE QUALITY ----
    df['page_quality'] = np.where(df['char_count'] < 100, 'low', 'ok')
    
    # ---- TEXT CLEANING ----
    df['text'] = df['text'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    df['text'] = df['text'].apply(
        lambda x: re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else x
    )
    
    # ---- REMOVE DUPLICATES ----
    df = df.drop_duplicates(subset='text')
    
    # ---- VOCAB FEATURES ----
    def vocab_feature(text):
        if not isinstance(text, str):
            return pd.Series([0, 0])
        tokens = re.findall(r'\b[a-z]{3,}\b', text)
        if not tokens:
            return pd.Series([0, 0])
        unique_tokens = set(tokens)
        return pd.Series([
            len(unique_tokens),
            len(unique_tokens) / len(tokens)
        ])
    
    df[['unique_word_count', 'lexical_diversity']] = df['text'].apply(vocab_feature)
    
    # ---- NER EXTRACTION ----
    if NER_PIPELINE is not None:
        df_ner = df[df['page_quality'] == 'ok'].copy()
        
        def extract_entities(text):
            try:
                return NER_PIPELINE(text[:512])  # Limit text length for NER
            except:
                return []
        
        df_ner['entities'] = df_ner['text'].apply(extract_entities)
        
        def key_info_from_entities(entities):
            info = {
                "dates": [],
                "organizations": [],
                "amounts": [],
                "ids": []
            }
            for e in entities:
                if e['entity_group'] == 'DATE':
                    info['dates'].append(e['word'])
                elif e['entity_group'] == 'ORG':
                    info['organizations'].append(e['word'])
                elif e['entity_group'] in ['MONEY']:
                    info['amounts'].append(e['word'])
                elif e['entity_group'] in ['CARDINAL', 'ORDINAL']:
                    info['ids'].append(e['word'])
            return info
        
        df_ner['key_information'] = df_ner['entities'].apply(key_info_from_entities)
        df = df.merge(df_ner[['files', 'key_information']], on='files', how='left')
    else:
        df['key_information'] = [{"dates": [], "organizations": [], "amounts": [], "ids": []} for _ in range(len(df))]
    
    # ---- CREATE EMBEDDINGS AND VECTOR STORE ----
    sentences = []
    metadata = []
    
    for _, row in df.iterrows():
        for sent in nltk.sent_tokenize(row['text'][:2000]):  # Limit text for tokenization
            sentences.append(sent)
            metadata.append({
                "file": row['files'],
                "entities": row.get('key_information', {})
            })
    
    # Generate embeddings
    print(f"Generating embeddings for {len(sentences)} sentences...")
    embeddings = EMBEDDER.encode(
        sentences,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Create FAISS index
    dim = embeddings.shape[1]
    VECTOR_INDEX = faiss.IndexFlatIP(dim)
    VECTOR_INDEX.add(embeddings)
    
    # Create vector store
    VECTOR_STORE = {
        "sentences": sentences,
        "metadata": metadata,
        "dataframe": df  # Store dataframe for reference
    }
    
    print(f"Processing complete! Created index with {len(sentences)} sentence embeddings.")
    return True

# ---- QUERY FUNCTIONS ----
def llm(prompt: str, max_tokens: int = 2000, temperature: float = 0) -> str:
    """Wrapper for LLM call with error handling"""
    if LLM_CLIENT is None:
        return "Error: LLM client not initialized. Please check your API token."
    
    try:
        response = LLM_CLIENT.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent document understanding assistant. Use only the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def enhanced_semantic_search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Enhanced semantic search with better ranking and context assembly
    """
    if VECTOR_INDEX is None or VECTOR_STORE is None or EMBEDDER is None:
        return []
    
    q_emb = EMBEDDER.encode(
        [query],
        normalize_embeddings=True
    )
    scores, indices = VECTOR_INDEX.search(q_emb, top_k * 3)
    
    results = []
    seen_texts = set()
    
    for idx, score in zip(indices[0], scores[0]):
        if score < 0.2:
            continue
        
        text = VECTOR_STORE["sentences"][idx]
        text_key = text[:100]
        
        if text_key in seen_texts:
            continue
        
        seen_texts.add(text_key)
        
        # Get context window
        context_window = 1
        start_idx = max(0, idx - context_window)
        end_idx = min(len(VECTOR_STORE["sentences"]), idx + context_window + 1)
        
        context_sentences = VECTOR_STORE["sentences"][start_idx:end_idx]
        full_context = " ".join(context_sentences)
        
        if len(full_context) > 800:
            full_context = full_context[:800] + "..."
        
        results.append({
            "text": text,
            "full_context": full_context,
            "score": float(score),
            "metadata": VECTOR_STORE["metadata"][idx],
            "original_position": int(idx)
        })
        
        if len(results) >= top_k:
            break
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def format_context_for_llm(results: List[Dict], max_chars: int = 4000) -> str:
    """
    Format search results into a context string for LLM with length limit
    """
    context_parts = []
    total_chars = 0
    
    for i, result in enumerate(results, 1):
        source_text = f"[Source {i} from {result['metadata']['file']}]: {result['full_context']}"
        
        if total_chars + len(source_text) > max_chars:
            remaining_chars = max_chars - total_chars
            if remaining_chars > 100:
                truncated = source_text[:remaining_chars] + "..."
                context_parts.append(truncated)
                total_chars += len(truncated)
            break
        else:
            context_parts.append(source_text)
            total_chars += len(source_text)
    
    return "\n\n".join(context_parts)

def generate_answer(query: str, context: str) -> str:
    """
    Generate answer using LLM with the provided context
    """
    prompt = f"""Based on the following document excerpts, answer this question: {query}

DOCUMENT EXCERPTS:
{context}

INSTRUCTIONS:
1. Answer using ONLY information from the excerpts above
2. If information isn't found, say "Not found in the documents"
3. Be concise and specific
4. Reference which source the information came from

ANSWER:"""
    
    try:
        response = llm(prompt, max_tokens=1000)
        return response.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def ask_documents(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Search documents and generate answer
    """
    if VECTOR_INDEX is None or VECTOR_STORE is None:
        return {
            "answer": "Please process documents first using process_uploaded_pdfs()",
            "sources": [],
            "confidence": 0.0,
            "context_length": 0
        }
    
    search_results = enhanced_semantic_search(query, top_k=top_k)
    
    if not search_results:
        return {
            "answer": "No relevant content found in the documents.",
            "sources": [],
            "confidence": 0.0,
            "context_length": 0
        }
    
    context = format_context_for_llm(search_results, max_chars=3000)
    answer = generate_answer(query, context)
    
    # Prepare sources
    sources = []
    avg_score = 0.0
    
    for result in search_results:
        snippet = result['text']
        if len(snippet) > 150:
            snippet = snippet[:150] + "..."
        
        sources.append({
            "file": result["metadata"]["file"],
            "snippet": snippet,
            "relevance_score": result["score"]
        })
        avg_score += result["score"]
    
    avg_score = avg_score / len(search_results) if search_results else 0.0
    
    return {
        "answer": answer,
        "sources": sources,
        "confidence": float(avg_score),
        "context_length": len(context)
    }

def answer_query(query: str):
    """
    Called every time user asks a question
    FAST: only embedding + FAISS + LLM
    """
    return ask_documents(query, top_k=3)

# ---- MAIN TEST FUNCTION ----
if __name__ == "__main__":
    # Test with sample files
    files = [
        "doc1 - Copy (1).pdf",
        "doc1 - Copy (2).pdf",
        "doc1 - Copy (3).pdf",
        "doc1 - Copy (4).pdf",
        "doc1 - Copy (5).pdf",
        "doc1 - Copy (6).pdf",
        "doc1 - Copy (7).pdf",
        "doc1 - Copy (8).pdf",
        "doc1 - Copy (9).pdf",
        "doc1 - Copy (10).pdf"
    ]
    
    # Filter for existing files
    existing_files = [f for f in files if os.path.exists(f)]
    
    if existing_files:
        print(f"Found {len(existing_files)} files to process")
        process_uploaded_pdfs(existing_files)
        
        # Test queries
        test_queries = [
            "What is batch processing?",
            "What are the main topics?",
            "Summarize the key points"
        ]
        
        for query in test_queries:
            print("\n" + "="*80)
            print(f"QUESTION: {query}")
            result = answer_query(query)
            
            print(f"\nANSWER:\n{result['answer']}")
            print(f"\nConfidence: {result['confidence']:.2%}")
            
            if result['sources']:
                print(f"\n📚 TOP SOURCES:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['file']} (score: {source['relevance_score']:.3f})")
                    print(f"   {source['snippet']}")
    else:
        print("No PDF files found. Please ensure the PDF files exist in the current directory.")