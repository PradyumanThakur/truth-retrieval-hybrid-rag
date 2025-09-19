import os
from sentence_transformers import CrossEncoder, SentenceTransformer
import faiss
import pickle
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from utlis.preprocessing import normalize, load_dataset

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "ramayana_index.faiss")
VERSES_PATH = os.path.join(BASE_DIR, "..", "data", "valmiki-ramayana-verses.csv")
BM25_PATH = os.path.join(BASE_DIR, "ramayana_bm25.pkl")

# ---------- Singleton storage ----------
_models = None

# ---------- 1. Sentence Transformer ----------
def load_sentence_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Loading SentenceTransformer on {device}...")
    return SentenceTransformer("intfloat/e5-large-v2", device=device)


# ---------- 2. FAISS ----------
def load_faiss_index():
    print("[+] Loading FAISS index...")
    return faiss.read_index(FAISS_INDEX_PATH)


# ---------- 3. BM25 ----------
def load_bm25():
    print("[+] Loading BM25 index...")
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)


# ---------- 4. Verse Data ----------
def load_verses():
    print("[+] Loading Verse Data...")
    df = load_dataset(VERSES_PATH)
    chunks = []
    for _, row in df.iterrows():
        chunks.append({
            "id": row["ID"],
            "text": normalize(row["Verse_Text"]),
            "meta": {
                "book": row["Book"],
                "chapter": row["Chapter"],
                "verse_num": row["Verse_number"]
            }
        })
    return chunks


# ---------- 5. CrossEncoder ----------
def load_cross_encoder():
    print("[+] Loading CrossEncoder reranker...")
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


# ---------- 6. Llama.cpp (Mistral) ----------
def load_llama_cpp_model(repo_id, filename, local_dir="models", ctx_size=2048):
    """
    Downloads a GGUF model if not present locally, then loads it with llama.cpp.
    Uses GPU if available, else CPU.
    """
    os.makedirs(local_dir, exist_ok=True)
    local_path = Path(local_dir) / filename

    # Download from Hugging Face if not cached
    if not local_path.exists():
        print(f"[+] Downloading {filename} from {repo_id}...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
    else:
        print(f"[+] Using cached model at {local_path}")

    gpu_available = torch.cuda.is_available()

    try:
        print(f"[+] Loading Llama.cpp on {'GPU' if gpu_available else 'CPU'}...")
        return Llama(
            model_path=str(local_path),
            n_ctx=ctx_size,
            n_threads=os.cpu_count() or 4,
            seed=42,
            verbose=False,
            n_gpu_layers=-1 if gpu_available else 0
        )
    except Exception as e:
        print(f"[!] Error loading LLaMA: {e}. Falling back to CPU...")
        return Llama(
            model_path=str(local_path),
            n_ctx=ctx_size,
            n_threads=os.cpu_count() or 4,
            seed=42,
            verbose=False,
            n_gpu_layers=0
        )


# ---------- Singleton loader ----------
def load_all_models():
    """
    Load all models once and return as a dictionary.
    Subsequent calls return the same models (singleton).
    """
    global _models
    if _models is not None:
        return _models

    sentence_model = load_sentence_model()
    faiss_index = load_faiss_index()
    bm25_model = load_bm25()
    verses = load_verses()
    cross_encoder_model = load_cross_encoder()

    repo_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    filename = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    mistral_llm = load_llama_cpp_model(repo_id, filename)

    _models = {
        "sentence_model": sentence_model,
        "faiss_index": faiss_index,
        "bm25_model": bm25_model,
        "verses": verses,
        "cross_encoder_model": cross_encoder_model,
        "mistral_llm": mistral_llm,
    }
    return _models