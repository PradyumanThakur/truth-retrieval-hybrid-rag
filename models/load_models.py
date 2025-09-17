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
LLAMA_CPP_MISTRAL_gguf = os.path.join(BASE_DIR, "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Check if GPU is availabel
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}...")

# ---------- 1. Sentence Transformer (for embeddings) ----------
print("[+] Loading SentenceTransformer...")
sentence_model = SentenceTransformer("intfloat/e5-large-v2", device=device) #all-MiniLM-L6-v2, intfloat/e5-large-v2


# ---------- 2. FAISS Index ----------
print("[+] Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)


# ---------- 3. BM25 ----------
print("[+] Loading BM25 index...")
with open(BM25_PATH, "rb") as f:
    bm25_model = pickle.load(f)

# ---------- 4. Verses Metadata ----------
print("[+] Loading Verse Data as chunks...")
# Load data
df = load_dataset(VERSES_PATH)

# Create a list of chunk dicts
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


# ---------- 5. CrossEncoder Model ----------
print("[+] Loading CrossEncoder reranker...")
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')


# ---------- 6. Mistral Model Loading via Llama.cpp ----------  
print("[+] Loading Mistral via llama.cpp...")

def load_llama_cpp_model(repo_id, filename, local_dir="models", ctx_size=2048):
    """
    Downloads a GGUF model if not present locally, then loads it with llama.cpp
    using GPU if available (controlled by LLAMA_CUBLAS=1).
    """
    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)
    local_path = Path(local_dir) / filename

    # Download from Hugging Face if not already present
    if not local_path.exists():
        print(f"[+] Downloading {filename} from {repo_id}...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False  # store actual file, not symlink
        )
    else:
        print(f"[+] Using cached model at {local_path}")

    # Try to load with GPU, fallback to CPU
    try:
        if os.getenv("LLAMA_CUBLAS", "0") == "1":
            print("[+] Trying to load llama.cpp on GPU...")
            return Llama(
                model_path=str(local_path),
                n_ctx=ctx_size,
                n_threads=os.cpu_count() or 4,
                seed=42,
                verbose=False,
                n_gpu_layers=-1  # full GPU offload
            )
        else:
            raise RuntimeError("GPU flag not set, falling back to CPU")
    except Exception as e:
        print(f"[!] GPU load failed: {e}. Falling back to CPU...")
        return Llama(
            model_path=str(local_path),
            n_ctx=ctx_size,
            n_threads=os.cpu_count() or 4,
            seed=42,
            verbose=False,
            n_gpu_layers=0  # CPU only
        )

repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
mistral_llm = load_llama_cpp_model(repo_id, filename)

# ---------- Model Registry ----------
# All models centralized for reuse
models = {
    "sentence_model": sentence_model,
    "faiss_index": faiss_index,
    "bm25_model": bm25_model,
    "verses": chunks,
    "cross_encoder_model": cross_encoder_model,
    "mistral_llm": mistral_llm,
}

def load_all_models():
    return models