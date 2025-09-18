import os
import numpy as np
import pickle
import faiss
import torch
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import string

nltk.download("punkt")
nltk.download('punkt_tab')

from models.load_models import load_all_models

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "ramayana_index.faiss")
EMBED_PATH = os.path.join(BASE_DIR, "ramayana_embeddings.npy")
BM25_PATH = os.path.join(BASE_DIR, "ramayana_bm25.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "valmiki-ramayana-verses.csv")


# Check if GPU is availabel
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}...")


def generate_all_indexes():
    # Load models and data
    models = load_all_models()
    chunks = models['verses']

    # Only build FAISS index if not already present
    if not os.path.exists(FAISS_INDEX_PATH):
        print("[+] Building FAISS index...")

        # Generate embeddings if not already saved
        if os.path.exists(EMBED_PATH):
            print("[+] Loading precomputed embeddings...")
            embeddings = np.load(EMBED_PATH)
        else:
            print("[+] Computing embeddings...")

            # Prepare inputs
            corpus = [chunk['text'] for chunk in chunks]
            # Encode and normalize
            embedder = models['sentence_model']
            embeddings = embedder.encode(corpus, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
            np.save(EMBED_PATH, embeddings)
        
        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
        print("[✓] FAISS index created and saved.")
    else:
        print("[!] FAISS index already exists. Skipping rebuild.")


    # Check if BM25 exists
    if not os.path.exists(BM25_PATH):
        print("[+] Building BM25 index...")
              
        # Remove punctuation and tokenize
        tokenized_corpus = [
            [token for token in word_tokenize(chunk["text"]) if token not in string.punctuation]
            for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized_corpus)
        with open(BM25_PATH, "wb") as f:
            pickle.dump(bm25, f)
    else:
        print("[✓] BM25 index already exists.")