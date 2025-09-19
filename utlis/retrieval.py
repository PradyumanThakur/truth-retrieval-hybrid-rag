from collections import defaultdict
from nltk.tokenize import word_tokenize

from models.load_models import load_all_models


# ---------- Singleton models ----------
_models = None
_chunks = None

def get_models():
    """Lazy-load models singleton for retrieval."""
    global _models, _chunks
    if _models is None:
        _models = load_all_models()
        _chunks = _models["verses"]
    return _models, _chunks

# --------------- Hybrid Retrival Engine - RAG (Retrival-Augmented Generation) ---------------- #

# ---------- 1. Dense (FAISS) Retrieval ----------
def get_faiss_ranks(query: str, top_k: int=10, sem_threshold=0.8):
    models, chunks = get_models()
    embedder = models["sentence_model"]
    faiss_index = models["faiss_index"]

    query_emb = embedder.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(query_emb, top_k)
    
    filtered = [(idx, score) for idx, score in zip(I[0], D[0]) if score > sem_threshold]
    if not filtered:
        return None  # FAISS failed → return None to halt hybrid

    return [int(idx) for idx, _ in filtered]

# ---------- 2. Sparse (BM25) Retrieval ----------
def get_bm25_ranks(query: str, top_k: int=10):
    models, chunks = get_models()
    bm25 = models["bm25_model"]

    query_tokens = word_tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Filter for scores > 0 only
    scored_results = [(i, s) for i, s in enumerate(scores) if s > 0]
    if not scored_results:
        return [] # No meaningful BM25 results

    # Sort by score
    ranked = sorted(enumerate(scored_results), key=lambda x: x[1][1], reverse=True)[:top_k]
    return [int(s[0]) for _, s in ranked]

# ---------- Ensemble Retriever: FAISS + BM25 -> Reciprocal Rank Fusion (RRF) ---------- #
class EnsembleRetriever:
    def __init__(self, retrievers: list, k_rrf: int = 60):
        """
        Args:
            retrievers: list of retriever objects with .retrieve(query, top_k) → list[int]
            k: RRF constant (default = 60)
        """
        self.retrievers = retrievers
        self.k_rrf = k_rrf
        _, self.chunks = get_models()

    def _rrf(self, rank):
        return 1 / (self.k_rrf + rank)

    def retrieve(self, top_n: int = 10) -> list:
        rrf_scores = defaultdict(float)
        for rank_list in self.retrievers:
            for rank, doc_id in enumerate(rank_list):
                rrf_scores[doc_id] += self._rrf(rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for idx, score in ranked:
            doc = self.chunks[idx]
            results.append({
                "idx": idx,
                "id": doc["id"],
                "book": doc["meta"]["book"],
                "chapter": doc["meta"]["chapter"],
                "text": doc["text"],
                "rrf_score": score
            })
        return results

