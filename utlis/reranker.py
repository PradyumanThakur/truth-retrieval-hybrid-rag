import numpy as np
from models.load_models import models

# Load model
rank_model = models["cross_encoder_model"]

def rerank_results(query: str, retrieved_verses: list, top_k: int = 10):
    """
    Reranks retrieved verses based on semantic similarity to the query.
    
    Args:
        query: User claim/query.
        retrieved_chunks: List of dicts with "text" key (output of hybrid retrieval).
        top_k: Number of top verses to return after reranking.
    
    Returns:
        Reranked list of verse dicts.
    """
    if not retrieved_verses:
        return []

    # Prepare query-verse pairs for CrossEncoder
    pair_inputs = [(query, verse["text"]) for verse in retrieved_verses]

    # Get similarity scores
    raw_scores = rank_model.predict(pair_inputs)

    # Apply softmax normalization
    exp_scores = np.exp(raw_scores - np.max(raw_scores))  # subtract max for numerical stability
    softmax_scores = exp_scores / np.sum(exp_scores)

    # Combine verses with their rerank scores and sort them
    reranked = sorted(zip(retrieved_verses, softmax_scores), key=lambda x: x[1], reverse=True)

    # Add 'rerank_score' field to each verse dict
    for verse, score in reranked:
        verse["rerank_score"] = round(float(score), 4)  # FastAPI requires float, not np.float32

    return [item for item, _ in reranked[:top_k]]