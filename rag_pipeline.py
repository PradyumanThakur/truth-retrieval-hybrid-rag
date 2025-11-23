import time 
import re
import json

from utlis.reranker import rerank_results
from utlis.preprocessing import normalize
from utlis.retrieval import get_faiss_ranks, get_bm25_ranks, EnsembleRetriever
from utlis.llm_verifier import generate_stream
from utlis.logger import log_result
from utlis.postprocessing import extract_ref, explanation_processing

# ---------- Singleton model access ----------
# Module-level globals
_models = None
_chunks = None

def get_models():
    """Lazy-load models singleton for retrieval or reranker."""
    global _models, _chunks
    if _models is None:
        from models.load_models import load_all_models
        _models = load_all_models()
        _chunks = _models["verses"]
    return _models, _chunks


def main(claim: str, top_n: int = 10):
    models, chunks = get_models()

    query = normalize(claim)
    faiss_ranks = get_faiss_ranks(query, top_k=100, sem_threshold=0.80)
    bm25_ranks = get_bm25_ranks(query, top_k=100)

    # Hybrid RAG Engine
    print("[+] Retrieving Top relevant verses using Hybrid RAG...")
    start = time.time()
    retriever = EnsembleRetriever([faiss_ranks, bm25_ranks])
    verses = retriever.retrieve(top_n=100)

    top_verses = rerank_results(query, verses, top_k=top_n)

    # Use generator to get stream and track full output
    print("[+] Verifying claim against verses using LLM...")

    # Streaming tokens
    stream = generate_stream(claim, top_verses)
    full_text = ""
    for tok in stream:
        print(tok, end="", flush=True)
        full_text += tok
    end = time.time()
    print(f"\nTotal time: {end - start:.2f} seconds")

    # Parse final JSON
    try:
        match = re.search(r"\{.*\}", full_text, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
        else:
            result = {"claim": claim, "label": "ERROR", "reference": "", "explanation": "No JSON found."}
    except Exception as e:
        result = {"claim": claim, "label": "ERROR", "reference": "", "explanation": f"Parse error: {e}"}

    # Log full result (with claim included)
    result_with_claim = {
        "claim": claim,
        "label": result.get("label", "ERROR"),
        "reference": result.get("reference", []),
        "explanation": result.get("explanation", ""),
        "time_taken": round(end - start, 2),
    }

    # postprocess reference and explanation
    predicted_json = extract_ref(result_with_claim, top_verses, chunks)
    final_explanation = explanation_processing(predicted_json)
    predicted_json["explanation"] = final_explanation
    log_result(predicted_json)
    
    return predicted_json
