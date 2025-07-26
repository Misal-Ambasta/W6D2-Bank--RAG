import hashlib
import os
import pickle
from typing import List

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_CACHE_DIR = "embedding_cache"
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "retrieval.md")

os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def embed_texts(texts: List[str]) -> List[List[float]]:
    embeddings = []
    cache_hits = 0
    cache_misses = 0
    for text in texts:
        h = _content_hash(text)
        cache_path = os.path.join(EMBEDDING_CACHE_DIR, f"{h}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                emb = pickle.load(f)
            cache_hits += 1
        else:
            emb = model.encode([text], convert_to_tensor=False)[0]
            with open(cache_path, "wb") as f:
                pickle.dump(emb, f)
            cache_misses += 1
        embeddings.append(emb)
    _log_embedding_run(len(texts), cache_hits, cache_misses)
    return embeddings

def _log_embedding_run(num_texts, hits, misses):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\nEmbedding run: {num_texts} texts | Cache hits: {hits} | Cache misses: {misses}\n")
