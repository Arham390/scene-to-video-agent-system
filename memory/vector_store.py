"""
FAISS-backed vector store using HuggingFace sentence-transformers embeddings.
"""

import os
import json
import numpy as np
from config import config


class VectorStore:
    def __init__(self):
        self._index = None
        self._documents = []
        self._metadata = []
        self._dim = 384  

    def _get_embedding(self, text: str) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            return model.encode([text], normalize_embeddings=True)[0].astype("float32")
        except ImportError:
            return np.zeros(self._dim, dtype="float32")

    def add_document(self, text: str, metadata: dict = None):
        import faiss
        if self._index is None:
            self._index = faiss.IndexFlatL2(self._dim)
        vec = self._get_embedding(text).reshape(1, -1)
        self._index.add(vec)
        self._documents.append(text)
        self._metadata.append(metadata or {})

    def search(self, query: str, k: int = 3) -> list[dict]:
        import faiss
        if self._index is None or self._index.ntotal == 0:
            return []
        vec = self._get_embedding(query).reshape(1, -1)
        k = min(k, self._index.ntotal)
        distances, indices = self._index.search(vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append({
                    "text": self._documents[idx],
                    "metadata": self._metadata[idx],
                    "distance": float(dist),
                })
        return results

    def persist(self):
        import faiss
        os.makedirs(config.faiss_index_path, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, os.path.join(config.faiss_index_path, "index.bin"))
        with open(os.path.join(config.faiss_index_path, "docs.json"), "w") as f:
            json.dump({"documents": self._documents, "metadata": self._metadata}, f, indent=2)

    def load(self):
        import faiss
        index_path = os.path.join(config.faiss_index_path, "index.bin")
        docs_path = os.path.join(config.faiss_index_path, "docs.json")
        if os.path.exists(index_path):
            self._index = faiss.read_index(index_path)
        if os.path.exists(docs_path):
            with open(docs_path) as f:
                data = json.load(f)
                self._documents = data.get("documents", [])
                self._metadata = data.get("metadata", [])
