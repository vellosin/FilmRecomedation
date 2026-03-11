import faiss
import numpy as np

from .paths import MODEL_EMBEDDINGS_FILE, VECTOR_INDEX_FILE


class VectorIndexService:
    def save(self, embeddings: np.ndarray) -> dict:
        vectors = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(VECTOR_INDEX_FILE))
        np.save(MODEL_EMBEDDINGS_FILE, vectors)
        return {
            "index_type": "faiss.IndexFlatIP",
            "vectors": int(vectors.shape[0]),
            "dimensions": int(vectors.shape[1]),
        }

    def load(self):
        index = faiss.read_index(str(VECTOR_INDEX_FILE))
        embeddings = np.load(MODEL_EMBEDDINGS_FILE).astype("float32")
        return index, embeddings

    def search(self, query_vector: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray]:
        index, _ = self.load()
        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query)
        scores, indices = index.search(query, top_n)
        return scores[0], indices[0]