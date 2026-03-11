import joblib
import numpy as np

from .paths import MODEL_BUNDLE_FILE, MODEL_EMBEDDINGS_FILE


class VectorStoreService:
    def load(self) -> tuple[dict, np.ndarray]:
        if not MODEL_BUNDLE_FILE.exists() or not MODEL_EMBEDDINGS_FILE.exists():
            raise FileNotFoundError("Base vetorial ainda nao foi gerada. Rode o treino do modelo primeiro.")

        bundle = joblib.load(MODEL_BUNDLE_FILE)
        embeddings = np.load(MODEL_EMBEDDINGS_FILE)
        return bundle, embeddings

    def neighbors_for_index(self, item_index: int, top_n: int) -> tuple[np.ndarray, np.ndarray, dict]:
        bundle, embeddings = self.load()
        distances, indices = bundle["neighbors"].kneighbors(
            embeddings[item_index].reshape(1, -1),
            n_neighbors=min(top_n + 1, len(embeddings)),
        )
        return distances[0], indices[0], bundle

    def rank_by_query_vector(self, query_vector: np.ndarray) -> tuple[np.ndarray, dict]:
        bundle, embeddings = self.load()
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        similarities = embeddings @ query_vector
        ranking = np.argsort(similarities)[::-1]
        return ranking, {**bundle, "similarities": similarities}