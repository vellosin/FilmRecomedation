import joblib
import numpy as np
import pandas as pd

from .catalog_service import CatalogService
from .feedback_service import FeedbackService
from .paths import MODEL_BUNDLE_FILE, VECTOR_INDEX_FILE
from .vector_index_service import VectorIndexService


class RecommendationService:
    def __init__(self) -> None:
        self.catalog_service = CatalogService()
        self.feedback_service = FeedbackService()
        self.vector_index_service = VectorIndexService()

    def _load(self) -> tuple[dict, np.ndarray, pd.DataFrame, dict[int, int]]:
        if not MODEL_BUNDLE_FILE.exists() or not VECTOR_INDEX_FILE.exists():
            raise FileNotFoundError("Modelo nao treinado. Rode o treino antes de pedir recomendacoes.")

        bundle = joblib.load(MODEL_BUNDLE_FILE)
        _, embeddings = self.vector_index_service.load()
        catalog = self.catalog_service.load_catalog()
        index_by_movie_id = {
            int(movie_id): index for index, movie_id in enumerate(bundle["movie_ids"])
        }
        return bundle, embeddings, catalog, index_by_movie_id

    def recommend_by_movie(self, movie_id: int, top_n: int, excluded_movie_ids: list[int] | None = None) -> list[dict]:
        bundle, embeddings, catalog, index_by_movie_id = self._load()
        if movie_id not in index_by_movie_id:
            raise KeyError(f"Filme {movie_id} nao esta no modelo treinado.")

        excluded_ids = set(excluded_movie_ids or [])
        excluded_ids.add(movie_id)

        movie_index = index_by_movie_id[movie_id]
        scores, indices = self.vector_index_service.search(
            embeddings[movie_index],
            top_n=min(max(top_n + len(excluded_ids), top_n + 1), len(embeddings)),
        )

        results = []
        for score, candidate_index in zip(scores, indices):
            candidate_id = int(bundle["movie_ids"][candidate_index])
            if candidate_id in excluded_ids:
                continue
            movie = self.catalog_service.serialize_movie(
                catalog[catalog["movie_id"] == candidate_id].iloc[0].to_dict()
            )
            movie["score"] = float(score)
            results.append(movie)
            if len(results) >= top_n:
                break
        return results

    def recommend_by_profile(
        self,
        user_id: str,
        likes: list[int],
        dislikes: list[int],
        favorites: list[int],
        excluded_movie_ids: list[int] | None,
        top_n: int,
    ) -> list[dict]:
        bundle, embeddings, catalog, index_by_movie_id = self._load()

        merged_profile = self._merge_profile_feedback(
            user_id=user_id,
            likes=likes,
            dislikes=dislikes,
            favorites=favorites,
        )

        likes = merged_profile["likes"]
        dislikes = merged_profile["dislikes"]
        favorites = merged_profile["favorites"]
        action_weights = merged_profile["weights_by_movie"]

        positive_vectors = []
        negative_vectors = []
        excluded_ids = set(likes) | set(dislikes) | set(favorites) | set(excluded_movie_ids or [])
        positive_ids = list(dict.fromkeys([*likes, *favorites]))

        if len(positive_ids) < 3:
            raise ValueError("Curta ou favorite pelo menos 3 filmes antes de pedir recomendacoes. Se quiser, voce pode adicionar mais filmes para melhorar o perfil.")

        for movie_id in likes:
            index = index_by_movie_id.get(movie_id)
            if index is not None:
                positive_vectors.append(embeddings[index] * action_weights.get(movie_id, 1.0))

        for movie_id in favorites:
            index = index_by_movie_id.get(movie_id)
            if index is not None:
                positive_vectors.append(embeddings[index] * action_weights.get(movie_id, 1.5))

        for movie_id in dislikes:
            index = index_by_movie_id.get(movie_id)
            if index is not None:
                negative_vectors.append(embeddings[index] * action_weights.get(movie_id, 0.7))

        if not positive_vectors:
            raise ValueError("Envie ao menos um filme curtido ou favorito para montar o perfil.")

        profile = np.mean(np.vstack(positive_vectors), axis=0)
        if negative_vectors:
            profile = profile - np.mean(np.vstack(negative_vectors), axis=0)

        profile_norm = np.linalg.norm(profile)
        if profile_norm > 0 and np.isfinite(profile_norm):
            profile = profile / profile_norm
        else:
            profile = np.mean(np.vstack(positive_vectors), axis=0)
            fallback_norm = np.linalg.norm(profile)
            if fallback_norm > 0 and np.isfinite(fallback_norm):
                profile = profile / fallback_norm
            else:
                raise ValueError("Nao foi possivel montar um vetor valido para o perfil. Adicione mais filmes ao perfil.")

        scores, indices = self.vector_index_service.search(profile, top_n=len(embeddings))

        liked_cast_sets = self._collect_cast_sets(catalog, likes)
        favorite_cast_sets = self._collect_cast_sets(catalog, favorites)
        disliked_cast_sets = self._collect_cast_sets(catalog, dislikes)

        results = []
        for score, candidate_index in zip(scores, indices):
            candidate_id = int(bundle["movie_ids"][candidate_index])
            if candidate_id in excluded_ids:
                continue
            movie = self.catalog_service.serialize_movie(
                catalog[catalog["movie_id"] == candidate_id].iloc[0].to_dict()
            )
            final_score = self._apply_profile_weights(
                base_score=float(score),
                candidate_movie=movie,
                liked_cast_sets=liked_cast_sets,
                favorite_cast_sets=favorite_cast_sets,
                disliked_cast_sets=disliked_cast_sets,
            )
            if not np.isfinite(final_score):
                continue
            movie["score"] = float(final_score)
            results.append(movie)
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_n]

    def _merge_profile_feedback(
        self,
        user_id: str,
        likes: list[int],
        dislikes: list[int],
        favorites: list[int],
    ) -> dict:
        stored = self.feedback_service.load_user_state(user_id)
        merged_actions: dict[int, dict] = {}

        current_likes = [int(movie_id) for movie_id in likes]
        current_dislikes = [int(movie_id) for movie_id in dislikes]
        current_favorites = [int(movie_id) for movie_id in favorites]
        current_positive_ids = list(dict.fromkeys([*current_likes, *current_favorites]))

        # Quando o usuario envia um perfil positivo consistente na sessao atual,
        # ele deve prevalecer sobre a memoria historica para evitar contaminacao.
        should_prioritize_current_profile = len(current_positive_ids) >= 3

        if should_prioritize_current_profile:
            for movie_id in current_dislikes:
                merged_actions[movie_id] = {
                    "action": "dislike",
                    "weight": self._weight_for_action("dislike", 1.2),
                }
            for movie_id in current_likes:
                merged_actions[movie_id] = {
                    "action": "like",
                    "weight": self._weight_for_action("like", 1.2),
                }
            for movie_id in current_favorites:
                merged_actions[movie_id] = {
                    "action": "favorite",
                    "weight": self._weight_for_action("favorite", 1.25),
                }
        else:
            stored_actions = stored.get("actions", [])
            total_stored = max(len(stored_actions) - 1, 1)
            for index, entry in enumerate(stored_actions):
                movie_id = int(entry["movie_id"])
                action = str(entry.get("action", "like"))
                recency_boost = 1.0 + ((index / total_stored) * 0.35)
                merged_actions[movie_id] = {
                    "action": action,
                    "weight": self._weight_for_action(action, recency_boost),
                }

        for movie_id in current_dislikes:
            merged_actions[int(movie_id)] = {
                "action": "dislike",
                "weight": self._weight_for_action("dislike", 1.2),
            }
        for movie_id in current_likes:
            merged_actions[int(movie_id)] = {
                "action": "like",
                "weight": self._weight_for_action("like", 1.2),
            }
        for movie_id in current_favorites:
            merged_actions[int(movie_id)] = {
                "action": "favorite",
                "weight": self._weight_for_action("favorite", 1.25),
            }

        merged_likes = [movie_id for movie_id, meta in merged_actions.items() if meta["action"] == "like"]
        merged_favorites = [movie_id for movie_id, meta in merged_actions.items() if meta["action"] == "favorite"]
        merged_dislikes = [movie_id for movie_id, meta in merged_actions.items() if meta["action"] == "dislike"]

        return {
            "likes": merged_likes,
            "favorites": merged_favorites,
            "dislikes": merged_dislikes,
            "weights_by_movie": {movie_id: meta["weight"] for movie_id, meta in merged_actions.items()},
        }

    def _weight_for_action(self, action: str, recency_boost: float) -> float:
        base_weights = {
            "like": 1.0,
            "favorite": 1.6,
            "dislike": 0.8,
        }
        return base_weights.get(action, 1.0) * recency_boost

    def _collect_cast_sets(self, catalog: pd.DataFrame, movie_ids: list[int]) -> list[set[str]]:
        cast_sets = []
        for movie_id in movie_ids:
            row = catalog[catalog["movie_id"] == movie_id]
            if row.empty:
                continue
            cast_sets.append(self._tokenize_cast(row.iloc[0].get("cast_text", "")))
        return cast_sets

    def _tokenize_cast(self, cast_text: str) -> set[str]:
        return {item.strip().lower() for item in str(cast_text).split() if item.strip()}

    def _apply_profile_weights(
        self,
        base_score: float,
        candidate_movie: dict,
        liked_cast_sets: list[set[str]],
        favorite_cast_sets: list[set[str]],
        disliked_cast_sets: list[set[str]],
    ) -> float:
        candidate_cast = self._tokenize_cast(candidate_movie.get("cast_text", ""))
        liked_overlap = self._average_overlap(candidate_cast, liked_cast_sets)
        favorite_overlap = self._average_overlap(candidate_cast, favorite_cast_sets)
        disliked_overlap = self._average_overlap(candidate_cast, disliked_cast_sets)

        # Favoritos pesam mais, descartados penalizam o score final.
        final_score = (
            base_score
            + (liked_overlap * 0.18)
            + (favorite_overlap * 0.32)
            - (disliked_overlap * 0.28)
        )
        return float(final_score) if np.isfinite(final_score) else float(base_score if np.isfinite(base_score) else 0.0)

    def _average_overlap(self, candidate_cast: set[str], reference_sets: list[set[str]]) -> float:
        if not candidate_cast or not reference_sets:
            return 0.0

        overlaps = []
        for reference_set in reference_sets:
            if not reference_set:
                continue
            union = candidate_cast | reference_set
            if not union:
                continue
            overlaps.append(len(candidate_cast & reference_set) / len(union))
        return float(np.mean(overlaps)) if overlaps else 0.0
