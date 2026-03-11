import json
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, normalize

from .paths import (
    AUTOENCODER_MODEL_FILE,
    FEEDBACK_FILE,
    ENCODER_MODEL_FILE,
    MODEL_BUNDLE_FILE,
    MODEL_CONFIG_FILE,
    PROCESSED_MOVIES_FILE,
    TRAINING_REPORT_FILE,
    ensure_directories,
)
from .preprocessing_service import PreprocessingService
from .training_status_service import TrainingStatusService
from .vector_index_service import VectorIndexService


TEXT_FEATURE_CONFIG = {
    "title": {"max_features": 2500, "ngram_range": (1, 2), "weight": 1.35},
    "overview": {"max_features": 3200, "ngram_range": (1, 2), "weight": 0.95},
    "genres_text": {"max_features": 300, "ngram_range": (1, 2), "weight": 1.60},
    "cast_text": {"max_features": 2500, "ngram_range": (1, 2), "weight": 1.75},
    "director_text": {"max_features": 500, "ngram_range": (1, 2), "weight": 1.05},
    "writer_text": {"max_features": 700, "ngram_range": (1, 2), "weight": 0.95},
    "producer_text": {"max_features": 600, "ngram_range": (1, 2), "weight": 0.60},
    "review_text": {"max_features": 1200, "ngram_range": (1, 1), "weight": 0.25},
}

NUMERIC_FEATURE_WEIGHTS = {
    "vote_average": 1.35,
    "popularity": 1.15,
    "runtime": 0.35,
    "release_year": 0.45,
    "review_score": 1.10,
    "review_count": 0.80,
}

MODEL_HYPERPARAMETERS = {
    "svd_components": 256,
    "encoder_layers": (192, 96),
    "embedding_dim": 64,
    "decoder_layers": (96, 192),
    "activation": "relu",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "loss": "mse",
    "batch_size": 128,
    "epochs": 300,
    "validation_split": 0.1,
    "early_stopping_patience": 18,
    "reduce_lr_patience": 6,
    "reduce_lr_factor": 0.5,
    "min_learning_rate": 1e-5,
    "random_state": 42,
}


class TrainingReporter(tf.keras.callbacks.Callback):
    def __init__(self, status_service: TrainingStatusService, report_context: dict, max_epochs: int) -> None:
        super().__init__()
        self.status_service = status_service
        self.report_context = report_context
        self.max_epochs = max_epochs
        self.report = {
            "training_summary": {
                "movies": report_context["movies"],
                "autoencoder_loss": None,
                "epochs_ran": 0,
                "best_validation_score": None,
                "final_validation_score": None,
                "estimated_precision": None,
                "loss_curve_length": 0,
            },
            "training_report": {
                "loss_curve": [],
                "validation_scores": [],
                "warnings": [],
            },
            "ranking_metrics": {},
            "hyperparameters": report_context["hyperparameters"],
            "text_feature_config": report_context["text_feature_config"],
            "numeric_feature_weights": report_context["numeric_feature_weights"],
            "vector_index": None,
        }

    def on_train_begin(self, logs=None):
        self._persist()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = float(logs.get("loss", 0.0))
        val_loss = logs.get("val_loss")
        val_score = None if val_loss is None else float(1.0 / (1.0 + val_loss))

        self.report["training_report"]["loss_curve"].append(loss)
        self.report["training_report"]["validation_scores"].append(val_score)
        self.report["training_summary"]["autoencoder_loss"] = loss
        self.report["training_summary"]["epochs_ran"] = epoch + 1
        self.report["training_summary"]["loss_curve_length"] = len(self.report["training_report"]["loss_curve"])
        self.report["training_summary"]["final_validation_score"] = val_score

        valid_scores = [value for value in self.report["training_report"]["validation_scores"] if value is not None]
        self.report["training_summary"]["best_validation_score"] = max(valid_scores) if valid_scores else None
        self.report["training_summary"]["estimated_precision"] = val_score

        progress = int(((epoch + 1) / self.max_epochs) * 100)
        self.status_service.write(
            {
                "stage": "training",
                "message": f"Treinando autoencoder TensorFlow: epoca {epoch + 1}/{self.max_epochs}",
                "progress": min(progress, 99),
                "loss": loss,
                "val_loss": None if val_loss is None else float(val_loss),
            }
        )
        self._persist()

    def on_train_end(self, logs=None):
        self._persist()

    def finalize(self, warnings_list: list[str], vector_index: dict, ranking_metrics: dict) -> dict:
        self.report["training_report"]["warnings"] = warnings_list
        self.report["vector_index"] = vector_index
        self.report["ranking_metrics"] = ranking_metrics
        self._persist()
        return deepcopy(self.report)

    def _persist(self) -> None:
        TRAINING_REPORT_FILE.write_text(
            json.dumps(self.report, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


class ModelTrainingService:
    def __init__(self) -> None:
        ensure_directories()
        self.preprocessing_service = PreprocessingService()
        self.status_service = TrainingStatusService()
        self.vector_index_service = VectorIndexService()

    def train(self) -> dict:
        tf.keras.backend.clear_session()
        tf.random.set_seed(MODEL_HYPERPARAMETERS["random_state"])
        np.random.seed(MODEL_HYPERPARAMETERS["random_state"])

        self.status_service.write(
            {"stage": "training", "message": "Preparando features e treinando o recomendador."}
        )

        movies_df = self.preprocessing_service.build_training_dataframe()

        if movies_df.empty:
            raise RuntimeError("Nao ha filmes processados para treinar o modelo.")

        text_vectorizers, reduced_features, preprocess_bundle = self._prepare_features(movies_df)
        reporter = TrainingReporter(
            status_service=self.status_service,
            report_context={
                "movies": int(len(movies_df)),
                "hyperparameters": {
                    **MODEL_HYPERPARAMETERS,
                    "effective_svd_components": int(reduced_features.shape[1]),
                    "bottleneck_dimensions": int(MODEL_HYPERPARAMETERS["embedding_dim"]),
                },
                "text_feature_config": TEXT_FEATURE_CONFIG,
                "numeric_feature_weights": NUMERIC_FEATURE_WEIGHTS,
            },
            max_epochs=MODEL_HYPERPARAMETERS["epochs"],
        )

        self.status_service.write(
            {"stage": "training", "message": "Treinando autoencoder com tf.keras e callbacks por epoca."}
        )

        autoencoder, encoder = self._build_models(reduced_features.shape[1])
        optimizer = tf.keras.optimizers.Adam(learning_rate=MODEL_HYPERPARAMETERS["learning_rate"])
        autoencoder.compile(optimizer=optimizer, loss=MODEL_HYPERPARAMETERS["loss"])

        history = autoencoder.fit(
            reduced_features,
            reduced_features,
            epochs=MODEL_HYPERPARAMETERS["epochs"],
            batch_size=MODEL_HYPERPARAMETERS["batch_size"],
            shuffle=True,
            validation_split=MODEL_HYPERPARAMETERS["validation_split"],
            verbose=0,
            callbacks=[
                reporter,
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=MODEL_HYPERPARAMETERS["early_stopping_patience"],
                    restore_best_weights=True,
                    verbose=0,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=MODEL_HYPERPARAMETERS["reduce_lr_factor"],
                    patience=MODEL_HYPERPARAMETERS["reduce_lr_patience"],
                    min_lr=MODEL_HYPERPARAMETERS["min_learning_rate"],
                    verbose=0,
                ),
            ],
        )

        embeddings = encoder.predict(reduced_features, batch_size=MODEL_HYPERPARAMETERS["batch_size"], verbose=0)
        embeddings = normalize(np.asarray(embeddings, dtype="float32"))

        autoencoder.save(AUTOENCODER_MODEL_FILE, overwrite=True)
        encoder.save(ENCODER_MODEL_FILE, overwrite=True)
        vector_index_metadata = self.vector_index_service.save(embeddings)

        warning_messages = []
        if len(history.history.get("loss", [])) >= MODEL_HYPERPARAMETERS["epochs"]:
            warning_messages.append("Treino atingiu o limite maximo de epocas configurado.")

        ranking_metrics = self._evaluate_ranking_metrics(
            embeddings=embeddings,
            movie_ids=movies_df["movie_id"].astype(int).tolist(),
        )

        metadata = reporter.finalize(
            warnings_list=warning_messages,
            vector_index=vector_index_metadata,
            ranking_metrics=ranking_metrics,
        )

        joblib.dump(
            {
                "text_vectorizers": text_vectorizers,
                **preprocess_bundle,
                "movie_ids": movies_df["movie_id"].astype(int).tolist(),
                "metadata": metadata,
            },
            MODEL_BUNDLE_FILE,
        )
        MODEL_CONFIG_FILE.write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        final_loss = float(history.history["loss"][-1])
        val_losses = history.history.get("val_loss", [])
        best_validation_score = max((1.0 / (1.0 + value) for value in val_losses), default=None)

        payload = {
            "stage": "trained",
            "message": "Modelo TensorFlow treinado e indice vetorial gerado com sucesso.",
            "movies": int(len(movies_df)),
            "embedding_dimensions": int(embeddings.shape[1]),
            "loss": final_loss,
            "epochs_ran": int(len(history.history["loss"])),
            "best_validation_score": best_validation_score,
            "final_validation_score": float(1.0 / (1.0 + val_losses[-1])) if val_losses else None,
            "progress": 100,
        }
        self.status_service.write(payload)
        return payload

    def _evaluate_ranking_metrics(self, embeddings: np.ndarray, movie_ids: list[int]) -> dict:
        if not FEEDBACK_FILE.exists():
            return {
                "available": False,
                "reason": "Ainda nao ha feedback persistido para avaliar ranking offline.",
            }

        feedback_entries = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        if not feedback_entries:
            return {
                "available": False,
                "reason": "O arquivo de feedback esta vazio.",
            }

        index_by_movie_id = {movie_id: index for index, movie_id in enumerate(movie_ids)}
        users = self._build_feedback_profiles(feedback_entries, index_by_movie_id)
        if not users:
            return {
                "available": False,
                "reason": "Nao ha usuarios com feedback suficiente para avaliacao offline.",
            }

        cutoffs = (5, 10)
        precision_totals = {cutoff: [] for cutoff in cutoffs}
        recall_totals = {cutoff: [] for cutoff in cutoffs}
        ndcg_totals = {cutoff: [] for cutoff in cutoffs}
        reciprocal_ranks = []

        normalized_embeddings = np.asarray(embeddings, dtype="float32")

        for profile in users:
            profile_vector = self._build_profile_vector(
                embeddings=normalized_embeddings,
                likes=profile["likes"],
                favorites=profile["favorites"],
                dislikes=profile["dislikes"],
                index_by_movie_id=index_by_movie_id,
            )
            if profile_vector is None:
                continue

            held_out = profile["held_out"]
            ranked_ids = self._rank_candidates(
                profile_vector=profile_vector,
                embeddings=normalized_embeddings,
                movie_ids=movie_ids,
                excluded_ids=profile["excluded_ids"],
            )
            if not ranked_ids:
                continue

            try:
                rank_index = ranked_ids.index(held_out)
            except ValueError:
                rank_index = None

            reciprocal_ranks.append(0.0 if rank_index is None else 1.0 / (rank_index + 1))

            for cutoff in cutoffs:
                hit = 1.0 if rank_index is not None and rank_index < cutoff else 0.0
                precision_totals[cutoff].append(hit / cutoff)
                recall_totals[cutoff].append(hit)
                ndcg_totals[cutoff].append(
                    0.0 if rank_index is None or rank_index >= cutoff else 1.0 / np.log2(rank_index + 2)
                )

        evaluated_users = len(reciprocal_ranks)
        if not evaluated_users:
            return {
                "available": False,
                "reason": "Nao foi possivel montar perfis validos para avaliacao offline.",
            }

        metrics = {
            "available": True,
            "evaluated_users": evaluated_users,
            "mrr": float(np.mean(reciprocal_ranks)),
        }
        for cutoff in cutoffs:
            metrics[f"precision_at_{cutoff}"] = float(np.mean(precision_totals[cutoff]))
            metrics[f"recall_at_{cutoff}"] = float(np.mean(recall_totals[cutoff]))
            metrics[f"ndcg_at_{cutoff}"] = float(np.mean(ndcg_totals[cutoff]))
        return metrics

    def _build_feedback_profiles(self, feedback_entries: list[dict], index_by_movie_id: dict[int, int]) -> list[dict]:
        latest_actions_by_user: dict[str, dict[int, dict]] = {}
        for entry in sorted(feedback_entries, key=lambda item: item.get("created_at", "")):
            user_id = str(entry.get("user_id", "local-user"))
            movie_id = int(entry.get("movie_id"))
            if movie_id not in index_by_movie_id:
                continue
            latest_actions_by_user.setdefault(user_id, {})[movie_id] = entry

        profiles = []
        for user_id, actions_by_movie in latest_actions_by_user.items():
            latest_actions = sorted(actions_by_movie.values(), key=lambda item: item.get("created_at", ""))
            positives = [item for item in latest_actions if item.get("action") in {"like", "favorite"}]
            dislikes = [int(item["movie_id"]) for item in latest_actions if item.get("action") == "dislike"]
            if len(positives) < 4:
                continue

            held_out_entry = positives[-1]
            history_entries = positives[:-1]
            likes = [int(item["movie_id"]) for item in history_entries if item.get("action") == "like"]
            favorites = [int(item["movie_id"]) for item in history_entries if item.get("action") == "favorite"]
            if len(set([*likes, *favorites])) < 3:
                continue

            profiles.append(
                {
                    "user_id": user_id,
                    "likes": likes,
                    "favorites": favorites,
                    "dislikes": dislikes,
                    "held_out": int(held_out_entry["movie_id"]),
                    "excluded_ids": set([*likes, *favorites, *dislikes]),
                }
            )

        return profiles

    def _build_profile_vector(
        self,
        embeddings: np.ndarray,
        likes: list[int],
        favorites: list[int],
        dislikes: list[int],
        index_by_movie_id: dict[int, int],
    ) -> np.ndarray | None:
        positive_vectors = []
        negative_vectors = []

        for movie_id in likes:
            index = index_by_movie_id.get(movie_id)
            if index is not None:
                positive_vectors.append(embeddings[index])

        for movie_id in favorites:
            index = index_by_movie_id.get(movie_id)
            if index is not None:
                positive_vectors.append(embeddings[index] * 1.5)

        for movie_id in dislikes:
            index = index_by_movie_id.get(movie_id)
            if index is not None:
                negative_vectors.append(embeddings[index] * 0.7)

        if not positive_vectors:
            return None

        profile = np.mean(np.vstack(positive_vectors), axis=0)
        if negative_vectors:
            profile = profile - np.mean(np.vstack(negative_vectors), axis=0)

        norm = np.linalg.norm(profile)
        if norm <= 0 or not np.isfinite(norm):
            return None
        return np.asarray(profile / norm, dtype="float32")

    def _rank_candidates(
        self,
        profile_vector: np.ndarray,
        embeddings: np.ndarray,
        movie_ids: list[int],
        excluded_ids: set[int],
    ) -> list[int]:
        scores = embeddings @ profile_vector
        ranked_indices = np.argsort(scores)[::-1]
        ranked_ids = []
        for candidate_index in ranked_indices:
            movie_id = int(movie_ids[int(candidate_index)])
            if movie_id in excluded_ids:
                continue
            ranked_ids.append(movie_id)
        return ranked_ids

    def _prepare_features(self, movies_df: pd.DataFrame) -> tuple[dict, np.ndarray, dict]:
        self.status_service.write(
            {"stage": "training", "message": "Vetorizando blocos de texto e atributos numericos ponderados."}
        )

        text_vectorizers = {}
        text_matrices = []
        for column, config in TEXT_FEATURE_CONFIG.items():
            vectorizer = TfidfVectorizer(
                max_features=config["max_features"],
                stop_words="english",
                ngram_range=config["ngram_range"],
            )
            matrix = vectorizer.fit_transform(movies_df[column].fillna(""))
            text_vectorizers[column] = vectorizer
            text_matrices.append(matrix.multiply(config["weight"]))

        numeric_columns = list(NUMERIC_FEATURE_WEIGHTS)
        numeric_frame = movies_df[numeric_columns].fillna(0).copy()
        for column, weight in NUMERIC_FEATURE_WEIGHTS.items():
            numeric_frame[column] = numeric_frame[column].astype(float) * weight

        numeric_matrix = csr_matrix(numeric_frame.to_numpy(dtype=float))
        numeric_scaler = MaxAbsScaler()
        numeric_matrix = numeric_scaler.fit_transform(numeric_matrix)
        features = hstack([*text_matrices, numeric_matrix]).tocsr()

        self.status_service.write(
            {"stage": "training", "message": "Aplicando reducao inicial de dimensionalidade antes da rede neural TensorFlow."}
        )

        max_allowed_components = min(features.shape[1] - 1, features.shape[0] - 1)
        svd_components = min(MODEL_HYPERPARAMETERS["svd_components"], max_allowed_components)
        if svd_components < 2:
            raise RuntimeError("A base ficou pequena demais para treinar a rede neural com estabilidade.")

        reducer = TruncatedSVD(n_components=svd_components, random_state=MODEL_HYPERPARAMETERS["random_state"])
        reduced_features = reducer.fit_transform(features)

        dense_scaler = StandardScaler()
        reduced_features = dense_scaler.fit_transform(reduced_features).astype("float32")
        return text_vectorizers, reduced_features, {
            "numeric_scaler": numeric_scaler,
            "dense_scaler": dense_scaler,
            "reducer": reducer,
            "numeric_columns": numeric_columns,
        }

    def _build_models(self, input_dim: int):
        inputs = tf.keras.Input(shape=(input_dim,), name="movie_features")
        current = inputs

        for units in MODEL_HYPERPARAMETERS["encoder_layers"]:
            current = tf.keras.layers.Dense(units, activation=MODEL_HYPERPARAMETERS["activation"])(current)

        bottleneck = tf.keras.layers.Dense(
            MODEL_HYPERPARAMETERS["embedding_dim"],
            activation="linear",
            name="movie_embedding",
        )(current)
        current = bottleneck

        for units in MODEL_HYPERPARAMETERS["decoder_layers"]:
            current = tf.keras.layers.Dense(units, activation=MODEL_HYPERPARAMETERS["activation"])(current)

        outputs = tf.keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(current)
        autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs, name="movie_autoencoder")
        encoder = tf.keras.Model(inputs=inputs, outputs=bottleneck, name="movie_encoder")
        return autoencoder, encoder
