from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

PROCESSED_MOVIES_FILE = PROCESSED_DIR / "movies_processed.csv"
FEEDBACK_FILE = PROCESSED_DIR / "user_feedback.json"
TRAINING_STATUS_FILE = ARTIFACTS_DIR / "training_status.json"
MODEL_BUNDLE_FILE = ARTIFACTS_DIR / "movie_recommender.joblib"
MODEL_EMBEDDINGS_FILE = ARTIFACTS_DIR / "movie_embeddings.npy"
MODEL_CONFIG_FILE = ARTIFACTS_DIR / "model_config.json"
AUTOENCODER_MODEL_FILE = ARTIFACTS_DIR / "movie_autoencoder.keras"
ENCODER_MODEL_FILE = ARTIFACTS_DIR / "movie_encoder.keras"
TRAINING_REPORT_FILE = ARTIFACTS_DIR / "training_report.json"
VECTOR_INDEX_FILE = ARTIFACTS_DIR / "movie_vectors.faiss"


def ensure_directories() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
