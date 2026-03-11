import json

from fastapi import APIRouter, HTTPException

from app.models.schemas import FeedbackRequest, FeedbackResetRequest
from app.services.paths import MODEL_CONFIG_FILE, TRAINING_REPORT_FILE
from app.services.dataset_service import DatasetService
from app.services.feedback_service import FeedbackService
from app.services.model_training_service import ModelTrainingService
from app.services.training_runtime_service import TrainingRuntimeService
from app.services.training_status_service import TrainingStatusService


router = APIRouter(tags=["training"])
dataset_service = DatasetService()
training_service = ModelTrainingService()
status_service = TrainingStatusService()
feedback_service = FeedbackService()
training_runtime_service = TrainingRuntimeService()


@router.post("/dataset/download")
def download_dataset():
    try:
        return dataset_service.download_latest()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/dataset/status")
def dataset_status():
    return dataset_service.dataset_status()


@router.post("/train")
def train_model():
    try:
        return training_runtime_service.start_training()
    except (FileNotFoundError, KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/training/status")
def training_status():
    payload = status_service.read()
    payload["is_running"] = training_runtime_service.is_running()
    return payload


@router.get("/training/config")
def training_config():
    if not MODEL_CONFIG_FILE.exists():
        raise HTTPException(status_code=404, detail="Configuracao do modelo ainda nao foi gerada.")
    return json.loads(MODEL_CONFIG_FILE.read_text(encoding="utf-8"))


@router.get("/training/report")
def training_report():
    if TRAINING_REPORT_FILE.exists():
        return json.loads(TRAINING_REPORT_FILE.read_text(encoding="utf-8"))
    if not MODEL_CONFIG_FILE.exists():
        raise HTTPException(status_code=404, detail="Relatorio do modelo ainda nao foi gerado.")
    config = json.loads(MODEL_CONFIG_FILE.read_text(encoding="utf-8"))
    return {
        "training_summary": config.get("training_summary", {}),
        "training_report": config.get("training_report", {}),
        "hyperparameters": config.get("hyperparameters", {}),
        "text_feature_config": config.get("text_feature_config", {}),
        "numeric_feature_weights": config.get("numeric_feature_weights", {}),
    }


@router.post("/feedback")
def save_feedback(payload: FeedbackRequest):
    return feedback_service.append(
        user_id=payload.user_id,
        movie_id=payload.movie_id,
        action=payload.action,
    )


@router.post("/feedback/clear")
def clear_feedback(payload: FeedbackResetRequest):
    return feedback_service.clear_user(payload.user_id)
