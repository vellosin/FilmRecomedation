from __future__ import annotations

from threading import Lock, Thread

from .model_training_service import ModelTrainingService
from .training_status_service import TrainingStatusService


class TrainingRuntimeService:
    def __init__(self) -> None:
        self._lock = Lock()
        self._thread: Thread | None = None
        self._status_service = TrainingStatusService()

    def start_training(self) -> dict:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {
                    "accepted": False,
                    "stage": "training",
                    "message": "Ja existe um treino em execucao.",
                }

            self._status_service.write(
                {"stage": "queued", "message": "Treino colocado em fila e prestes a iniciar."}
            )
            self._thread = Thread(target=self._run_training, daemon=True)
            self._thread.start()

        return {
            "accepted": True,
            "stage": "queued",
            "message": "Treino iniciado em segundo plano.",
        }

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def _run_training(self) -> None:
        try:
            ModelTrainingService().train()
        except Exception as exc:
            self._status_service.write(
                {"stage": "failed", "message": f"Falha durante o treino: {exc}"}
            )
