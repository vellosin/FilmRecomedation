import json
from datetime import datetime, timezone
from json import JSONDecodeError

from .paths import TRAINING_STATUS_FILE, ensure_directories


DEFAULT_STATUS = {
    "stage": "idle",
    "message": "Aguardando download e treino.",
    "updated_at": None,
}


class TrainingStatusService:
    def __init__(self) -> None:
        ensure_directories()

    def read(self) -> dict:
        if not TRAINING_STATUS_FILE.exists():
            self.write(DEFAULT_STATUS)
        try:
            return json.loads(TRAINING_STATUS_FILE.read_text(encoding="utf-8"))
        except JSONDecodeError:
            return DEFAULT_STATUS.copy()

    def write(self, payload: dict) -> dict:
        payload = {
            **payload,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temp_file = TRAINING_STATUS_FILE.with_suffix(".tmp")
        temp_file.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        temp_file.replace(TRAINING_STATUS_FILE)
        return payload
