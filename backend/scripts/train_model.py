from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.model_training_service import ModelTrainingService


def main() -> None:
    service = ModelTrainingService()
    result = service.train()
    print("Treino concluido")
    print(result)


if __name__ == "__main__":
    main()
