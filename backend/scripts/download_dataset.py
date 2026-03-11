from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.dataset_service import DatasetService


def main() -> None:
    service = DatasetService()
    result = service.download_latest()
    print("Dataset baixado com sucesso")
    print(result)


if __name__ == "__main__":
    main()
