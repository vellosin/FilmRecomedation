import shutil
from pathlib import Path

from .paths import RAW_DIR, ensure_directories


KAGGLE_DATASET = "rishabhkumar2003/the-movie-database-tmdb-comprehensive-dataset"


class DatasetService:
    def __init__(self) -> None:
        ensure_directories()

    def download_latest(self) -> dict:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError(
                "kagglehub nao esta instalado. Rode 'pip install -r requirements.txt'."
            ) from exc

        dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        copied_files = []

        for csv_file in dataset_path.glob("*.csv"):
            target = RAW_DIR / csv_file.name
            shutil.copy2(csv_file, target)
            copied_files.append(target.name)

        if not copied_files:
            raise RuntimeError("Nenhum arquivo CSV foi encontrado no dataset baixado.")

        return {
            "dataset": KAGGLE_DATASET,
            "source_path": str(dataset_path),
            "raw_path": str(RAW_DIR),
            "files": sorted(copied_files),
        }

    def dataset_status(self) -> dict:
        files = sorted(path.name for path in RAW_DIR.glob("*.csv"))
        return {
            "available": bool(files),
            "files": files,
            "raw_path": str(RAW_DIR),
        }
