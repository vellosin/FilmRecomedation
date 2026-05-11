import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
PROCESSED_MOVIES_FILE = BASE_DIR / 'data' / 'processed' / 'movies_processed.csv'
FRONTEND_DATA_DIR = BASE_DIR.parent / 'frontend' / 'data'

CATALOG_COLUMNS = [
    'movie_id',
    'title',
    'overview',
    'genres_text',
    'cast_text',
    'vote_average',
    'popularity',
    'release_year',
    'poster_path',
]


def json_safe(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    normalized = embeddings.astype('float32', copy=True)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized /= norms
    return normalized


def export_catalog(movie_ids: list[int]) -> list[dict]:
    catalog = pd.read_csv(PROCESSED_MOVIES_FILE)
    catalog = catalog[catalog['movie_id'].isin(movie_ids)].copy()
    catalog['movie_id'] = catalog['movie_id'].astype(int)
    catalog['__order'] = pd.Categorical(catalog['movie_id'], categories=movie_ids, ordered=True)
    catalog = catalog.sort_values('__order')

    records = []
    for row in catalog[CATALOG_COLUMNS].to_dict(orient='records'):
      records.append({key: json_safe(value) for key, value in row.items()})
    return records


def main() -> None:
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(ARTIFACTS_DIR / 'movie_recommender.joblib')
    movie_ids = [int(movie_id) for movie_id in bundle['movie_ids']]
    embeddings = np.load(ARTIFACTS_DIR / 'movie_embeddings.npy')
    normalized_embeddings = normalize_embeddings(embeddings)
    catalog_records = export_catalog(movie_ids)

    (FRONTEND_DATA_DIR / 'catalog.json').write_text(
        json.dumps(catalog_records, ensure_ascii=False, separators=(',', ':')),
        encoding='utf-8',
    )
    (FRONTEND_DATA_DIR / 'movie_ids.json').write_text(
        json.dumps(movie_ids, ensure_ascii=True, separators=(',', ':')),
        encoding='utf-8',
    )
    normalized_embeddings.tofile(FRONTEND_DATA_DIR / 'embeddings.f32')

    for source_name, target_name in [
        ('model_config.json', 'model-config.json'),
        ('training_report.json', 'training-report.json'),
    ]:
        source_file = ARTIFACTS_DIR / source_name
        if source_file.exists():
            (FRONTEND_DATA_DIR / target_name).write_text(source_file.read_text(encoding='utf-8'), encoding='utf-8')

    runtime_manifest = {
        'mode': 'browser',
        'catalog_count': len(catalog_records),
        'embedding_shape': list(normalized_embeddings.shape),
        'embedding_dtype': str(normalized_embeddings.dtype),
        'training_available': (ARTIFACTS_DIR / 'training_report.json').exists(),
        'message': 'Modelo treinado offline e pronto para recomendacoes no navegador.',
    }
    (FRONTEND_DATA_DIR / 'runtime-manifest.json').write_text(
        json.dumps(runtime_manifest, ensure_ascii=True, indent=2),
        encoding='utf-8',
    )

    print(json.dumps(runtime_manifest, ensure_ascii=True))


if __name__ == '__main__':
    main()