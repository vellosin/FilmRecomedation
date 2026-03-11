import pandas as pd


def _json_safe_value(value):
    if pd.isna(value):
        return None
    return value

from .paths import PROCESSED_MOVIES_FILE


class CatalogService:
    def load_catalog(self) -> pd.DataFrame:
        if not PROCESSED_MOVIES_FILE.exists():
            raise FileNotFoundError(
                "Catalogo processado nao encontrado. Rode o preprocessamento ou o treino primeiro."
            )
        return pd.read_csv(PROCESSED_MOVIES_FILE)

    def search_movies(self, query: str = "", limit: int = 20) -> list[dict]:
        df = self.load_catalog()
        working = df.copy()

        if query.strip():
            mask = working["title"].fillna("").str.contains(query, case=False, na=False)
            working = working[mask]
        else:
            working = working.sort_values(["vote_average", "popularity"], ascending=False)

        columns = [
            "movie_id",
            "title",
            "overview",
            "genres_text",
            "vote_average",
            "popularity",
            "release_year",
            "poster_path",
        ]
        return [self.serialize_movie(row) for row in working.head(limit)[columns].to_dict(orient="records")]

    def get_movie(self, movie_id: int) -> dict:
        df = self.load_catalog()
        row = df[df["movie_id"] == movie_id]
        if row.empty:
            raise KeyError(f"Filme {movie_id} nao encontrado.")
        return self.serialize_movie(row.iloc[0].to_dict())

    def serialize_movie(self, movie: dict) -> dict:
        return {key: _json_safe_value(value) for key, value in movie.items()}
