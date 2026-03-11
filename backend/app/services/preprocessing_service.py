import ast
import re

import pandas as pd

from .paths import PROCESSED_MOVIES_FILE, RAW_DIR, ensure_directories
from .training_status_service import TrainingStatusService


MAX_REVIEWS_PER_MOVIE = 2
MAX_REVIEW_CHARS = 280
MAX_OVERVIEW_CHARS = 700
MAX_CAST_MEMBERS = 10
MAX_DIRECTORS = 3
MAX_WRITERS = 4
MAX_PRODUCERS = 4


class PreprocessingService:
    def __init__(self) -> None:
        ensure_directories()
        self.status_service = TrainingStatusService()

    def build_training_dataframe(self) -> pd.DataFrame:
        self.status_service.write(
            {"stage": "preprocessing", "message": "Carregando e transformando os CSVs do TMDB."}
        )

        movies = self._read_csv("movies")
        cast = self._read_csv("cast")
        crew = self._read_csv("crew")
        reviews = self._read_csv("reviews")

        movie_id_column = "id" if "id" in movies.columns else "movie_id"
        movies = movies.copy()
        movies["movie_id"] = pd.to_numeric(movies[movie_id_column], errors="coerce")
        movies = movies.dropna(subset=["movie_id"]).copy()
        movies["movie_id"] = movies["movie_id"].astype(int)

        title_column = self._pick_column(movies.columns, ["title", "movie_title", "original_title"])
        overview_column = self._pick_column(movies.columns, ["overview", "description", "tagline"])
        genres_column = self._pick_column(movies.columns, ["genres"])
        release_date_column = self._pick_column(movies.columns, ["release_date"])
        vote_average_column = self._pick_column(movies.columns, ["vote_average", "rating", "vote_mean"])
        popularity_column = self._pick_column(movies.columns, ["popularity"])
        runtime_column = self._pick_column(movies.columns, ["runtime"])
        poster_column = self._pick_column(movies.columns, ["poster_path", "poster_url"], required=False)

        cast_agg = self._aggregate_cast(cast)
        crew_agg = self._aggregate_crew(crew)
        reviews_agg = self._aggregate_reviews(reviews)

        df = movies.merge(cast_agg, how="left", on="movie_id")
        df = df.merge(crew_agg, how="left", on="movie_id")
        df = df.merge(reviews_agg, how="left", on="movie_id")

        processed = pd.DataFrame(
            {
                "movie_id": df["movie_id"],
                "title": df[title_column].fillna("Sem titulo").map(self._clean_text),
                "overview": df[overview_column].fillna("").map(
                    lambda value: self._clean_text(value, limit=MAX_OVERVIEW_CHARS)
                ),
                "genres_text": df[genres_column].fillna("").map(self._parse_genres),
                "cast_text": df.get("cast_text", "").fillna("").map(self._clean_text),
                "director_text": df.get("director_text", "").fillna("").map(self._clean_text),
                "writer_text": df.get("writer_text", "").fillna("").map(self._clean_text),
                "producer_text": df.get("producer_text", "").fillna("").map(self._clean_text),
                "review_text": df.get("review_text", "").fillna("").map(self._clean_text),
                "review_score": pd.to_numeric(df.get("review_score", 0), errors="coerce").fillna(0),
                "review_count": pd.to_numeric(df.get("review_count", 0), errors="coerce").fillna(0),
                "vote_average": pd.to_numeric(df[vote_average_column], errors="coerce").fillna(0),
                "popularity": pd.to_numeric(df[popularity_column], errors="coerce").fillna(0),
                "runtime": pd.to_numeric(df[runtime_column], errors="coerce").fillna(0),
                "release_year": pd.to_datetime(df[release_date_column], errors="coerce").dt.year.fillna(0).astype(int),
                "poster_path": df[poster_column].fillna("") if poster_column else "",
            }
        )

        processed["feature_text"] = (
            processed["title"]
            + " "
            + processed["overview"]
            + " "
            + processed["genres_text"]
            + " "
            + processed["cast_text"]
            + " "
            + processed["director_text"]
            + " "
            + processed["writer_text"]
            + " "
            + processed["producer_text"]
            + " "
            + processed["review_text"]
        ).str.replace(r"\s+", " ", regex=True).str.strip()

        processed = processed.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)
        processed.to_csv(PROCESSED_MOVIES_FILE, index=False)

        self.status_service.write(
            {"stage": "preprocessed", "message": "Base processada e pronta para treino."}
        )
        return processed

    def _read_csv(self, prefix: str) -> pd.DataFrame:
        matches = sorted(RAW_DIR.glob(f"{prefix}*.csv"))
        if not matches:
            raise FileNotFoundError(
                f"Arquivo com prefixo '{prefix}' nao foi encontrado em {RAW_DIR}."
            )
        csv_path = matches[0]
        try:
            return pd.read_csv(csv_path, low_memory=False)
        except pd.errors.ParserError:
            return pd.read_csv(
                csv_path,
                engine="python",
                on_bad_lines="skip",
            )

    def _pick_column(self, columns, candidates: list[str], required: bool = True) -> str | None:
        columns_lower = {column.lower(): column for column in columns}
        for candidate in candidates:
            if candidate.lower() in columns_lower:
                return columns_lower[candidate.lower()]
        if required:
            raise KeyError(f"Nenhuma coluna correspondente foi encontrada para: {candidates}")
        return None

    def _aggregate_cast(self, cast_df: pd.DataFrame) -> pd.DataFrame:
        if cast_df.empty:
            return pd.DataFrame(columns=["movie_id", "cast_text"])

        working = cast_df.copy()
        working["movie_id"] = pd.to_numeric(working["movie_id"], errors="coerce")
        working = working.dropna(subset=["movie_id"])
        if "cast_order" in working.columns:
            working["cast_order"] = pd.to_numeric(working["cast_order"], errors="coerce").fillna(999)
            working = working.sort_values(["movie_id", "cast_order"])
        grouped = (
            working.groupby("movie_id")["name"]
            .apply(lambda names: self._join_people(names, limit=MAX_CAST_MEMBERS))
            .reset_index(name="cast_text")
        )
        grouped["cast_text"] = grouped["cast_text"].map(self._clean_text)
        return grouped

    def _aggregate_crew(self, crew_df: pd.DataFrame) -> pd.DataFrame:
        if crew_df.empty:
            return pd.DataFrame(columns=["movie_id", "director_text", "writer_text", "producer_text"])

        working = crew_df.copy()
        working["movie_id"] = pd.to_numeric(working["movie_id"], errors="coerce")
        working = working.dropna(subset=["movie_id"])
        job_column = self._pick_column(working.columns, ["job"], required=False)
        department_column = self._pick_column(working.columns, ["department"], required=False)
        normalized_job = working[job_column].fillna("").astype(str).str.lower() if job_column else pd.Series("", index=working.index)
        normalized_department = (
            working[department_column].fillna("").astype(str).str.lower()
            if department_column
            else pd.Series("", index=working.index)
        )

        directors = working[(normalized_job == "director") | (normalized_department == "directing")]
        writers = working[
            normalized_job.isin({"writer", "screenplay", "story", "characters"})
            | normalized_department.eq("writing")
        ]
        producers = working[
            normalized_job.isin({"producer", "executive producer", "co-producer"})
            | normalized_department.eq("production")
        ]

        grouped = pd.DataFrame({"movie_id": sorted(working["movie_id"].astype(int).unique())})
        grouped = grouped.merge(
            directors.groupby("movie_id")["name"]
            .apply(lambda names: self._join_people(names, limit=MAX_DIRECTORS))
            .reset_index(name="director_text"),
            how="left",
            on="movie_id",
        )
        grouped = grouped.merge(
            writers.groupby("movie_id")["name"]
            .apply(lambda names: self._join_people(names, limit=MAX_WRITERS))
            .reset_index(name="writer_text"),
            how="left",
            on="movie_id",
        )
        grouped = grouped.merge(
            producers.groupby("movie_id")["name"]
            .apply(lambda names: self._join_people(names, limit=MAX_PRODUCERS))
            .reset_index(name="producer_text"),
            how="left",
            on="movie_id",
        )

        for column in ["director_text", "writer_text", "producer_text"]:
            grouped[column] = grouped[column].fillna("").map(self._clean_text)
        return grouped

    def _aggregate_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        if reviews_df.empty:
            return pd.DataFrame(columns=["movie_id", "review_text", "review_score", "review_count"])

        working = reviews_df.copy()
        working["movie_id"] = pd.to_numeric(working["movie_id"], errors="coerce")
        working = working.dropna(subset=["movie_id"])
        content_column = self._pick_column(working.columns, ["content", "review", "text"])
        score_column = self._pick_column(working.columns, ["author_rating", "rating"], required=False)

        aggregated = working.groupby("movie_id").agg(
            review_text=(
                content_column,
                lambda values: " ".join(
                    pd.Series(values)
                    .dropna()
                    .astype(str)
                    .map(lambda review: self._clean_text(review, limit=MAX_REVIEW_CHARS))
                    .head(MAX_REVIEWS_PER_MOVIE)
                ),
            ),
            review_count=(content_column, "count"),
        )
        if score_column:
            aggregated["review_score"] = pd.to_numeric(
                working.groupby("movie_id")[score_column].mean(), errors="coerce"
            ).fillna(0)
        else:
            aggregated["review_score"] = 0
        return aggregated.reset_index()

    def _parse_genres(self, value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = str(value)
        if not text.strip():
            return ""
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                names = []
                for item in parsed:
                    if isinstance(item, dict) and "name" in item:
                        names.append(str(item["name"]))
                    elif isinstance(item, str):
                        names.append(item)
                if names:
                    return " ".join(names)
        except (ValueError, SyntaxError):
            pass
        return " ".join(re.findall(r"[A-Za-z][A-Za-z\- ]+", text))

    def _clean_text(self, value, limit: int | None = None) -> str:
        text = str(value or "")
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^\w\s\-.,:;!?()'/]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if limit and len(text) > limit:
            text = text[:limit].rsplit(" ", 1)[0].strip()
        return text

    def _join_people(self, names: pd.Series, limit: int) -> str:
        unique_names = pd.Series(names).dropna().astype(str).map(str.strip)
        unique_names = unique_names[unique_names != ""]
        unique_names = unique_names.drop_duplicates()
        return " ".join(unique_names.head(limit))
