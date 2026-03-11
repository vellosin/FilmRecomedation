from fastapi import APIRouter, HTTPException, Query

from app.services.catalog_service import CatalogService


router = APIRouter(prefix="/movies", tags=["movies"])
service = CatalogService()


@router.get("")
def search_movies(query: str = Query(default=""), limit: int = Query(default=20, ge=1, le=100)):
    try:
        return {"items": service.search_movies(query=query, limit=limit)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{movie_id}")
def get_movie(movie_id: int):
    try:
        return service.get_movie(movie_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
