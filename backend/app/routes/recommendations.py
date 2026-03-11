from fastapi import APIRouter, HTTPException

from app.models.schemas import ProfileRecommendationRequest, MovieRecommendationRequest
from app.services.recommendation_service import RecommendationService


router = APIRouter(prefix="/recommendations", tags=["recommendations"])
service = RecommendationService()


@router.post("/by-movie")
def recommend_by_movie(payload: MovieRecommendationRequest):
    try:
        return {
            "items": service.recommend_by_movie(
                movie_id=payload.movie_id,
                top_n=payload.top_n,
                excluded_movie_ids=payload.excluded_movie_ids,
            )
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/by-profile")
def recommend_by_profile(payload: ProfileRecommendationRequest):
    try:
        return {
            "items": service.recommend_by_profile(
                user_id=payload.user_id,
                likes=payload.likes,
                dislikes=payload.dislikes,
                favorites=payload.favorites,
                excluded_movie_ids=payload.excluded_movie_ids,
                top_n=payload.top_n,
            )
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        import traceback
        print("Erro inesperado em /recommendations/by-profile:", exc)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erro inesperado: {exc}") from exc
