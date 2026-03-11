from typing import Literal

from pydantic import BaseModel, Field


class MovieRecommendationRequest(BaseModel):
    movie_id: int
    excluded_movie_ids: list[int] = Field(default_factory=list)
    top_n: int = Field(default=10, ge=1, le=50)


class ProfileRecommendationRequest(BaseModel):
    user_id: str = "local-user"
    likes: list[int] = Field(default_factory=list)
    dislikes: list[int] = Field(default_factory=list)
    favorites: list[int] = Field(default_factory=list)
    excluded_movie_ids: list[int] = Field(default_factory=list)
    top_n: int = Field(default=10, ge=1, le=50)


class FeedbackRequest(BaseModel):
    user_id: str = "local-user"
    movie_id: int
    action: Literal["like", "dislike", "favorite"]


class FeedbackResetRequest(BaseModel):
    user_id: str = "local-user"
