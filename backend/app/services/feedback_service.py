import json
from datetime import datetime, timezone

from .paths import FEEDBACK_FILE, ensure_directories


class FeedbackService:
    def __init__(self) -> None:
        ensure_directories()

    def append(self, user_id: str, movie_id: int, action: str) -> dict:
        current = []
        if FEEDBACK_FILE.exists():
            current = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))

        item = {
            "user_id": user_id,
            "movie_id": movie_id,
            "action": action,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        current.append(item)
        FEEDBACK_FILE.write_text(json.dumps(current, ensure_ascii=True, indent=2), encoding="utf-8")
        return item

    def clear_user(self, user_id: str) -> dict:
        if not FEEDBACK_FILE.exists():
            return {"user_id": user_id, "removed_entries": 0}

        current = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        remaining = [entry for entry in current if str(entry.get("user_id", "local-user")) != user_id]
        removed_entries = len(current) - len(remaining)
        FEEDBACK_FILE.write_text(json.dumps(remaining, ensure_ascii=True, indent=2), encoding="utf-8")
        return {"user_id": user_id, "removed_entries": removed_entries}

    def load_user_state(self, user_id: str) -> dict:
        if not FEEDBACK_FILE.exists():
            return {"likes": [], "favorites": [], "dislikes": [], "actions": []}

        current = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        latest_by_movie = {}
        for entry in current:
            if str(entry.get("user_id", "local-user")) != user_id:
                continue
            latest_by_movie[int(entry["movie_id"])] = entry

        actions = sorted(latest_by_movie.values(), key=lambda item: item.get("created_at", ""))
        likes = [int(item["movie_id"]) for item in actions if item.get("action") == "like"]
        favorites = [int(item["movie_id"]) for item in actions if item.get("action") == "favorite"]
        dislikes = [int(item["movie_id"]) for item in actions if item.get("action") == "dislike"]
        return {
            "likes": likes,
            "favorites": favorites,
            "dislikes": dislikes,
            "actions": actions,
        }
