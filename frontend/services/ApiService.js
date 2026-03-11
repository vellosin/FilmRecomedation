export class ApiService {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async request(path, options = {}) {
    const response = await fetch(`${this.baseUrl}${path}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
      ...options,
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || 'Falha ao comunicar com a API.');
    }
    return payload;
  }

  getHealth() {
    return this.request('/health');
  }

  getDatasetStatus() {
    return this.request('/dataset/status');
  }

  downloadDataset() {
    return this.request('/dataset/download', { method: 'POST' });
  }

  trainModel() {
    return this.request('/train', { method: 'POST' });
  }

  getTrainingStatus() {
    return this.request('/training/status');
  }

  getTrainingConfig() {
    return this.request('/training/config');
  }

  getTrainingReport() {
    return this.request('/training/report');
  }

  searchMovies(query = '', limit = 24) {
    const params = new URLSearchParams({ query, limit: String(limit) });
    return this.request(`/movies?${params.toString()}`);
  }

  getMovie(movieId) {
    return this.request(`/movies/${movieId}`);
  }

  getRecommendationsByMovie(movieId, excludedMovieIds = [], topN = 12) {
    return this.request('/recommendations/by-movie', {
      method: 'POST',
      body: JSON.stringify({
        movie_id: Number(movieId),
        excluded_movie_ids: excludedMovieIds,
        top_n: topN,
      }),
    });
  }

  getRecommendationsByProfile(profile, excludedMovieIds = [], topN = 12, userId = 'local-user') {
    return this.request('/recommendations/by-profile', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
        likes: profile.likes,
        dislikes: profile.dislikes,
        favorites: profile.favorites,
        excluded_movie_ids: excludedMovieIds,
        top_n: topN,
      }),
    });
  }

  saveFeedback({ userId, movieId, action }) {
    return this.request('/feedback', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
        movie_id: Number(movieId),
        action,
      }),
    });
  }

  clearFeedback(userId = 'local-user') {
    return this.request('/feedback/clear', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
      }),
    });
  }
}
