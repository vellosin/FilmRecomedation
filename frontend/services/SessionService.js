export class SessionService {
  constructor(storageKey = 'movie-recommender-profile') {
    this.storageKey = storageKey;
  }

  getProfile() {
    const raw = localStorage.getItem(this.storageKey);
    if (!raw) {
      return { likes: [], dislikes: [], favorites: [] };
    }

    try {
      return JSON.parse(raw);
    } catch {
      return { likes: [], dislikes: [], favorites: [] };
    }
  }

  saveProfile(profile) {
    localStorage.setItem(this.storageKey, JSON.stringify(profile));
    return profile;
  }

  clearProfile() {
    localStorage.removeItem(this.storageKey);
    return { likes: [], dislikes: [], favorites: [] };
  }

  addMovie(movie, action) {
    const profile = this.getProfile();
    const movieId = Number(movie.movie_id);

    profile.likes = profile.likes.filter((id) => id !== movieId);
    profile.dislikes = profile.dislikes.filter((id) => id !== movieId);
    profile.favorites = profile.favorites.filter((id) => id !== movieId);

    if (action === 'like') {
      profile.likes.push(movieId);
    }
    if (action === 'dislike') {
      profile.dislikes.push(movieId);
    }
    if (action === 'favorite') {
      profile.favorites.push(movieId);
      if (!profile.likes.includes(movieId)) {
        profile.likes.push(movieId);
      }
    }

    return this.saveProfile(profile);
  }
}
