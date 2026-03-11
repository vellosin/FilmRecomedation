export class SearchController {
  constructor({ apiService, movieListView }) {
    this.apiService = apiService;
    this.movieListView = movieListView;
    this.onMovieSelected = null;
    this.onMovieAction = null;
    this.moviesById = new Map();
    this.currentMovies = [];
    this.currentProfile = { likes: [], dislikes: [], favorites: [] };
  }

  bind() {
    const form = document.querySelector('#search-form');
    const input = document.querySelector('#search-input');

    form?.addEventListener('submit', async (event) => {
      event.preventDefault();
      await this.search(input?.value || '');
    });

    input?.addEventListener('input', () => {
      if (!input.value.trim()) {
        this.currentMovies = [];
        this.movieListView.renderMessage('Digite um titulo para pesquisar filmes.');
      }
    });
  }

  async initialLoad() {
    this.currentMovies = [];
    this.movieListView.renderMessage('Digite um titulo para pesquisar filmes.');
  }

  async search(query) {
    if (!query.trim()) {
      this.currentMovies = [];
      this.movieListView.renderMessage('Digite um titulo para pesquisar filmes.');
      return [];
    }

    const response = await this.apiService.searchMovies(query);
    this.currentMovies = response.items;
    response.items.forEach((movie) => this.moviesById.set(Number(movie.movie_id), movie));
    this.movieListView.render(response.items, {
      onSelect: (movieId) => this.onMovieSelected?.(movieId),
      onAction: (movieId, action) => this.onMovieAction?.(movieId, action),
      profile: this.currentProfile,
    });
    return response.items;
  }

  registerMovieSelection(callback) {
    this.onMovieSelected = callback;
  }

  registerMovieAction(callback) {
    this.onMovieAction = callback;
  }

  updateProfile(profile) {
    this.currentProfile = profile;
    this.movieListView.render(this.currentMovies, {
      onSelect: (movieId) => this.onMovieSelected?.(movieId),
      onAction: (movieId, action) => this.onMovieAction?.(movieId, action),
      profile: this.currentProfile,
    });
  }
}
