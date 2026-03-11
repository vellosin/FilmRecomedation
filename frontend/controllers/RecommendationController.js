export class RecommendationController {
  constructor({ apiService, sessionService, movieDetailsView, recommendationsView, searchController }) {
    this.apiService = apiService;
    this.sessionService = sessionService;
    this.movieDetailsView = movieDetailsView;
    this.recommendationsView = recommendationsView;
    this.searchController = searchController;
    this.movieCache = new Map();
    this.currentRecommendations = [];
    this.hint = document.querySelector('#recommendation-hint');
    this.userId = 'local-user';
    this.isLoadingRecommendations = false;
  }

  bind() {
    const recommendByMovieButton = document.querySelector('#recommend-by-movie-button');
    const recommendProfileButton = document.querySelector('#recommend-profile-button');
    const clearProfileButton = document.querySelector('#clear-profile-button');

    recommendByMovieButton?.addEventListener('click', () => this.recommendByProfile());
    recommendProfileButton?.addEventListener('click', () => this.recommendByProfile());
    clearProfileButton?.addEventListener('click', () => this.clearProfile());

    this.renderProfile(this.sessionService.getProfile());
  }

  async showMovie(movieId) {
    const movie = await this.apiService.getMovie(movieId);
    this.movieCache.set(Number(movie.movie_id), movie);
    this.movieDetailsView.render(movie, {
      onAction: (selectedMovie, action) => this.handleFeedback(selectedMovie, action),
    });
  }

  async handleFeedback(movie, action) {
    this.movieCache.set(Number(movie.movie_id), movie);
    const profile = this.sessionService.addMovie(movie, action);
    this.renderProfile(profile);
    await this.apiService.saveFeedback({
      userId: this.userId,
      movieId: movie.movie_id,
      action,
    });
    await this.refreshRecommendationsAfterFeedback();
  }

  async handleFeedbackById(movieId, action) {
    let movie = this.movieCache.get(Number(movieId));
    if (!movie) {
      movie = await this.apiService.getMovie(movieId);
      this.movieCache.set(Number(movie.movie_id), movie);
    }
    await this.handleFeedback(movie, action);
  }

  async recommendByProfile() {
    if (this.isLoadingRecommendations) {
      return;
    }

    const profile = this.sessionService.getProfile();
    const positiveCount = new Set([...profile.likes, ...profile.favorites]).size;
    if (positiveCount < 3) {
      this.showHint('Curta ou favorite pelo menos 3 filmes antes de pedir recomendacoes. Se quiser, voce pode adicionar mais filmes para melhorar o perfil.');
      this.recommendationsView.renderMessage('Ainda nao ha base suficiente no perfil para recomendar.');
      return;
    }

    const excludedMovieIds = [...profile.likes, ...profile.dislikes, ...profile.favorites];
    this.setRecommendationsLoading(true);
    try {
      const response = await this.apiService.getRecommendationsByProfile(profile, excludedMovieIds, 12, this.userId);
      this.showHint('Sugestoes construidas a partir do seu perfil atual.');
      this.renderRecommendations(response.items);
    } catch (error) {
      this.showHint(error.message);
      this.recommendationsView.renderMessage(error.message);
    } finally {
      this.setRecommendationsLoading(false);
    }
  }

  renderProfile(profile) {
    this.movieDetailsView.updateProfileLists(profile, this.movieCache);
    this.searchController?.updateProfile(profile);
    this.renderRecommendations(this.currentRecommendations);
    const positiveCount = new Set([...profile.likes, ...profile.favorites]).size;
    if (positiveCount < 3) {
      this.showHint(`Curta ou favorite pelo menos 3 filmes antes de pedir recomendacoes. Atualmente: ${positiveCount}/3. Se quiser, voce pode adicionar mais filmes para melhorar o perfil.`);
    }
  }

  renderRecommendations(items) {
    const profile = this.sessionService.getProfile();
    const excludedIds = new Set([...profile.likes, ...profile.dislikes, ...profile.favorites]);
    this.currentRecommendations = (items || []).filter((movie) => !excludedIds.has(Number(movie.movie_id)));
    this.currentRecommendations.forEach((item) => this.movieCache.set(Number(item.movie_id), item));
    if (!this.currentRecommendations.length) {
      this.recommendationsView.renderMessage('Nenhuma recomendacao disponivel para o perfil atual.');
      return;
    }
    this.recommendationsView.render(this.currentRecommendations, {
      onSelect: (movieId) => this.showMovie(movieId),
      onAction: (movieId, action) => this.handleFeedbackById(movieId, action),
      profile,
    });
  }

  async refreshRecommendationsAfterFeedback() {
    const profile = this.sessionService.getProfile();
    if (profile.likes.length || profile.favorites.length) {
      await this.recommendByProfile();
      return;
    }

    this.renderRecommendations(this.currentRecommendations);
  }

  async clearProfile() {
    try {
      await this.apiService.clearFeedback(this.userId);
    } catch (error) {
      this.showHint(`Nao foi possivel limpar o historico salvo: ${error.message}`);
      return;
    }

    const profile = this.sessionService.clearProfile();
    this.currentRecommendations = [];
    this.renderProfile(profile);
    this.renderRecommendations([]);
    this.showHint('Perfil local e memoria persistida foram limpos. Curta ou favorite pelo menos 3 filmes para montar um novo perfil.');
  }

  showHint(message) {
    if (this.hint) {
      this.hint.textContent = message;
    }
  }

  setRecommendationsLoading(isLoading) {
    this.isLoadingRecommendations = isLoading;

    const buttons = [
      document.querySelector('#recommend-by-movie-button'),
      document.querySelector('#recommend-profile-button'),
    ].filter(Boolean);

    buttons.forEach((button) => {
      if (!button.dataset.defaultLabel) {
        button.dataset.defaultLabel = button.textContent.trim();
      }
      button.disabled = isLoading;
      button.setAttribute('aria-busy', String(isLoading));
      button.textContent = isLoading ? 'Buscando recomendacoes...' : button.dataset.defaultLabel;
    });

    if (isLoading) {
      this.showHint('Montando recomendacoes para o seu perfil. Aguarde alguns segundos.');
      this.recommendationsView.renderLoading('Buscando recomendacoes para voce...');
    }
  }
}
