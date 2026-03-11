export class AppController {
  constructor({ searchController, recommendationController, trainingController }) {
    this.searchController = searchController;
    this.recommendationController = recommendationController;
    this.trainingController = trainingController;
  }

  static init(deps) {
    const controller = new AppController(deps);
    controller.init();
    return controller;
  }

  init() {
    this.searchController.bind();
    this.recommendationController.bind();
    this.trainingController.bind();

    this.searchController.registerMovieSelection(async (movieId) => {
      await this.recommendationController.showMovie(movieId);
    });

    this.searchController.registerMovieAction(async (movieId, action) => {
      await this.recommendationController.handleFeedbackById(movieId, action);
    });

    this.trainingController.refreshStatus();
    this.searchController.initialLoad().catch((error) => {
      const target = document.querySelector('#search-results');
      if (target) {
        target.innerHTML = `<div class="movie-card">${error.message}</div>`;
      }
    });
  }
}
