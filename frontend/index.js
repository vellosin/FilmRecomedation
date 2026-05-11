import { BrowserApiService } from './services/BrowserApiService.js?v=20260511-1';
import { SessionService } from './services/SessionService.js';
import { MovieListView } from './views/MovieListView.js';
import { MovieDetailsView } from './views/MovieDetailsView.js?v=20260511-2';
import { TrainingView } from './views/TrainingView.js';
import { SearchController } from './controllers/SearchController.js';
import { RecommendationController } from './controllers/RecommendationController.js?v=20260511-3';
import { TrainingController } from './controllers/TrainingController.js';
import { AppController } from './controllers/AppController.js';

const apiService = new BrowserApiService();
const sessionService = new SessionService();

const searchResultsView = new MovieListView('#search-results');
const recommendationsView = new MovieListView('#recommendations-results');
const movieDetailsView = new MovieDetailsView('#movie-details');
const trainingView = new TrainingView();

const searchController = new SearchController({
  apiService,
  movieListView: searchResultsView,
});

const recommendationController = new RecommendationController({
  apiService,
  sessionService,
  movieDetailsView,
  recommendationsView,
  searchController,
});

const trainingController = new TrainingController({
  apiService,
  trainingView,
});

AppController.init({
  searchController,
  recommendationController,
  trainingController,
});
