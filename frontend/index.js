import { ApiService } from './services/ApiService.js';
import { SessionService } from './services/SessionService.js';
import { MovieListView } from './views/MovieListView.js';
import { MovieDetailsView } from './views/MovieDetailsView.js';
import { TrainingView } from './views/TrainingView.js';
import { SearchController } from './controllers/SearchController.js';
import { RecommendationController } from './controllers/RecommendationController.js';
import { TrainingController } from './controllers/TrainingController.js';
import { AppController } from './controllers/AppController.js';

const apiService = new ApiService();
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
