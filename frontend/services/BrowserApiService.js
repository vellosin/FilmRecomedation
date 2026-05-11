export class BrowserApiService {
  constructor(dataBaseUrl = './data') {
    this.dataBaseUrl = String(dataBaseUrl).replace(/\/$/, '');
    this.isBrowserRuntime = true;
    this.catalogPromise = null;
    this.runtimePromise = null;
    this.reportPromise = null;
    this.configPromise = null;
    this.embeddingStorePromise = null;
  }

  async requestJson(fileName, fallbackValue = null) {
    const response = await fetch(`${this.dataBaseUrl}/${fileName}`, {
      headers: { Accept: 'application/json' },
    });

    if (!response.ok) {
      if (fallbackValue !== null) return fallbackValue;
      throw new Error(`Nao foi possivel carregar ${fileName}.`);
    }

    return response.json();
  }

  async getCatalog() {
    if (!this.catalogPromise) {
      this.catalogPromise = this.requestJson('catalog.json').then((items) => {
        const byId = new Map();
        items.forEach((movie, index) => {
          byId.set(Number(movie.movie_id), { ...movie, __index: index });
        });
        return { items, byId };
      });
    }
    return this.catalogPromise;
  }

  async getRuntimeManifest() {
    if (!this.runtimePromise) {
      this.runtimePromise = this.requestJson('runtime-manifest.json');
    }
    return this.runtimePromise;
  }

  async getTrainingConfig() {
    if (!this.configPromise) {
      this.configPromise = this.requestJson('model-config.json', {});
    }
    return this.configPromise;
  }

  async getTrainingReport() {
    if (!this.reportPromise) {
      this.reportPromise = this.requestJson('training-report.json', {});
    }
    return this.reportPromise;
  }

  async getEmbeddingStore() {
    if (!this.embeddingStorePromise) {
      this.embeddingStorePromise = Promise.all([
        this.getRuntimeManifest(),
        fetch(`${this.dataBaseUrl}/embeddings.f32`),
      ]).then(async ([manifest, response]) => {
        if (!response.ok) {
          throw new Error('Nao foi possivel carregar os embeddings do modelo.');
        }

        const buffer = await response.arrayBuffer();
        const vectorShape = manifest.embedding_shape || [0, 0];
        const vectorCount = Number(vectorShape[0] || 0);
        const vectorSize = Number(vectorShape[1] || 0);
        const embeddings = new Float32Array(buffer);

        if (!vectorCount || !vectorSize || embeddings.length !== vectorCount * vectorSize) {
          throw new Error('Embeddings invalidos para o runtime do navegador.');
        }

        return { embeddings, vectorCount, vectorSize };
      });
    }
    return this.embeddingStorePromise;
  }

  async getHealth() {
    const manifest = await this.getRuntimeManifest();
    return {
      status: 'ok',
      mode: manifest.mode || 'browser',
      catalog_count: manifest.catalog_count || 0,
    };
  }

  async getDatasetStatus() {
    const manifest = await this.getRuntimeManifest();
    return {
      available: true,
      files: ['catalog.json', 'embeddings.f32', 'model-config.json', 'training-report.json'],
      source: 'browser-runtime',
      total_movies: manifest.catalog_count || 0,
    };
  }

  async downloadDataset() {
    throw new Error('A versao publica usa artefatos pre-treinados no navegador. Download de dataset fica restrito ao ambiente de treino offline.');
  }

  async trainModel() {
    throw new Error('O treino foi movido para o fluxo offline. A versao publica do recomendador nao treina no servidor nem no navegador.');
  }

  async getTrainingStatus() {
    const manifest = await this.getRuntimeManifest();
    const report = await this.getTrainingReport();
    const summary = report.training_summary || {};

    return {
      stage: 'browser-runtime',
      message: manifest.message || 'Modelo pre-treinado carregado para recomendacoes locais.',
      progress: 100,
      is_running: false,
      loss: typeof summary.autoencoder_loss === 'number' ? summary.autoencoder_loss : null,
      val_loss: typeof summary.best_validation_score === 'number' ? summary.best_validation_score : null,
    };
  }

  async searchMovies(query = '', limit = 24) {
    const { items } = await this.getCatalog();
    const normalizedQuery = String(query || '').trim().toLowerCase();

    let working = items;
    if (normalizedQuery) {
      working = items.filter((movie) => String(movie.title || '').toLowerCase().includes(normalizedQuery));
    } else {
      working = [...items].sort((left, right) => {
        const scoreDiff = Number(right.vote_average || 0) - Number(left.vote_average || 0);
        if (scoreDiff !== 0) return scoreDiff;
        return Number(right.popularity || 0) - Number(left.popularity || 0);
      });
    }

    return { items: working.slice(0, limit).map((movie) => this.#stripRuntimeFields(movie)) };
  }

  async getMovie(movieId) {
    const { byId } = await this.getCatalog();
    const movie = byId.get(Number(movieId));
    if (!movie) throw new Error(`Filme ${movieId} nao encontrado.`);
    return this.#stripRuntimeFields(movie);
  }

  async getRecommendationsByMovie(movieId, excludedMovieIds = [], topN = 12) {
    const runtime = await this.#loadRuntime();
    const movie = runtime.byId.get(Number(movieId));
    if (!movie) throw new Error(`Filme ${movieId} nao encontrado.`);

    const queryVector = this.#embeddingSlice(runtime.embeddingStore, movie.__index);
    const excludedIds = new Set((excludedMovieIds || []).map((value) => Number(value)));
    excludedIds.add(Number(movieId));

    const items = this.#rankVector(runtime, queryVector, excludedIds, topN).map((item) => ({
      ...this.#stripRuntimeFields(item.movie),
      score: item.score,
    }));
    return { items };
  }

  async getRecommendationsByProfile(profile, excludedMovieIds = [], topN = 12, userId = 'local-user') {
    const runtime = await this.#loadRuntime();
    const mergedProfile = this.#mergeProfile(profile, userId);
    const positiveIds = [...new Set([...mergedProfile.likes, ...mergedProfile.favorites])];
    if (positiveIds.length < 3) {
      throw new Error('Curta ou favorite pelo menos 3 filmes antes de pedir recomendacoes. Se quiser, voce pode adicionar mais filmes para melhorar o perfil.');
    }

    const vector = this.#buildProfileVector(runtime, mergedProfile);
    const excludedIds = new Set([
      ...positiveIds,
      ...mergedProfile.dislikes,
      ...(excludedMovieIds || []).map((value) => Number(value)),
    ]);
    const likedCastSets = this.#collectCastSets(runtime, mergedProfile.likes);
    const favoriteCastSets = this.#collectCastSets(runtime, mergedProfile.favorites);
    const dislikedCastSets = this.#collectCastSets(runtime, mergedProfile.dislikes);

    const items = this.#rankVector(runtime, vector, excludedIds, topN, ({ movie, score }) => {
      const adjustedScore = this.#applyProfileWeights({
        baseScore: score,
        candidateMovie: movie,
        likedCastSets,
        favoriteCastSets,
        dislikedCastSets,
      });
      return {
        ...this.#stripRuntimeFields(movie),
        score: adjustedScore,
      };
    });

    items.sort((left, right) => right.score - left.score);
    return { items };
  }

  async saveFeedback() {
    return { ok: true, persisted: 'local-profile' };
  }

  async clearFeedback() {
    return { ok: true, persisted: 'local-profile' };
  }

  async #loadRuntime() {
    const [{ items, byId }, embeddingStore] = await Promise.all([this.getCatalog(), this.getEmbeddingStore()]);
    return { items, byId, embeddingStore };
  }

  #stripRuntimeFields(movie) {
    const { __index, ...publicMovie } = movie;
    return publicMovie;
  }

  #embeddingSlice(embeddingStore, index) {
    const start = index * embeddingStore.vectorSize;
    const end = start + embeddingStore.vectorSize;
    return embeddingStore.embeddings.subarray(start, end);
  }

  #rankVector(runtime, queryVector, excludedIds, topN, mapResult) {
    const results = [];
    const { items, embeddingStore } = runtime;

    for (let index = 0; index < embeddingStore.vectorCount; index += 1) {
      const movie = items[index];
      const movieId = Number(movie.movie_id);
      if (excludedIds.has(movieId)) continue;
      const score = this.#dotProduct(queryVector, this.#embeddingSlice(embeddingStore, index));
      if (!Number.isFinite(score)) continue;
      results.push({
        movie,
        score,
      });
    }

    results.sort((left, right) => right.score - left.score);
    return results.slice(0, Math.max(topN * 3, topN)).map((item) => mapResult ? mapResult(item) : {
      ...this.#stripRuntimeFields(item.movie),
      score: item.score,
    }).slice(0, topN);
  }

  #dotProduct(left, right) {
    let total = 0;
    for (let index = 0; index < left.length; index += 1) {
      total += left[index] * right[index];
    }
    return total;
  }

  #mergeProfile(profile, userId) {
    const likes = (profile?.likes || []).map((value) => Number(value));
    const dislikes = (profile?.dislikes || []).map((value) => Number(value));
    const favorites = (profile?.favorites || []).map((value) => Number(value));

    return {
      userId,
      likes,
      dislikes,
      favorites,
      weightsByMovie: Object.fromEntries([
        ...likes.map((movieId) => [movieId, 1.2]),
        ...dislikes.map((movieId) => [movieId, 0.96]),
        ...favorites.map((movieId) => [movieId, 2.0]),
      ]),
    };
  }

  #buildProfileVector(runtime, mergedProfile) {
    const { embeddingStore, byId } = runtime;
    const vector = new Float32Array(embeddingStore.vectorSize);
    let positiveCount = 0;

    for (const movieId of mergedProfile.likes) {
      const movie = byId.get(movieId);
      if (!movie) continue;
      this.#accumulateVector(vector, this.#embeddingSlice(embeddingStore, movie.__index), mergedProfile.weightsByMovie[movieId] || 1.0);
      positiveCount += 1;
    }

    for (const movieId of mergedProfile.favorites) {
      const movie = byId.get(movieId);
      if (!movie) continue;
      this.#accumulateVector(vector, this.#embeddingSlice(embeddingStore, movie.__index), mergedProfile.weightsByMovie[movieId] || 1.6);
      positiveCount += 1;
    }

    for (const movieId of mergedProfile.dislikes) {
      const movie = byId.get(movieId);
      if (!movie) continue;
      this.#accumulateVector(vector, this.#embeddingSlice(embeddingStore, movie.__index), -(mergedProfile.weightsByMovie[movieId] || 0.8));
    }

    if (!positiveCount) {
      throw new Error('Envie ao menos um filme curtido ou favorito para montar o perfil.');
    }

    const norm = Math.hypot(...vector);
    if (!Number.isFinite(norm) || norm === 0) {
      throw new Error('Nao foi possivel montar um vetor valido para o perfil. Adicione mais filmes ao perfil.');
    }

    for (let index = 0; index < vector.length; index += 1) {
      vector[index] /= norm;
    }
    return vector;
  }

  #accumulateVector(target, source, weight) {
    for (let index = 0; index < target.length; index += 1) {
      target[index] += source[index] * weight;
    }
  }

  #collectCastSets(runtime, movieIds) {
    return movieIds
      .map((movieId) => runtime.byId.get(Number(movieId)))
      .filter(Boolean)
      .map((movie) => this.#tokenizeCast(movie.cast_text));
  }

  #tokenizeCast(castText = '') {
    return new Set(String(castText || '').split(/\s+/).map((item) => item.trim().toLowerCase()).filter(Boolean));
  }

  #applyProfileWeights({ baseScore, candidateMovie, likedCastSets, favoriteCastSets, dislikedCastSets }) {
    const candidateCast = this.#tokenizeCast(candidateMovie.cast_text);
    const likedOverlap = this.#averageOverlap(candidateCast, likedCastSets);
    const favoriteOverlap = this.#averageOverlap(candidateCast, favoriteCastSets);
    const dislikedOverlap = this.#averageOverlap(candidateCast, dislikedCastSets);
    return baseScore + (likedOverlap * 0.18) + (favoriteOverlap * 0.32) - (dislikedOverlap * 0.22);
  }

  #averageOverlap(candidateCast, castSets) {
    if (!castSets.length || !candidateCast.size) return 0;
    let total = 0;
    castSets.forEach((castSet) => {
      if (!castSet.size) return;
      let shared = 0;
      castSet.forEach((castName) => {
        if (candidateCast.has(castName)) shared += 1;
      });
      total += shared / castSet.size;
    });
    return total / castSets.length;
  }
}