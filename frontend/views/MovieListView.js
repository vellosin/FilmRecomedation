export class MovieListView {
  constructor(containerSelector) {
    this.container = document.querySelector(containerSelector);
  }

  render(movies, { onSelect, onAction, profile } = {}) {
    if (!this.container) {
      return;
    }

    if (!movies || movies.length === 0) {
      this.container.innerHTML = '<div class="movie-card">Nenhum filme encontrado.</div>';
      return;
    }

    this.container.innerHTML = movies.map((movie) => {
      const state = this.#profileState(movie, profile);
      return `
      <article class="movie-card" data-movie-id="${movie.movie_id}">
        <div>
          <h3>${movie.title}</h3>
          <p class="movie-meta">${movie.release_year || 'Ano desconhecido'} · Nota ${Number(movie.vote_average || 0).toFixed(1)}</p>
        </div>
        <p class="movie-overview">${this.#truncate(movie.overview)}</p>
        ${movie.score ? `<span class="pill-score">afinidade ${Number(movie.score).toFixed(3)}</span>` : ''}
        <div class="movie-card-actions">
          <button type="button" data-role="details">Ver detalhes</button>
          <button type="button" data-role="like" class="${state.likeClass}">${state.likeLabel}</button>
          <button type="button" data-role="favorite" class="${state.favoriteClass}">${state.favoriteLabel}</button>
          <button type="button" data-role="dislike" class="ghost ${state.dislikeClass}">${state.dislikeLabel}</button>
        </div>
      </article>
    `;
    }).join('');

    this.container.querySelectorAll('[data-movie-id]').forEach((card) => {
      card.querySelector('[data-role="details"]').addEventListener('click', () => {
        onSelect?.(Number(card.dataset.movieId));
      });
      card.querySelector('[data-role="like"]').addEventListener('click', () => {
        onAction?.(Number(card.dataset.movieId), 'like');
      });
      card.querySelector('[data-role="favorite"]').addEventListener('click', () => {
        onAction?.(Number(card.dataset.movieId), 'favorite');
      });
      card.querySelector('[data-role="dislike"]').addEventListener('click', () => {
        onAction?.(Number(card.dataset.movieId), 'dislike');
      });
    });
  }

  renderMessage(message) {
    if (!this.container) {
      return;
    }
    this.container.innerHTML = `<div class="movie-card">${message}</div>`;
  }

  renderLoading(message = 'Carregando...') {
    if (!this.container) {
      return;
    }
    this.container.innerHTML = `
      <div class="movie-card loading-card" aria-live="polite">
        <div class="loading-indicator" aria-hidden="true"></div>
        <p>${message}</p>
      </div>
    `;
  }

  #profileState(movie, profile = { likes: [], dislikes: [], favorites: [] }) {
    const movieId = Number(movie.movie_id);
    const isFavorite = profile.favorites?.includes(movieId);
    const isLiked = profile.likes?.includes(movieId) || isFavorite;
    const isDisliked = profile.dislikes?.includes(movieId);

    return {
      likeLabel: isLiked ? 'Curtido' : 'Curtir',
      favoriteLabel: isFavorite ? 'Favorito' : 'Favoritar',
      dislikeLabel: isDisliked ? 'Descartado' : 'Descartar',
      likeClass: isLiked ? 'button-active' : '',
      favoriteClass: isFavorite ? 'button-active' : '',
      dislikeClass: isDisliked ? 'button-active-danger' : '',
    };
  }

  #truncate(text = '', size = 140) {
    if (text.length <= size) {
      return text || 'Sem overview disponivel.';
    }
    return `${text.slice(0, size).trim()}...`;
  }
}
