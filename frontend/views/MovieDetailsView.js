export class MovieDetailsView {
  constructor(containerSelector) {
    this.container = document.querySelector(containerSelector);
    this.selectedMovie = null;
  }

  render(movie, { onAction } = {}) {
    this.selectedMovie = movie;
    this.container.classList.remove('empty-state');
    this.container.innerHTML = `
      <div>
        <h3>${movie.title}</h3>
        <p class="movie-meta">${movie.release_year || 'Ano desconhecido'} · Nota ${Number(movie.vote_average || 0).toFixed(1)} · Popularidade ${Number(movie.popularity || 0).toFixed(1)}</p>
        <div class="movie-tags">
          ${this.#genreTags(movie.genres_text)}
        </div>
        <p class="movie-overview">${movie.overview || 'Sem overview disponivel.'}</p>
      </div>
      <div class="status-actions">
        <button type="button" data-action="like">Curtir</button>
        <button type="button" data-action="favorite">Favoritar</button>
        <button type="button" data-action="dislike" class="ghost">Nao curtir</button>
      </div>
    `;

    this.container.querySelectorAll('[data-action]').forEach((button) => {
      button.addEventListener('click', () => onAction?.(movie, button.dataset.action));
    });
  }

  updateProfileLists({ likes = [], favorites = [], dislikes = [] }, allMoviesById) {
    this.#renderList('#liked-list', likes, allMoviesById);
    this.#renderList('#favorite-list', favorites, allMoviesById);
    this.#renderList('#disliked-list', dislikes, allMoviesById);
  }

  getSelectedMovie() {
    return this.selectedMovie;
  }

  #renderList(selector, ids, allMoviesById) {
    const target = document.querySelector(selector);
    if (!target) {
      return;
    }
    target.innerHTML = ids.length
      ? ids.map((id) => `<li>${allMoviesById.get(id)?.title || `Filme ${id}`}</li>`).join('')
      : '<li>Nenhum item ainda.</li>';
  }

  #genreTags(genresText = '') {
    return genresText
      .split(/\s{2,}|,/)
      .map((item) => item.trim())
      .filter(Boolean)
      .slice(0, 6)
      .map((item) => `<span>${item}</span>`)
      .join('');
  }
}
