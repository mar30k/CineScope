import type { MovieSearchResult } from '../types'
import { MoviePoster } from './MoviePoster'

interface MovieCardProps {
  movie: MovieSearchResult
  selected: boolean
  onClick: () => void
}

export function MovieCard({ movie, selected, onClick }: MovieCardProps) {
  const genres = movie.genres.slice(0, 3)
  const posterUrl = movie.image?.poster_url ?? null

  return (
    <button
      type="button"
      className={`movie-card ${selected ? 'selected' : ''}`}
      onClick={onClick}
    >
      <MoviePoster title={movie.title} src={posterUrl} variant="card" />

      <span className="movie-card-body">
        <span className="movie-card-top">
          <span className="movie-title">{movie.title}</span>
          {movie.year != null && (
            <span className="movie-year">{movie.year}</span>
          )}
        </span>

        {genres.length > 0 && (
          <span className="movie-genres">
            {genres.map((genre) => (
              <span key={genre} className="movie-genre">
                {genre}
              </span>
            ))}
          </span>
        )}

        <span className="movie-card-foot">
          <span className="movie-id">ID {movie.movie_id}</span>
          <span className="movie-open">
            Open movie <span aria-hidden="true">→</span>
          </span>
        </span>
      </span>
    </button>
  )
}
