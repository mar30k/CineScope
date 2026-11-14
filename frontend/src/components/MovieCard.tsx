import type { MovieSearchResult } from '../types'

interface MovieCardProps {
  movie: MovieSearchResult
  selected: boolean
  onClick: () => void
}

export function MovieCard({ movie, selected, onClick }: MovieCardProps) {
  return (
    <button type="button" className={`movie-card ${selected ? 'selected' : ''}`} onClick={onClick}>
      <span className="movie-title">{movie.title}</span>
      <span className="movie-meta">
        {movie.year != null ? String(movie.year) : 'No year'}
        {' · '}
        <span className="movie-id">ID {movie.movie_id}</span>
      </span>
    </button>
  )
}
