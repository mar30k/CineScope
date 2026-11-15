import type { RecommendationItem, SelectedMovieInfo } from '../types'
import { useMovieDetails } from '../hooks/useMovieDetails'
import { MoviePoster } from './MoviePoster'

interface RecommendationCardProps {
  item: RecommendationItem
  onSelect: (item: SelectedMovieInfo) => void
}

function formatScore(score: number) {
  return score.toFixed(3)
}

function formatCoverage(coverage: number) {
  return coverage.toFixed(3)
}

function formatAgreement(agreement: number | null) {
  return agreement == null ? 'No shared blocks' : agreement.toFixed(3)
}

export function RecommendationCard({
  item,
  onSelect,
}: RecommendationCardProps) {
  const detailsState = useMovieDetails(item.movie_id)
  const details = detailsState.kind === 'ready' ? detailsState.details : null

  const year = item.year ?? details?.year ?? null
  const runtime = details?.runtime ?? null
  const genres = details?.genres ?? []
  const director = details?.director ?? []
  const overview = details?.overview?.trim() ?? ''
  const castNames = (details?.cast ?? [])
    .slice(0, 5)
    .map((member) => member.name)
    .join(' · ')
  const posterUrl = item.image?.poster_url ?? details?.image?.poster_url ?? null

  const metaParts: string[] = []
  if (year != null) {
    metaParts.push(String(year))
  }
  if (runtime != null) {
    metaParts.push(`${runtime} min`)
  }

  return (
    <button
      type="button"
      className="recommendation-card"
      onClick={() => onSelect(item)}
      aria-label={`View details for ${item.title}${
        item.year != null ? ` (${item.year})` : ''
      }, ranked ${item.rank}, recommendation score ${formatScore(item.score)}`}
    >
      <MoviePoster
        title={item.title}
        src={posterUrl}
        variant="recommendation"
      />

      <span className="recommendation-card-main">
        <span className="recommendation-card-header">
          <span className="recommendation-rank">Rank #{item.rank}</span>
          <span className="recommendation-open-hint">
            View details <span aria-hidden="true">→</span>
          </span>
        </span>

        <span className="recommendation-title">{item.title}</span>

        {metaParts.length > 0 && (
          <span className="recommendation-meta">{metaParts.join(' · ')}</span>
        )}

        {genres.length > 0 && (
          <span className="recommendation-genres">
            {genres.map((genre) => (
              <span key={genre} className="recommendation-genre">
                {genre}
              </span>
            ))}
          </span>
        )}

        {detailsState.kind === 'loading' && (
          <span className="recommendation-detail-note">
            Loading genres and overview…
          </span>
        )}

        {detailsState.kind === 'error' && (
          <span className="recommendation-detail-note">
            Additional details are unavailable for this movie.
          </span>
        )}

        {detailsState.kind === 'ready' && overview.length > 0 && (
          <span className="recommendation-overview">{overview}</span>
        )}

        {detailsState.kind === 'ready' && overview.length === 0 && (
          <span className="recommendation-detail-note">
            No overview is available for this movie.
          </span>
        )}

        {director.length > 0 && (
          <span className="recommendation-director">
            Directed by {director.join(', ')}
          </span>
        )}

        {castNames.length > 0 && (
          <span className="recommendation-cast">
            <span className="recommendation-cast-label">Cast</span>
            {castNames}
          </span>
        )}

        <span className="recommendation-card-footer">
          <span className="recommendation-score-block">
            <span className="recommendation-score-label">Score</span>
            <span className="recommendation-score-value">
              {formatScore(item.score)}
            </span>
          </span>

          <span className="recommendation-technical">
            <span>Coverage {formatCoverage(item.coverage)}</span>
            <span>Agreement {formatAgreement(item.agreement)}</span>
          </span>
        </span>
      </span>
    </button>
  )
}
