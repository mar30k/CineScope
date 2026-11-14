import type { MovieSearchResult, MovieDetails, RecommendationResponse } from '../types'
import { BackButton } from './BackButton'

interface MovieDetailsViewProps {
  movie: MovieSearchResult
  details: MovieDetails
  savedResults: { query: string; count: number; results: Array<{ movie_id: number; title: string; year: number | null }> }
  onBack: () => void
  recommendations?: RecommendationResponse
}

function optionalRuntime(runtime: number | null) {
  if (runtime == null) {
    return null
  }
  return `${runtime} min`
}

function optionalRating(rating: number | null, count: number | null) {
  if (rating == null) {
    return null
  }
  const countText = count != null ? ` · ${count.toLocaleString()}` : ''
  return `${rating.toFixed(1)} / 10${countText}`
}

function formatScore(score: number) {
  return score.toFixed(3)
}

function formatCoverage(coverage: number) {
  // Coverage is always present in [0,1] on this endpoint.
  return coverage.toFixed(3)
}

function formatAgreement(agreement: number | null) {
  if (agreement == null) {
    return 'No shared feature blocks'
  }
  return agreement.toFixed(3)
}

function recommendationList(recommendations: RecommendationResponse) {
  if (recommendations.recommendations.length === 0) {
    return (
      <p className="recommendations-empty">
        No recommendations are available for this movie.
      </p>
    )
  }

  return (
    <ol className="recommendations">
      {recommendations.recommendations.map((item) => (
        <li key={item.rank} className="recommendation">
          <div className="recommendation-rank">
            <span className="recommendation-rank-label">#{item.rank}</span>
            <span className="recommendation-movie-id">ID {item.movie_id}</span>
          </div>
          <div className="recommendation-main">
            <span className="recommendation-title">{item.title}</span>
            <span className="recommendation-meta">
              {item.year != null ? String(item.year) : 'No year'}{' · '}ID {item.movie_id}
            </span>
          </div>
          <dl className="recommendation-score">
            <div className="recommendation-field">
              <dt>Score</dt>
              <dd>{formatScore(item.score)}</dd>
            </div>
            <div className="recommendation-field">
              <dt>Coverage</dt>
              <dd>{formatCoverage(item.coverage)}</dd>
            </div>
            <div className="recommendation-field">
              <dt>Agreement</dt>
              <dd>{formatAgreement(item.agreement)}</dd>
            </div>
          </dl>
        </li>
      ))}
    </ol>
  )
}

export function MovieDetailsView({
  movie,
  details,
  savedResults,
  onBack,
  recommendations,
}: MovieDetailsViewProps) {
  const genreList =
    details.genres.length > 0 ? details.genres.join(', ') : null

  return (
    <div className={`details ${savedResults.results.length > 0 ? 'has-back' : ''}`}>
      {savedResults.results.length > 0 && (
        <BackButton onClick={onBack}>Back to results</BackButton>
      )}

      <div className="details-identity">
        <h2 className="details-title">{details.title}</h2>
        <p className="details-subtitle">
          TMDB ID {details.movie_id}
        </p>
      </div>

      <dl className="details-list">
        <div className="details-row">
          <dt>Year</dt>
          <dd>{details.year != null ? String(details.year) : 'No year recorded'}</dd>
        </div>

        {details.genres.length > 0 && (
          <div className="details-row">
            <dt>Genres</dt>
            <dd>{details.genres.join(', ')}</dd>
          </div>
        )}

        {details.director.length > 0 && (
          <div className="details-row">
            <dt>Director</dt>
            <dd>{details.director.join(', ')}</dd>
          </div>
        )}

        {details.runtime != null && (
          <div className="details-row">
            <dt>Runtime</dt>
            <dd>{optionalRuntime(details.runtime)}</dd>
          </div>
        )}

        {details.vote_average != null && (
          <div className="details-row">
            <dt>Rating</dt>
            <dd>{optionalRating(details.vote_average, details.vote_count)}</dd>
          </div>
        )}

        {details.overview != null && details.overview.trim().length > 0 && (
          <div className="details-row">
            <dt>Overview</dt>
            <dd className="details-overview">{details.overview}</dd>
          </div>
        )}
      </dl>

      {genreList == null && details.director.length === 0 && details.overview == null && (
        <p className="details-empty">
          No additional details are available for this movie in the catalogue.
        </p>
      )}

      {recommendations && (
        <section className="recommendations-section" aria-labelledby="recommendations-heading">
          <h3 id="recommendations-heading" className="recommendations-heading">
            Recommended movies
          </h3>
          <p className="recommendations-intro">
            These movies are ranked by content agreement with this one. The score is
            adjusted by how much of the configured feature set the two movies actually
            share, so a smaller score can mean less evidence rather than a worse fit.
            It is not a probability that you will like the movie.
          </p>
          {recommendationList(recommendations)}
        </section>
      )}

      <p className="details-identity details-id">
        Selected: {movie.title}
        {' · '}
        {movie.year != null ? String(movie.year) : 'No year'}
        {' · '}
        ID {movie.movie_id}
      </p>
    </div>
  )
}
