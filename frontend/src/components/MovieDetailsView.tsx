import type {
  MovieDetails,
  RecommendationResponse,
  SelectedMovieInfo,
} from '../types'
import { ErrorMessage } from './ErrorMessage'
import { RecommendationCard } from './RecommendationCard'
import { MovieBackdrop, MoviePoster } from './MoviePoster'

interface MovieDetailsViewProps {
  details: MovieDetails
  recommendations?: RecommendationResponse
  isLoadingRecommendations?: boolean
  recommendationsError?: string
  onSelectRecommendation: (item: SelectedMovieInfo) => void
}

function formatRuntime(runtime: number) {
  if (runtime < 60) {
    return `${runtime} min`
  }

  const hours = Math.floor(runtime / 60)
  const minutes = runtime % 60
  return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`
}

function recommendationList(
  recommendations: RecommendationResponse,
  onSelect: (item: SelectedMovieInfo) => void
) {
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
        <li key={item.movie_id} className="recommendation-item">
          <RecommendationCard item={item} onSelect={onSelect} />
        </li>
      ))}
    </ol>
  )
}

function recommendationSkeleton() {
  return (
    <div className="recommendations-skeleton" role="status">
      <span className="sr-only">Loading recommendations…</span>
      <div className="recommendation-skeleton-grid" aria-hidden="true">
        {[0, 1, 2, 3].map((index) => (
          <div className="recommendation-skeleton" key={index}>
            <span className="recommendation-skeleton-poster" />
            <span className="recommendation-skeleton-body">
              <span className="skeleton-line skeleton-line-short" />
              <span className="skeleton-line skeleton-line-title" />
              <span className="skeleton-line" />
              <span className="skeleton-line skeleton-line-medium" />
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function MovieDetailsView({
  details,
  recommendations,
  isLoadingRecommendations = false,
  recommendationsError,
  onSelectRecommendation,
}: MovieDetailsViewProps) {
  const overview = details.overview?.trim() ?? ''
  const castNames = details.cast
    .slice(0, 8)
    .map((member) => member.name)
    .join(' · ')
  const hasSupplementaryDetails =
    overview.length > 0 ||
    details.genres.length > 0 ||
    details.director.length > 0 ||
    castNames.length > 0

  return (
    <div className="discovery-content">
      <article className="movie-hero">
        <MovieBackdrop
          title={details.title}
          src={details.image?.backdrop_url ?? null}
        />
        <MoviePoster
          title={details.title}
          src={details.image?.poster_url ?? null}
          variant="hero"
          loading="eager"
        />

        <div className="movie-hero-content">
          <p className="eyebrow">Now exploring</p>
          <h2 className="movie-hero-title">{details.title}</h2>

          <div className="movie-hero-meta">
            {details.year != null && <span>{details.year}</span>}
            {details.runtime != null && (
              <span>{formatRuntime(details.runtime)}</span>
            )}
            {details.vote_average != null && (
              <span className="movie-hero-rating">
                {details.vote_average.toFixed(1)} / 10
                {details.vote_count != null && (
                  <span className="movie-hero-votes">
                    {details.vote_count.toLocaleString()} votes
                  </span>
                )}
              </span>
            )}
            <span className="movie-hero-id">ID {details.movie_id}</span>
          </div>

          {details.genres.length > 0 && (
            <div className="movie-hero-genres">
              {details.genres.map((genre) => (
                <span className="genre-chip" key={genre}>
                  {genre}
                </span>
              ))}
            </div>
          )}

          {overview.length > 0 && (
            <p className="movie-hero-overview">{overview}</p>
          )}

          {details.director.length > 0 && (
            <p className="movie-hero-director">
              Directed by {details.director.join(', ')}
            </p>
          )}

          {castNames.length > 0 && (
            <p className="movie-hero-cast">
              <span className="movie-hero-cast-label">Cast</span>
              {castNames}
            </p>
          )}

          {!hasSupplementaryDetails && (
            <p className="movie-hero-note">
              Catalogue details are limited for this title.
            </p>
          )}
        </div>
      </article>

      <section
        className="recommendations-section"
        aria-labelledby="recommendations-heading"
      >
        <header className="recommendations-header">
          <p className="eyebrow">Recommendations</p>
          <h3
            id="recommendations-heading"
            className="recommendations-title"
          >
            Because you explored <span>{details.title}</span>
          </h3>
          <p className="recommendations-intro">
            CineScope uses a K-nearest-neighbours model over genre, keyword,
            overview, cast, and director features to surface films that share
            the most content with this one.
          </p>
          <p className="recommendations-note">
            Scores are content-similarity measures, not probabilities that you
            will like a movie.
          </p>
        </header>

        {isLoadingRecommendations && recommendationSkeleton()}

        {recommendationsError && (
          <ErrorMessage message={recommendationsError} />
        )}

        {recommendations &&
          recommendationList(recommendations, onSelectRecommendation)}
      </section>
    </div>
  )
}
