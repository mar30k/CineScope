import { useCallback, useState } from 'react'
import type { MovieSearchResponse } from '../types'
import { api } from '../api/client'
import { useMovieSelection } from '../hooks/useMovieSelection'
import { SearchBar } from '../components/SearchBar'
import { MovieCard } from '../components/MovieCard'
import { ErrorMessage } from '../components/ErrorMessage'
import { EmptyState } from '../components/EmptyState'
import { DiscoveryToolbar } from '../components/DiscoveryToolbar'
import { MovieDetailsView } from '../components/MovieDetailsView'
import tmdbLogo from '../assets/tmdb-logo.svg'

type SearchState =
  | { kind: 'idle' }
  | { kind: 'searching' }
  | { kind: 'results'; response: MovieSearchResponse }
  | { kind: 'search-error'; message: string }

function searchResultsSkeleton() {
  return (
    <div className="search-loading" role="status">
      <span className="sr-only">Searching the catalogue…</span>
      <div className="movie-grid" aria-hidden="true">
        {[0, 1, 2, 3].map((index) => (
          <div className="movie-card movie-card-skeleton" key={index}>
            <span className="movie-card-poster skeleton-block" />
            <span className="movie-card-body">
              <span className="skeleton-line skeleton-line-title" />
              <span className="skeleton-line skeleton-line-medium" />
              <span className="skeleton-line skeleton-line-short" />
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function SearchPage() {
  const [query, setQuery] = useState('')
  const [searchState, setSearchState] = useState<SearchState>({ kind: 'idle' })

  const {
    view,
    savedResults,
    setSavedResults,
    selectSearchResult,
    selectRecommendation,
    goBack,
    backToResults,
    history,
  } = useMovieSelection()

  const runSearch = useCallback(
    async (searchQuery: string) => {
      setSearchState({ kind: 'searching' })
      backToResults()
      try {
        const response = await api.searchMovies(searchQuery, 10)
        setSearchState({ kind: 'results', response })
        setSavedResults(response)
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : 'Failed to reach the movie search service.'
        setSearchState({ kind: 'search-error', message })
      }
    },
    [backToResults, setSavedResults]
  )

  const handleClear = useCallback(() => {
    setQuery('')
    setSearchState({ kind: 'idle' })
    setSavedResults(null)
    backToResults()
  }, [backToResults, setSavedResults])

  function handleSubmit() {
    const trimmed = query.trim()
    if (trimmed.length < 2) {
      setSearchState({
        kind: 'search-error',
        message: 'Please enter at least 2 characters.',
      })
      return
    }
    runSearch(trimmed)
  }

  const currentMovie =
    view === null ? null : 'movie' in view ? view.movie : view.entry.movie

  return (
    <main className="app">
      <header className={`app-header ${view !== null ? 'app-header-compact' : ''}`}>
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            C
          </span>
          <div className="brand-copy">
            <h1 className="brand-name">CineScope</h1>
            <p className="brand-tagline">
              {view === null
                ? 'Follow the connections between films.'
                : 'A KNN-powered discovery workspace.'}
            </p>
          </div>
        </div>

        {view !== null && (
          <div
            className="header-context"
            title={
              savedResults?.query ? `Search: ${savedResults.query}` : undefined
            }
          >
            {savedResults?.query ? (
              <>
                Search <strong>“{savedResults.query}”</strong>
              </>
            ) : (
              'Discovery'
            )}
          </div>
        )}
      </header>

      {view === null ? (
        <div className="search-workspace">
          <section className="search-hero" aria-labelledby="search-heading">
            <p className="eyebrow">Content-based discovery</p>
            <h2 id="search-heading" className="search-title">
              Start with a film you know.
            </h2>
            <p className="search-subtitle">
              Search the catalogue, explore its details, then follow
              recommendations deeper into what connects.
            </p>
            <SearchBar
              query={query}
              onChange={setQuery}
              onSubmit={handleSubmit}
              onClear={handleClear}
              disabled={searchState.kind === 'searching'}
            />
          </section>

          <section
            className="search-results-area"
            aria-live="polite"
            aria-busy={searchState.kind === 'searching'}
          >
            {searchState.kind === 'idle' && (
              <div className="search-empty">
                <div className="search-empty-art" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </div>
                <h3>Every film is a doorway.</h3>
                <p>
                  Search for a title to open its details and follow
                  recommendations deeper into the catalogue.
                </p>
              </div>
            )}

            {searchState.kind === 'searching' && searchResultsSkeleton()}

            {searchState.kind === 'search-error' && (
              <ErrorMessage message={searchState.message} />
            )}

            {searchState.kind === 'results' &&
              (searchState.response.results.length === 0 ? (
                <EmptyState query={searchState.response.query} />
              ) : (
                <>
                  <div className="results-header">
                    <div>
                      <p className="eyebrow">Search results</p>
                      <h2 className="results-title">
                        Results for “{searchState.response.query}”
                      </h2>
                    </div>
                    <span className="results-count">
                      {searchState.response.count}{' '}
                      {searchState.response.count === 1 ? 'film' : 'films'}
                    </span>
                  </div>

                  <ul className="movie-grid">
                    {searchState.response.results.map((movie) => (
                      <li key={movie.movie_id}>
                        <MovieCard
                          movie={movie}
                          selected={false}
                          onClick={() => selectSearchResult(movie)}
                        />
                      </li>
                    ))}
                  </ul>
                </>
              ))}
          </section>
        </div>
      ) : (
        <div className="discovery-workspace">
          {currentMovie && (
            <DiscoveryToolbar
              currentMovie={currentMovie}
              history={history}
              onBackToResults={backToResults}
              onPreviousMovie={goBack}
            />
          )}

          {view.kind === 'details-loading' && (
            <div className="discovery-state-card" role="status">
              <span className="loading-pulse" aria-hidden="true" />
              <h2>Loading {view.movie.title}</h2>
              <p>
                Fetching film details and finding close neighbours in the
                catalogue.
              </p>
            </div>
          )}

          {view.kind === 'details-error' && (
            <div className="discovery-state-card">
              <ErrorMessage message={view.message} />
              <p className="discovery-state-hint">
                Return to your search results or choose another movie to keep
                exploring.
              </p>
            </div>
          )}

          {(view.kind === 'recommendations-loading' ||
            view.kind === 'recommendations' ||
            view.kind === 'recommendations-error') && (
            <MovieDetailsView
              details={view.entry.details}
              recommendations={
                view.kind === 'recommendations'
                  ? view.entry.recommendations
                  : undefined
              }
              isLoadingRecommendations={
                view.kind === 'recommendations-loading'
              }
              recommendationsError={
                view.kind === 'recommendations-error'
                  ? view.message
                  : undefined
              }
              onSelectRecommendation={selectRecommendation}
            />
          )}
        </div>
      )}

      <footer className="app-footer">
        <div className="app-footer-inner">
          <span className="app-footer-label">Data attribution</span>
          <a
            className="tmdb-link"
            href="https://www.themoviedb.org/"
            target="_blank"
            rel="noreferrer"
          >
            <img
              className="tmdb-logo"
              src={tmdbLogo}
              alt="The Movie Database (TMDB)"
              width={273}
              height={36}
              loading="lazy"
            />
          </a>
          <p>
            This product uses the TMDB API but is not endorsed or certified by
            TMDB.
          </p>
        </div>
      </footer>
    </main>
  )
}
