import { useState, useCallback } from 'react'
import type {
  MovieSearchResponse,
  MovieSearchResult,
  MovieDetails,
  RecommendationResponse,
} from '../types'
import { api } from '../api/client'
import { SearchBar } from '../components/SearchBar'
import { MovieCard } from '../components/MovieCard'
import { StatusChip } from '../components/StatusChip'
import { ErrorMessage } from '../components/ErrorMessage'
import { EmptyState } from '../components/EmptyState'
import { BackButton } from '../components/BackButton'
import { MovieDetailsView } from '../components/MovieDetailsView'

type PageState =
  | { kind: 'idle' }
  | { kind: 'searching' }
  | { kind: 'results'; response: MovieSearchResponse }
  | { kind: 'error'; message: string }
  | {
      kind: 'details-loading'
      movie: MovieSearchResult
      savedResults: MovieSearchResponse
    }
  | {
      kind: 'details'
      movie: MovieSearchResult
      details: MovieDetails
      savedResults: MovieSearchResponse
    }
  | {
      kind: 'details-error'
      movie: MovieSearchResult
      message: string
      savedResults: MovieSearchResponse
    }
  | {
      kind: 'recommendations-loading'
      movie: MovieSearchResult
      details: MovieDetails
      savedResults: MovieSearchResponse
    }
  | {
      kind: 'recommendations'
      movie: MovieSearchResult
      details: MovieDetails
      recommendations: RecommendationResponse
      savedResults: MovieSearchResponse
    }
  | {
      kind: 'recommendations-error'
      movie: MovieSearchResult
      details: MovieDetails
      message: string
      savedResults: MovieSearchResponse
    }

export function SearchPage() {
  const [query, setQuery] = useState('')
  const [pageState, setPageState] = useState<PageState>({ kind: 'idle' })

  const runSearch = useCallback(
    async (searchQuery: string) => {
      setPageState({ kind: 'searching' })
      try {
        const response = await api.searchMovies(searchQuery, 10)
        setPageState({ kind: 'results', response })
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : 'Failed to reach the movie search service.'
        setPageState({ kind: 'error', message })
      }
    },
    []
  )

  function handleSubmit() {
    const trimmed = query.trim()
    if (trimmed.length < 2) {
      setPageState({ kind: 'error', message: 'Please enter at least 2 characters.' })
      return
    }
    runSearch(trimmed)
  }

  function backToResults(savedResults: MovieSearchResponse) {
    setPageState({ kind: 'results', response: savedResults })
  }

  async function selectMovie(movie: MovieSearchResult) {
    setPageState((current) => {
      if (current.kind === 'results') {
        return { kind: 'details-loading', movie, savedResults: current.response }
      }
      if (current.kind === 'details' || current.kind === 'details-error') {
        return { kind: 'details-loading', movie, savedResults: current.savedResults }
      }
      if (
        current.kind === 'recommendations' ||
        current.kind === 'recommendations-loading' ||
        current.kind === 'recommendations-error'
      ) {
        return {
          kind: 'details-loading',
          movie,
          savedResults: current.savedResults,
        }
      }
      return { kind: 'details-loading', movie, savedResults: { query: '', count: 0, results: [] } }
    })

    try {
      const details = await api.getMovieDetails(movie.movie_id)
      setPageState((current) => {
        if (current.kind !== 'details-loading') {
          return current
        }
        return { kind: 'details', movie, details, savedResults: current.savedResults }
      })
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to load movie details.'
      setPageState((current) => {
        if (current.kind !== 'details-loading') {
          return current
        }
        return { kind: 'details-error', movie, message, savedResults: current.savedResults }
      })
      return
    }

    setPageState(
      (current) => {
        if (current.kind !== 'details') {
          return current
        }
        return {
          kind: 'recommendations-loading',
          movie,
          details: current.details,
          savedResults: current.savedResults,
        }
      },
    )

    try {
      const recommendations = await api.getRecommendations(movie.movie_id, 10)
      setPageState(
        (current) => {
          if (current.kind !== 'recommendations-loading') {
            return current
          }
          return {
            kind: 'recommendations',
            movie,
            details: current.details,
            recommendations,
            savedResults: current.savedResults,
          }
        }
      )
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : 'Failed to load recommendations.'
      setPageState(
        (current) => {
          if (current.kind !== 'recommendations-loading') {
            return current
          }
          return {
            kind: 'recommendations-error',
            movie,
            details: current.details,
            message,
            savedResults: current.savedResults,
          }
        }
      )
    }
  }

  return (
    <main className="app">
      <header className="header">
        <h1 className="header-title">CineScope</h1>
        <p className="header-subtitle">Discover movies by title</p>
      </header>

      <section className="search-section">
        <SearchBar
          query={query}
          onChange={setQuery}
          onSubmit={handleSubmit}
          disabled={pageState.kind === 'searching'}
        />
      </section>

      <section className="results-section" aria-live="polite">
        {pageState.kind === 'idle' && (
          <div className="empty-state idle-state">
            <p>Search for a movie to begin.</p>
          </div>
        )}

        {pageState.kind === 'searching' && (
          <StatusChip message="Searching…" />
        )}

        {pageState.kind === 'error' && (
          <ErrorMessage message={pageState.message} />
        )}

        {pageState.kind === 'results' && (() => {
          const response = pageState.response
          if (response.results.length === 0) {
            return <EmptyState query={response.query} />
          }

          return (
            <ul className="movie-list">
              {response.results.map((movie) => (
                <li key={movie.movie_id}>
                  <MovieCard
                    movie={movie}
                    selected={false}
                    onClick={() => selectMovie(movie)}
                  />
                </li>
              ))}
            </ul>
          )
        })()}

        {pageState.kind === 'details-loading' && (
          <div className="details-shell">
            <StatusChip message={`Loading details for ${pageState.movie.title}…`} />
          </div>
        )}

        {pageState.kind === 'details-error' && (
          <div className="details-shell">
            <BackButton onClick={() => backToResults(pageState.savedResults)}>
              Back to results
            </BackButton>
            <ErrorMessage message={pageState.message} />
          </div>
        )}

        {pageState.kind === 'details' && (
          <MovieDetailsView
            movie={pageState.movie}
            details={pageState.details}
            savedResults={pageState.savedResults}
            onBack={() => backToResults(pageState.savedResults)}
          />
        )}

        {pageState.kind === 'recommendations-loading' && (
          <div className="details-shell">
            <MovieDetailsView
              movie={pageState.movie}
              details={pageState.details}
              savedResults={pageState.savedResults}
              onBack={() => backToResults(pageState.savedResults)}
            />
            <StatusChip message="Loading recommendations…" />
          </div>
        )}

        {pageState.kind === 'recommendations-error' && (
          <div className="details-shell">
            <MovieDetailsView
              movie={pageState.movie}
              details={pageState.details}
              savedResults={pageState.savedResults}
              onBack={() => backToResults(pageState.savedResults)}
            />
            <ErrorMessage message={pageState.message} />
          </div>
        )}

        {pageState.kind === 'recommendations' && (
          <MovieDetailsView
            movie={pageState.movie}
            details={pageState.details}
            savedResults={pageState.savedResults}
            onBack={() => backToResults(pageState.savedResults)}
            recommendations={pageState.recommendations}
          />
        )}
      </section>
    </main>
  )
}
