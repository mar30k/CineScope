import { useState, useCallback } from 'react'
import type { MovieSearchResponse, MovieSearchResult } from '../types'
import { api } from '../api/client'
import { SearchBar } from '../components/SearchBar'
import { MovieCard } from '../components/MovieCard'
import { StatusChip } from '../components/StatusChip'
import { ErrorMessage } from '../components/ErrorMessage'
import { EmptyState } from '../components/EmptyState'

type PageState =
  | { kind: 'idle' }
  | { kind: 'searching' }
  | { kind: 'results'; response: MovieSearchResponse }
  | { kind: 'error'; message: string }

export function SearchPage() {
  const [query, setQuery] = useState('')
  const [pageState, setPageState] = useState<PageState>({ kind: 'idle' })

  const runSearch = useCallback(async (searchQuery: string) => {
    setPageState({ kind: 'searching' })
    try {
      const response = await api.searchMovies(searchQuery, 10)
      setPageState({ kind: 'results', response })
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to reach the movie search service.'
      setPageState({ kind: 'error', message })
    }
  }, [])

  function handleSubmit() {
    const trimmed = query.trim()
    if (trimmed.length < 2) {
      setPageState({ kind: 'error', message: 'Please enter at least 2 characters.' })
      return
    }
    runSearch(trimmed)
  }

  function selectMovie(_movie: MovieSearchResult) {
    // Milestone 1: details are not implemented yet.
    // Selecting a movie currently keeps the list usable during search.
    // Reserved for Milestone 2: open details + recommendations for this id.
    setPageState((current) => {
      if (current.kind !== 'results') {
        return current
      }
      return { kind: 'results', response: current.response }
    })
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
      </section>
    </main>
  )
}
