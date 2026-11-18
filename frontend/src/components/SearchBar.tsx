import {
  useEffect,
  useRef,
  useState,
  type FocusEvent,
  type KeyboardEvent,
} from 'react'
import type { MovieSearchResult } from '../types'
import type { SearchSuggestionsState } from '../hooks/useSearchSuggestions'
import { MoviePoster } from './MoviePoster'

const LISTBOX_ID = 'search-suggestion-listbox'
const MIN_QUERY_LENGTH = 2

interface SearchBarProps {
  query: string
  onChange: (value: string) => void
  onSubmit: () => void
  onClear?: () => void
  onSuggestionSelect: (movie: MovieSearchResult) => void
  suggestions: SearchSuggestionsState
  disabled?: boolean
}

function suggestionOptionId(index: number) {
  return `search-suggestion-${index}`
}

export function SearchBar({
  query,
  onChange,
  onSubmit,
  onClear,
  onSuggestionSelect,
  suggestions,
  disabled = false,
}: SearchBarProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [activeIndex, setActiveIndex] = useState(-1)
  const rootRef = useRef<HTMLDivElement>(null)

  const term = query.trim()
  const currentSuggestions =
    suggestions.kind !== 'idle' && suggestions.query === term
      ? suggestions
      : null
  const results =
    currentSuggestions?.kind === 'ready' ? currentSuggestions.results : []
  const isDropdownOpen =
    isOpen && term.length >= MIN_QUERY_LENGTH && currentSuggestions !== null
  const activeSuggestion =
    isDropdownOpen && activeIndex >= 0 ? results[activeIndex] : undefined

  useEffect(() => {
    if (term.length < MIN_QUERY_LENGTH || suggestions.kind === 'idle') {
      setIsOpen(false)
      setActiveIndex(-1)
      return
    }

    if (suggestions.kind === 'loading') {
      setIsOpen(true)
    }
    setActiveIndex(-1)
  }, [term, suggestions.kind])

  useEffect(() => {
    function handlePointerDown(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    document.addEventListener('pointerdown', handlePointerDown)
    return () => document.removeEventListener('pointerdown', handlePointerDown)
  }, [])

  function closeSuggestions() {
    setIsOpen(false)
    setActiveIndex(-1)
  }

  function chooseSuggestion(movie: MovieSearchResult) {
    closeSuggestions()
    onSuggestionSelect(movie)
  }

  function submitSearch() {
    closeSuggestions()
    onSubmit()
  }

  function handleFocus() {
    if (currentSuggestions !== null && term.length >= MIN_QUERY_LENGTH) {
      setIsOpen(true)
    }
  }

  function handleBlur(event: FocusEvent<HTMLDivElement>) {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
      closeSuggestions()
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Escape' && isDropdownOpen) {
      event.preventDefault()
      closeSuggestions()
      return
    }

    if (event.key === 'ArrowDown' && results.length > 0) {
      event.preventDefault()
      setIsOpen(true)
      setActiveIndex((current) =>
        current < 0 || current >= results.length - 1 ? 0 : current + 1
      )
      return
    }

    if (event.key === 'ArrowUp' && results.length > 0) {
      event.preventDefault()
      setIsOpen(true)
      setActiveIndex((current) =>
        current <= 0 ? results.length - 1 : current - 1
      )
      return
    }

    if (event.key === 'Enter') {
      event.preventDefault()
      if (activeSuggestion) {
        chooseSuggestion(activeSuggestion)
        return
      }
      submitSearch()
    }
  }

  return (
    <div className="search-bar" ref={rootRef} onBlur={handleBlur}>
      <div className="search-field">
        <svg
          className="search-icon"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          aria-hidden="true"
        >
          <circle cx="11" cy="11" r="6.5" />
          <path d="m16 16 4 4" strokeLinecap="round" />
        </svg>
        <input
          type="search"
          className="search-input"
          placeholder="Search movies by title"
          value={query}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={handleFocus}
          disabled={disabled}
          autoComplete="off"
          role="combobox"
          aria-label="Movie title search"
          aria-autocomplete="list"
          aria-haspopup="listbox"
          aria-expanded={isDropdownOpen}
          aria-controls={
            currentSuggestions?.kind === 'ready' && results.length > 0
              ? LISTBOX_ID
              : undefined
          }
          aria-activedescendant={
            activeSuggestion
              ? suggestionOptionId(activeIndex)
              : undefined
          }
          aria-busy={currentSuggestions?.kind === 'loading'}
        />
        {query.length > 0 && onClear && (
          <button
            type="button"
            className="search-clear"
            onClick={onClear}
            disabled={disabled}
            aria-label="Clear search"
          >
            <span aria-hidden="true">×</span>
          </button>
        )}

        {isDropdownOpen && (
          <div className="search-suggestions">
            {currentSuggestions.kind === 'loading' && (
              <div className="search-suggestion-status" role="status">
                <span className="status-spinner" aria-hidden="true" />
                Searching the catalogue…
              </div>
            )}

            {currentSuggestions.kind === 'error' && (
              <div
                className="search-suggestion-status search-suggestion-error"
                role="status"
              >
                {currentSuggestions.message}
              </div>
            )}

            {currentSuggestions.kind === 'ready' &&
              currentSuggestions.results.length === 0 && (
                <div className="search-suggestion-status" role="status">
                  No matching films. Try another title or spelling.
                </div>
              )}

            {currentSuggestions.kind === 'ready' && results.length > 0 && (
              <ul
                className="search-suggestion-list"
                id={LISTBOX_ID}
                role="listbox"
                aria-label="Movie suggestions"
              >
                {results.map((movie, index) => (
                  <li
                    key={movie.movie_id}
                    id={suggestionOptionId(index)}
                    className={`search-suggestion${
                      index === activeIndex ? ' active' : ''
                    }`}
                    role="option"
                    aria-selected={index === activeIndex}
                    onMouseDown={(event) => event.preventDefault()}
                    onMouseEnter={() => setActiveIndex(index)}
                    onClick={() => chooseSuggestion(movie)}
                  >
                    <MoviePoster
                      title={movie.title}
                      src={movie.image?.poster_url ?? null}
                      variant="suggestion"
                    />
                    <span className="search-suggestion-copy">
                      <span className="search-suggestion-title-row">
                        <span
                          className="search-suggestion-title"
                          title={movie.title}
                        >
                          {movie.title}
                        </span>
                        {movie.year !== null && (
                          <span className="search-suggestion-year">
                            {movie.year}
                          </span>
                        )}
                      </span>
                      <span className="search-suggestion-genres">
                        {movie.genres.length > 0
                          ? movie.genres.slice(0, 3).join(' · ')
                          : 'Genre unavailable'}
                      </span>
                    </span>
                  </li>
                ))}
              </ul>
            )}

            <div className="search-suggestion-hint" aria-hidden="true">
              <span>↑↓ navigate</span>
              <span>Enter select</span>
              <span>Esc close</span>
            </div>
          </div>
        )}
      </div>
      <button
        type="button"
        className="search-button"
        onClick={submitSearch}
        disabled={disabled || query.length === 0}
      >
        Search
      </button>
    </div>
  )
}
