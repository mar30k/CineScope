import type { KeyboardEvent } from 'react'

interface SearchBarProps {
  query: string
  onChange: (value: string) => void
  onSubmit: () => void
  onClear?: () => void
  disabled?: boolean
}

export function SearchBar({
  query,
  onChange,
  onSubmit,
  onClear,
  disabled = false,
}: SearchBarProps) {
  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Enter') {
      event.preventDefault()
      onSubmit()
    }
  }

  return (
    <div className="search-bar">
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
          disabled={disabled}
          autoComplete="off"
          aria-label="Movie title search"
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
      </div>
      <button
        type="button"
        className="search-button"
        onClick={onSubmit}
        disabled={disabled || query.length === 0}
      >
        Search
      </button>
    </div>
  )
}
