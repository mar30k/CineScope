import type { KeyboardEvent } from 'react'

interface SearchBarProps {
  query: string
  onChange: (value: string) => void
  onSubmit: () => void
  disabled?: boolean
}

export function SearchBar({ query, onChange, onSubmit, disabled = false }: SearchBarProps) {
  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Enter') {
      event.preventDefault()
      onSubmit()
    }
  }

  return (
    <div className="search-bar">
      <input
        type="search"
        className="search-input"
        placeholder="Search movies by title"
        value={query}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        aria-label="Movie title search"
      />
      <button type="button" className="search-button" onClick={onSubmit} disabled={disabled || query.length === 0}>
        Search
      </button>
    </div>
  )
}
