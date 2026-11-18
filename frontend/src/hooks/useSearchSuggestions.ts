import { useEffect, useState } from 'react'
import type { MovieSearchResult } from '../types'
import { api } from '../api/client'

export type SearchSuggestionsState =
  | { kind: 'idle' }
  | { kind: 'loading'; query: string }
  | { kind: 'ready'; query: string; results: MovieSearchResult[] }
  | { kind: 'error'; query: string; message: string }

const MIN_QUERY_LENGTH = 2
const SUGGESTION_LIMIT = 8
const DEBOUNCE_MS = 275

function isAbortError(error: unknown) {
  return error instanceof DOMException && error.name === 'AbortError'
}

export function useSearchSuggestions(query: string): SearchSuggestionsState {
  const [state, setState] = useState<SearchSuggestionsState>({ kind: 'idle' })

  useEffect(() => {
    const term = query.trim()
    if (term.length < MIN_QUERY_LENGTH) {
      setState({ kind: 'idle' })
      return
    }

    setState({ kind: 'loading', query: term })

    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      api
        .searchMovies(term, SUGGESTION_LIMIT, controller.signal)
        .then((response) => {
          if (!controller.signal.aborted) {
            setState({ kind: 'ready', query: term, results: response.results })
          }
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted || isAbortError(error)) {
            return
          }
          setState({
            kind: 'error',
            query: term,
            message:
              error instanceof Error
                ? error.message
                : 'Suggestions are unavailable right now.',
          })
        })
    }, DEBOUNCE_MS)

    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [query])

  return state
}
