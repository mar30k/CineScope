import { useCallback, useEffect, useRef, useState } from 'react'
import type {
  MovieSearchResponse,
  MovieSearchResult,
  MovieDetails,
  RecommendationResponse,
  SelectedMovieInfo,
} from '../types'
import { api } from '../api/client'
import { fetchMovieDetails } from '../api/movieDetailsCache'

interface MovieDetailsEntry {
  movie: SelectedMovieInfo
  details: MovieDetails
}

interface LoadedMovieEntry extends MovieDetailsEntry {
  recommendations: RecommendationResponse
}

type MovieView =
  | { kind: 'details-loading'; movie: SelectedMovieInfo }
  | { kind: 'details-error'; movie: SelectedMovieInfo; message: string }
  | { kind: 'recommendations-loading'; entry: MovieDetailsEntry }
  | { kind: 'recommendations'; entry: LoadedMovieEntry }
  | { kind: 'recommendations-error'; entry: MovieDetailsEntry; message: string }

type SelectionOrigin = 'search' | 'recommendation' | 'history'

function toSelectedMovieInfo(movie: MovieSearchResult): SelectedMovieInfo {
  return {
    movie_id: movie.movie_id,
    title: movie.title,
    year: movie.year,
  }
}

function isAbortError(error: unknown) {
  return error instanceof DOMException && error.name === 'AbortError'
}

function describeError(error: unknown, fallback: string) {
  return error instanceof Error && error.message.length > 0
    ? error.message
    : fallback
}

export function useMovieSelection() {
  const [savedResults, setSavedResults] = useState<MovieSearchResponse | null>(null)
  const [view, setView] = useState<MovieView | null>(null)
  const [history, setHistory] = useState<SelectedMovieInfo[]>([])

  const historyRef = useRef<SelectedMovieInfo[]>([])
  const currentMovieRef = useRef<SelectedMovieInfo | null>(null)
  const requestRef = useRef<{ sequence: number; controller: AbortController }>({
    sequence: 0,
    controller: new AbortController(),
  })

  const replaceHistory = useCallback((next: SelectedMovieInfo[]) => {
    historyRef.current = next
    setHistory(next)
  }, [])

  const beginRequest = useCallback(() => {
    requestRef.current.controller.abort()

    const controller = new AbortController()
    const sequence = requestRef.current.sequence + 1
    requestRef.current = { sequence, controller }

    return { sequence, signal: controller.signal }
  }, [])

  useEffect(() => {
    return () => {
      requestRef.current.controller.abort()
      requestRef.current.sequence += 1
    }
  }, [])

  const loadMovie = useCallback(
    async (movie: SelectedMovieInfo, origin: SelectionOrigin) => {
      if (origin === 'search') {
        replaceHistory([])
      } else if (origin === 'recommendation') {
        const current = currentMovieRef.current
        const history = historyRef.current
        const last = history[history.length - 1]

        if (current && current.movie_id !== movie.movie_id && last?.movie_id !== current.movie_id) {
          replaceHistory([...history, current])
        }
      }

      const { sequence, signal } = beginRequest()
      setView({ kind: 'details-loading', movie })

      let details: MovieDetails
      try {
        details = await fetchMovieDetails(movie.movie_id)
      } catch (error) {
        if (sequence !== requestRef.current.sequence || isAbortError(error)) {
          return
        }
        setView({
          kind: 'details-error',
          movie,
          message: describeError(error, 'Failed to load movie details.'),
        })
        return
      }

      if (signal.aborted || sequence !== requestRef.current.sequence) {
        return
      }

      currentMovieRef.current = movie
      const entry: MovieDetailsEntry = { movie, details }
      setView({ kind: 'recommendations-loading', entry })

      try {
        const recommendations = await api.getRecommendations(movie.movie_id, 10, signal)
        if (signal.aborted || sequence !== requestRef.current.sequence) {
          return
        }
        setView({
          kind: 'recommendations',
          entry: { ...entry, recommendations },
        })
      } catch (error) {
        if (signal.aborted || sequence !== requestRef.current.sequence || isAbortError(error)) {
          return
        }
        setView({
          kind: 'recommendations-error',
          entry,
          message: describeError(error, 'Failed to load recommendations.'),
        })
      }
    },
    [beginRequest, replaceHistory]
  )

  const selectSearchResult = useCallback(
    (movie: MovieSearchResult) => {
      void loadMovie(toSelectedMovieInfo(movie), 'search')
    },
    [loadMovie]
  )

  const selectRecommendation = useCallback(
    (movie: SelectedMovieInfo) => {
      void loadMovie(movie, 'recommendation')
    },
    [loadMovie]
  )

  const goBack = useCallback(() => {
    const history = historyRef.current
    const previous = history[history.length - 1]
    if (!previous) {
      return false
    }

    replaceHistory(history.slice(0, -1))
    void loadMovie(previous, 'history')
    return true
  }, [loadMovie, replaceHistory])

  const backToResults = useCallback(() => {
    beginRequest()
    replaceHistory([])
    currentMovieRef.current = null
    setView(null)
  }, [beginRequest, replaceHistory])

  return {
    view,
    savedResults,
    setSavedResults,
    selectSearchResult,
    selectRecommendation,
    goBack,
    backToResults,
    history,
    hasHistory: history.length > 0,
  }
}
