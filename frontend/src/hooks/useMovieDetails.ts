import { useEffect, useState } from 'react'
import type { MovieDetails } from '../types'
import {
  fetchMovieDetails,
  getCachedMovieDetails,
} from '../api/movieDetailsCache'

type MovieDetailsState =
  | { kind: 'loading' }
  | { kind: 'ready'; details: MovieDetails }
  | { kind: 'error' }

export function useMovieDetails(movieId: number): MovieDetailsState {
  const [state, setState] = useState<MovieDetailsState>(() => {
    const cached = getCachedMovieDetails(movieId)
    return cached ? { kind: 'ready', details: cached } : { kind: 'loading' }
  })

  useEffect(() => {
    const cached = getCachedMovieDetails(movieId)
    if (cached) {
      setState({ kind: 'ready', details: cached })
      return
    }

    let active = true
    setState({ kind: 'loading' })

    fetchMovieDetails(movieId)
      .then((details) => {
        if (active) {
          setState({ kind: 'ready', details })
        }
      })
      .catch(() => {
        if (active) {
          setState({ kind: 'error' })
        }
      })

    return () => {
      active = false
    }
  }, [movieId])

  return state
}
