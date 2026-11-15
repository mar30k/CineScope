import type { MovieDetails } from '../types'
import { api } from './client'

const detailsCache = new Map<number, MovieDetails>()
const pendingRequests = new Map<number, Promise<MovieDetails>>()

export function getCachedMovieDetails(movieId: number): MovieDetails | null {
  return detailsCache.get(movieId) ?? null
}

export function fetchMovieDetails(movieId: number): Promise<MovieDetails> {
  const cached = detailsCache.get(movieId)
  if (cached) {
    return Promise.resolve(cached)
  }

  const pending = pendingRequests.get(movieId)
  if (pending) {
    return pending
  }

  const request = api
    .getMovieDetails(movieId)
    .then((details) => {
      detailsCache.set(movieId, details)
      return details
    })
    .finally(() => {
      pendingRequests.delete(movieId)
    })

  pendingRequests.set(movieId, request)
  return request
}
