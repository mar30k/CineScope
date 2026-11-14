export interface MovieSearchResult {
  movie_id: number
  title: string
  year: number | null
}

export interface MovieSearchResponse {
  query: string
  count: number
  results: MovieSearchResult[]
}

export interface ApiError {
  detail: string
}
