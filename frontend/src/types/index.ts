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

export interface MovieDetails {
  movie_id: number
  title: string
  year: number | null
  overview: string | null
  genres: string[]
  runtime: number | null
  vote_average: number | null
  vote_count: number | null
  director: string[]
}

export interface RecommendationItem {
  movie_id: number
  title: string
  year: number | null
  rank: number
  score: number
  coverage: number
  agreement: number | null
}

export interface RecommendationResponse {
  movie_id: number
  count: number
  recommendations: RecommendationItem[]
}

export interface SelectedMovieInfo {
  movie_id: number
  title: string
  year: number | null
}

export interface DiscoveryEntry {
  movie: SelectedMovieInfo
  details: MovieDetails
  recommendations: RecommendationResponse | null
}
