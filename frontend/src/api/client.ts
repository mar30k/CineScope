const baseUrl = import.meta.env.VITE_API_BASE_URL

function url(path: string, params?: Record<string, string | number>) {
  const u = new URL(`${baseUrl}${path}`)
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      u.searchParams.set(key, String(value))
    }
  }
  return u.toString()
}

async function get<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const response = await fetch(url(path, params), {
    method: 'GET',
    headers: { Accept: 'application/json' }
  })

  if (!response.ok) {
    let detail = 'An unexpected error occurred.'
    try {
      const body = (await response.json()) as { detail?: string }
      if (typeof body.detail === 'string') {
        detail = body.detail
      }
    } catch {
      // If the backend returns HTML (for example the FastAPI debug HTML on 500),
      // fall back to a generic message and do not expose internals.
    }

    throw new Error(detail)
  }

  // The backend always returns JSON on success for these endpoints.
  return response.json() as Promise<T>
}

export const api = {
  searchMovies: (query: string, limit = 10) =>
    get<import('../types').MovieSearchResponse>('/movies/search', { q: query, limit })
}
