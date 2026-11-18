import { useState } from 'react'

type PosterVariant = 'card' | 'hero' | 'recommendation' | 'suggestion'

interface MoviePosterProps {
  title: string
  src: string | null
  variant: PosterVariant
  loading?: 'eager' | 'lazy'
}

const POSTER_VARIANTS: Record<
  PosterVariant,
  { container: string; mark: string; label: string | null }
> = {
  card: {
    container: 'movie-card-poster',
    mark: 'movie-card-poster-mark',
    label: null,
  },
  hero: {
    container: 'movie-hero-poster',
    mark: 'movie-hero-poster-mark',
    label: 'movie-hero-poster-label',
  },
  recommendation: {
    container: 'recommendation-poster',
    mark: 'recommendation-poster-mark',
    label: 'recommendation-poster-label',
  },
  suggestion: {
    container: 'search-suggestion-poster',
    mark: 'search-suggestion-poster-mark',
    label: null,
  },
}

function posterInitial(title: string) {
  return title.trim().charAt(0).toUpperCase() || '?'
}

export function MoviePoster({
  title,
  src,
  variant,
  loading = 'lazy',
}: MoviePosterProps) {
  const [failedSource, setFailedSource] = useState<string | null>(null)
  const source = src?.trim() || null
  const showImage = source !== null && failedSource !== source
  const styles = POSTER_VARIANTS[variant]

  return (
    <span className={`${styles.container}${showImage ? ' has-image' : ''}`}>
      {showImage ? (
        <img
          className="poster-image"
          src={source}
          alt={`Poster for ${title}`}
          loading={loading}
          decoding="async"
          width={500}
          height={750}
          onError={() => setFailedSource(source)}
        />
      ) : (
        <>
          <span className={styles.mark} aria-hidden="true">
            {posterInitial(title)}
          </span>
          {styles.label && (
            <span className={styles.label} aria-hidden="true">
              CineScope
            </span>
          )}
        </>
      )}
    </span>
  )
}

interface MovieBackdropProps {
  title: string
  src: string | null
}

export function MovieBackdrop({ title, src }: MovieBackdropProps) {
  const [failedSource, setFailedSource] = useState<string | null>(null)
  const source = src?.trim() || null

  if (source === null || failedSource === source) {
    return null
  }

  return (
    <div className="movie-hero-backdrop-layer">
      <img
        className="movie-hero-backdrop"
        src={source}
        alt={`Backdrop for ${title}`}
        loading="eager"
        decoding="async"
        width={1280}
        height={720}
        onError={() => setFailedSource(source)}
      />
      <span className="movie-hero-scrim" aria-hidden="true" />
    </div>
  )
}
