import type { SelectedMovieInfo } from '../types'
import { BackButton } from './BackButton'
import { PrevButton } from './PrevButton'

interface DiscoveryToolbarProps {
  currentMovie: SelectedMovieInfo
  history: SelectedMovieInfo[]
  onBackToResults: () => void
  onPreviousMovie: () => void
}

export function DiscoveryToolbar({
  currentMovie,
  history,
  onBackToResults,
  onPreviousMovie,
}: DiscoveryToolbarProps) {
  return (
    <div className="discovery-toolbar">
      <nav className="discovery-trail" aria-label="Discovery trail">
        <span className="trail-label">Discovery</span>
        <span className="trail-separator" aria-hidden="true">
          /
        </span>
        <span className="trail-root">Search results</span>

        {history.map((entry) => (
          <span className="trail-segment" key={entry.movie_id}>
            <span className="trail-separator" aria-hidden="true">
              /
            </span>
            <span className="trail-item" title={entry.title}>
              {entry.title}
            </span>
          </span>
        ))}

        <span className="trail-separator" aria-hidden="true">
          /
        </span>
        <span className="trail-current" aria-current="page">
          {currentMovie.title}
        </span>
      </nav>

      <div className="discovery-actions">
        <BackButton onClick={onBackToResults}>Back to results</BackButton>
        {history.length > 0 && (
          <PrevButton onClick={onPreviousMovie}>Previous movie</PrevButton>
        )}
      </div>
    </div>
  )
}
