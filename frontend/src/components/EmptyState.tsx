export function EmptyState({ query }: { query: string }) {
  return (
    <div className="empty-state">
      <span className="empty-state-icon" aria-hidden="true">
        ⌕
      </span>
      <p>
        No movies matched <strong>“{query}”</strong>.
      </p>
      <p className="empty-state-hint">
        Try a different title or a shorter fragment.
      </p>
    </div>
  )
}
