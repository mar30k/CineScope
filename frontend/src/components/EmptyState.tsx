export function EmptyState({ query }: { query: string }) {
  return (
    <div className="empty-state">
      <p>No movies matched “<strong>{query}</strong>”.</p>
      <p>Try a different title or a shorter fragment.</p>
    </div>
  )
}
