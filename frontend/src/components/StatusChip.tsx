export function StatusChip({ message }: { message: string }) {
  return (
    <div className="status-chip" role="status" aria-live="polite">
      <span className="status-spinner" aria-hidden="true" />
      <span>{message}</span>
    </div>
  )
}
