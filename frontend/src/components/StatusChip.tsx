export function StatusChip({ message }: { message: string }) {
  return (
    <div className="status-chip" role="status" aria-live="polite">
      {message}
    </div>
  )
}
