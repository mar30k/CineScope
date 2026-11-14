export function ErrorMessage({ message }: { message: string }) {
  return (
    <div className="error-message" role="alert">
      <strong>Something went wrong.</strong>
      <span className="error-detail">{message}</span>
    </div>
  )
}
