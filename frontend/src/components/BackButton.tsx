interface BackButtonProps {
  children: React.ReactNode
  onClick: () => void
}

export function BackButton({ children, onClick }: BackButtonProps) {
  return (
    <button type="button" className="back-button" onClick={onClick}>
      {children}
    </button>
  )
}
