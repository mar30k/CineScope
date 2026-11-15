interface PrevButtonProps {
  children: React.ReactNode
  onClick: () => void
  disabled?: boolean
}

export function PrevButton({ children, onClick, disabled = false }: PrevButtonProps) {
  return (
    <button
      type="button"
      className="prev-button"
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  )
}
