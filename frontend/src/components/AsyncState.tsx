export function LoadingState({ label = "Loading StockSense data…" }: { label?: string }) {
  return <div className="async-state loading-state"><span className="spinner" aria-hidden="true" />{label}</div>;
}

export function ErrorState({ message }: { message: string }) {
  return <div className="async-state error-state" role="alert">{message}</div>;
}
