import type { PropsWithChildren } from "react";

export function PageFrame({ children }: PropsWithChildren) {
  return <div className="page-frame">{children}</div>;
}
