import { useState } from "react";
import type { Company } from "../types";

export function CompanyLogo({ company, className = "" }: { company?: Company; className?: string }) {
  const [failedPrimary, setFailedPrimary] = useState(false);
  const [failedFallback, setFailedFallback] = useState(false);
  if (!company) return <div className={`company-logo ${className}`}><span>?</span></div>;
  const source = !failedPrimary ? company.primary_logo_url : !failedFallback ? company.fallback_logo_url : "";
  return (
    <div className={`company-logo ${className}`} aria-label={`${company.company_name} logo`}>
      {source ? (
        <img src={source} alt={`${company.company_name} logo`} onError={() => !failedPrimary ? setFailedPrimary(true) : setFailedFallback(true)} />
      ) : <span>{company.initials}</span>}
    </div>
  );
}
