import type { Company } from "../types";

interface Props {
  label: string;
  companies: Company[];
  value: string;
  onChange: (ticker: string) => void;
  locked?: string[];
}

export function CompanySelector({ label, companies, value, onChange, locked = [] }: Props) {
  const lockedSet = new Set(locked);
  return (
    <label className="field-label">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {companies.map((company) => (
          <option key={company.ticker} value={company.ticker} disabled={lockedSet.has(company.ticker) && company.ticker !== value}>
            {lockedSet.has(company.ticker) && company.ticker !== value ? "🔒 " : ""}{company.company_name} ({company.ticker})
          </option>
        ))}
      </select>
    </label>
  );
}
