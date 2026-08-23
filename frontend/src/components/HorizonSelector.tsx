import type { Horizon } from "../types";

const HORIZON_OPTIONS: Array<{ value: Horizon; label: string; hint: string }> = [
  { value: 1, label: "1 trading day", hint: "Near term" },
  { value: 5, label: "5 trading days", hint: "About one market week" },
  { value: 10, label: "10 trading days", hint: "About two market weeks" },
];

interface Props {
  value: Horizon;
  onChange: (value: Horizon) => void;
  legend?: string;
}

/** A consistent, beginner-friendly horizon choice used across analysis pages. */
export function HorizonSelector({ value, onChange, legend = "Prediction horizon" }: Props) {
  return (
    <fieldset className="horizon-selector">
      <legend>{legend}</legend>
      <div className="horizon-options" role="radiogroup" aria-label={legend}>
        {HORIZON_OPTIONS.map((option) => (
          <label className={`horizon-option${value === option.value ? " selected" : ""}`} key={option.value}>
            <input
              type="radio"
              name={`horizon-${legend.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`}
              value={option.value}
              checked={value === option.value}
              onChange={() => onChange(option.value)}
            />
            <span className="horizon-option-copy">
              <strong>{option.label}</strong>
              <small>{option.hint}</small>
            </span>
          </label>
        ))}
      </div>
      <p className="horizon-help">Trading days exclude weekends and Bursa Malaysia market holidays.</p>
    </fieldset>
  );
}
