import { fireEvent, render, screen } from "@testing-library/react";
import { CompanySelector } from "./CompanySelector";
import type { Company } from "../types";

const companies: Company[] = [
  { ticker: "1155.KL", company_name: "Malayan Banking Berhad", domain: "maybank.com", primary_logo_url: "", fallback_logo_url: "", initials: "MB" },
  { ticker: "1295.KL", company_name: "Public Bank Berhad", domain: "publicbank.com.my", primary_logo_url: "", fallback_logo_url: "", initials: "PB" },
];

test("keeps all comparison options visible while disabling selected duplicates", () => {
  const onChange = vi.fn();
  render(<CompanySelector label="Ticker 2" companies={companies} value="1295.KL" locked={["1155.KL"]} onChange={onChange} />);
  expect(screen.getByRole("option", { name: /Malayan Banking/ })).toBeDisabled();
  expect(screen.getByRole("option", { name: /Public Bank/ })).not.toBeDisabled();
  fireEvent.change(screen.getByRole("combobox"), { target: { value: "1295.KL" } });
  expect(onChange).toHaveBeenCalledWith("1295.KL");
});
