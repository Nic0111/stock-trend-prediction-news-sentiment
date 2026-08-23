import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { CompanyTooltip } from "./Chart";

describe("CompanyTooltip", () => {
  it("shows the selected company and ticker for a single-company chart", () => {
    render(<CompanyTooltip active label="2026-07-24" identity={{ companyName: "Maybank", ticker: "1155.KL" }} payload={[{ dataKey: "value", name: "Adjusted close", value: 10.25, color: "#0071e3" }]} />);
    expect(screen.getByText("Maybank")).toBeInTheDocument();
    expect(screen.getByText("1155.KL")).toBeInTheDocument();
    expect(screen.getByText("2026-07-24")).toBeInTheDocument();
  });

  it("shows the correct company identity beside every comparison series", () => {
    render(<CompanyTooltip active label="2026-07-24" seriesIdentity={{ "1155.KL": { companyName: "Maybank", ticker: "1155.KL" }, "1295.KL": { companyName: "Public Bank", ticker: "1295.KL" } }} payload={[{ dataKey: "1155.KL", name: "1155.KL", value: 103.2, color: "#0071e3" }, { dataKey: "1295.KL", name: "1295.KL", value: 98.4, color: "#248a3d" }]} />);
    expect(screen.getByText("Maybank")).toBeInTheDocument();
    expect(screen.getByText("Public Bank")).toBeInTheDocument();
    expect(screen.getByText("1155.KL")).toBeInTheDocument();
    expect(screen.getByText("1295.KL")).toBeInTheDocument();
  });
});
