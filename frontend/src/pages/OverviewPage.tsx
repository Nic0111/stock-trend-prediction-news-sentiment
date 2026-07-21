import { useQuery } from "@tanstack/react-query";
import { getCompanies } from "../api/client";
import { ScrollWorld } from "../components/overview/ScrollWorld";

export function OverviewPage() {
  const companies = useQuery({
    queryKey: ["companies"],
    queryFn: ({ signal }) => getCompanies(signal),
    staleTime: 86400000,
  });

  // The landing experience is useful even when the API is warming up. Company
  // metadata is intentionally non-blocking so the first viewport paints fast.
  return <ScrollWorld companyCount={companies.data?.length} companyLoadFailed={companies.isError} />;
}
