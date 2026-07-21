import { useEffect } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { createAnalysis, getCompanies } from "../api/client";
import { LoadingState, ErrorState } from "../components/AsyncState";
import { CompanyLogo } from "../components/CompanyLogo";
import { CompanySelector } from "../components/CompanySelector";
import { isArticleContributingToContext, NewsCard } from "../components/NewsCard";
import { NewsSummaryCard, TrendCard } from "../components/TrendCard";
import { PageHeader } from "../components/PageHeader";
import { useAppState } from "../state/AppState";
import type { Horizon } from "../types";

export function StockOutlookPage() {
  const companies = useQuery({ queryKey: ["companies"], queryFn: ({ signal }) => getCompanies(signal), staleTime: 86400000 });
  const { outlookTicker, setOutlookTicker, outlookHorizon, setOutlookHorizon, analysis, setAnalysis } = useAppState();
  useEffect(() => { if (companies.data?.length && !companies.data.some((company) => company.ticker === outlookTicker)) setOutlookTicker(companies.data[0].ticker); }, [companies.data, outlookTicker, setOutlookTicker]);
  const analysisMutation = useMutation({ mutationFn: () => createAnalysis({ ticker: outlookTicker, horizon: outlookHorizon }), onSuccess: setAnalysis });
  if (companies.isLoading) return <><PageHeader eyebrow="Analyze" title="Stock Outlook" description="See the modelled direction and recent news evidence for one Bursa Malaysia company." /><LoadingState /></>;
  if (companies.isError || !companies.data) return <><PageHeader eyebrow="Analyze" title="Stock Outlook" description="See the modelled direction and recent news evidence for one Bursa Malaysia company." /><ErrorState message={companies.error?.message || "Companies could not be loaded."} /></>;
  const company = companies.data.find((item) => item.ticker === outlookTicker);
  const matchingAnalysis = analysis?.ticker === outlookTicker && analysis.horizon_days === outlookHorizon ? analysis : null;
  const newsContext = matchingAnalysis?.news?.contexts[String(outlookHorizon)] || {};
  return <><PageHeader eyebrow="Analyze" title="Stock Outlook" description="See the modelled direction and recent news evidence for one Bursa Malaysia company." /><section className="selector-panel"><CompanyLogo company={company} className="selector-logo" /><CompanySelector label="Company" companies={companies.data} value={outlookTicker} onChange={setOutlookTicker} /><fieldset><legend>Prediction horizon</legend><div className="radio-row">{([1, 5, 10] as Horizon[]).map((value) => <label key={value}><input type="radio" checked={outlookHorizon === value} onChange={() => setOutlookHorizon(value)} />{value} trading day(s)</label>)}</div></fieldset><button className="primary-button" onClick={() => analysisMutation.mutate()} disabled={analysisMutation.isPending}>{analysisMutation.isPending ? "Analyzing…" : "Analyze"}</button></section>{analysisMutation.isError && <ErrorState message={analysisMutation.error.message} />}{matchingAnalysis && <><div className="signal-grid"><TrendCard analysis={matchingAnalysis} /><NewsSummaryCard analysis={matchingAnalysis} /></div><section className="article-section"><h2 className="section-heading">Relevant News Articles</h2><div className="article-grid">{(matchingAnalysis.news?.articles || []).map((article, index) => <NewsCard article={article} contributingHorizon={isArticleContributingToContext(article, newsContext, matchingAnalysis.news?.as_of_date || "") ? outlookHorizon : undefined} key={`${article.url || article.title}-${index}`} />)}</div></section></>}</>;
}
