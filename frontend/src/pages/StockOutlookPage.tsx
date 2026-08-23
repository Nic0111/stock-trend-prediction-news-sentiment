import { useEffect } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { createAnalysis, getCompanies, resetNewsOverrides, updateNewsOverride } from "../api/client";
import { AggregateExplanationPanel } from "../components/AggregateExplanationPanel";
import { LoadingState, ErrorState } from "../components/AsyncState";
import { CompanyLogo } from "../components/CompanyLogo";
import { CompanySelector } from "../components/CompanySelector";
import { HorizonSelector } from "../components/HorizonSelector";
import { isArticleContributingToContext, NewsCard } from "../components/NewsCard";
import { CombinedOutlookCard, NewsSummaryCard, TrendCard } from "../components/TrendCard";
import { PageHeader } from "../components/PageHeader";
import { StockModelEvidence } from "../components/ModelDiagnostics";
import { useAppState } from "../state/AppState";

export function StockOutlookPage() {
  const companies = useQuery({ queryKey: ["companies"], queryFn: ({ signal }) => getCompanies(signal), staleTime: 86400000 });
  const { outlookTicker, setOutlookTicker, outlookHorizon, setOutlookHorizon, analysis, setAnalysis, clearAggregateTutorThread } = useAppState();

  useEffect(() => {
    if (companies.data?.length && !companies.data.some((company) => company.ticker === outlookTicker)) {
      setOutlookTicker(companies.data[0].ticker);
    }
  }, [companies.data, outlookTicker, setOutlookTicker]);

  const analysisMutation = useMutation({
    mutationFn: () => createAnalysis({ ticker: outlookTicker, horizon: outlookHorizon }),
    onSuccess: (nextAnalysis) => { setAnalysis(nextAnalysis); clearAggregateTutorThread(); },
  });
  const matchingAnalysis = analysis?.ticker === outlookTicker && analysis.horizon_days === outlookHorizon ? analysis : null;
  const correctionMutation = useMutation({
    mutationFn: ({ articleId, action }: { articleId: string; action: "include" | "exclude" }) =>
      updateNewsOverride(matchingAnalysis!.analysis_id, articleId, action),
    onSuccess: (nextAnalysis) => { setAnalysis(nextAnalysis); clearAggregateTutorThread(); },
  });
  const resetMutation = useMutation({
    mutationFn: () => resetNewsOverrides(matchingAnalysis!.analysis_id),
    onSuccess: (nextAnalysis) => { setAnalysis(nextAnalysis); clearAggregateTutorThread(); },
  });

  if (companies.isLoading) {
    return <><PageHeader eyebrow="Analyze" title="Stock Outlook" description="See the modelled direction and recent news evidence for one Bursa Malaysia company." /><LoadingState /></>;
  }
  if (companies.isError || !companies.data) {
    return <><PageHeader eyebrow="Analyze" title="Stock Outlook" description="See the modelled direction and recent news evidence for one Bursa Malaysia company." /><ErrorState message={companies.error?.message || "Companies could not be loaded."} /></>;
  }

  const company = companies.data.find((item) => item.ticker === outlookTicker);
  const newsContext = matchingAnalysis?.news?.contexts[String(outlookHorizon)] || {};
  const articles = matchingAnalysis?.news?.articles || [];
  const reviewArticles = matchingAnalysis?.news?.review_articles || [];

  return <>
    <PageHeader eyebrow="Analyze" title="Stock Outlook" description="See the modelled direction and recent news evidence for one Bursa Malaysia company." />
    <section className="selector-panel stacked-selector">
      <div className="selector-primary-row">
        <CompanyLogo company={company} className="selector-logo" />
        <CompanySelector label="Company" companies={companies.data} value={outlookTicker} onChange={setOutlookTicker} />
        <button className="primary-button" onClick={() => analysisMutation.mutate()} disabled={analysisMutation.isPending}>
          {analysisMutation.isPending ? "Analyzing..." : "Analyze"}
        </button>
      </div>
      <HorizonSelector value={outlookHorizon} onChange={setOutlookHorizon} />
    </section>
    {analysisMutation.isError && <ErrorState message={analysisMutation.error.message} />}
    {matchingAnalysis && <section className="synthesis-branch">
      <CombinedOutlookCard analysis={matchingAnalysis}>
        <AggregateExplanationPanel analysis={matchingAnalysis} />
      </CombinedOutlookCard>
      <div className="synthesis-connectors" aria-hidden="true">
        <span className="connector-stem" />
        <span className="connector-junction" />
        <span className="connector-branch connector-left" />
        <span className="connector-branch connector-right" />
      </div>
      <div className="signal-grid synthesis-children">
        <div className="synthesis-child"><div className="synthesis-child-label">Stock trend</div><TrendCard analysis={matchingAnalysis} /></div>
        <div className="synthesis-child"><div className="synthesis-child-label">News sentiment</div><NewsSummaryCard analysis={matchingAnalysis} /></div>
      </div>
      <StockModelEvidence horizon={outlookHorizon} compact />
      <section className="article-section">
        <div className="section-heading-row"><h2 className="section-heading">Relevant News Articles</h2>{Object.keys(matchingAnalysis.news?.user_overrides || {}).length > 0 && <button className="secondary-button" type="button" onClick={() => resetMutation.mutate()} disabled={resetMutation.isPending}>Reset</button>}</div>
        {articles.length ? <div className="article-grid">{articles.map((article, index) => <NewsCard article={article} contributingHorizon={article.contributes_to_horizons?.includes(outlookHorizon) ? outlookHorizon : (isArticleContributingToContext(article, newsContext, matchingAnalysis.news?.as_of_date || "") ? outlookHorizon : undefined)} onExclude={article.article_id ? () => correctionMutation.mutate({ articleId: article.article_id!, action: "exclude" }) : undefined} busy={correctionMutation.isPending} key={`${article.article_id || article.url || article.title}-${index}`} />)}</div> : <div className="notice">No articles are currently included in the sentiment result.</div>}
        <details className="review-articles"><summary>Review excluded articles ({reviewArticles.length})</summary><p className="review-articles-help">These articles matched the search but were not specific enough to influence sentiment automatically. Include only if they clearly refer to the selected company.</p>{reviewArticles.length ? <div className="article-grid">{reviewArticles.map((article, index) => <NewsCard review article={article} onInclude={article.article_id ? () => correctionMutation.mutate({ articleId: article.article_id!, action: "include" }) : undefined} busy={correctionMutation.isPending} key={`${article.article_id || article.url || article.title}-review-${index}`} />)}</div> : <div className="notice">No excluded candidates for this analysis.</div>}</details>
      </section>
    </section>}
  </>;
}
