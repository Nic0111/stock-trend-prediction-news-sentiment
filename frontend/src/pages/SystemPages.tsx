import { useQuery } from "@tanstack/react-query";
import { getSystemStatus } from "../api/client";
import { LoadingState, ErrorState } from "../components/AsyncState";
import { PageHeader } from "../components/PageHeader";

export function SettingsPage() {
  const status = useQuery({ queryKey: ["system-status"], queryFn: ({ signal }) => getSystemStatus(signal), staleTime: 60000 });
  return <><PageHeader eyebrow="System" title="Settings" description="Review the data-source boundaries and runtime configuration." /><section className="settings-grid"><div className="settings-panel"><h2>Data sources</h2><p>Market data is sourced from Yahoo Finance. News coverage uses NewsAPI with the quota-aware Marketaux fallback.</p><dl><dt>NewsAPI</dt><dd>{status.data?.newsapi_configured ? "Configured" : "Not configured"}</dd><dt>Marketaux</dt><dd>{status.data?.marketaux_configured ? "Configured" : "Not configured"}</dd><dt>FinBERT</dt><dd>{status.data?.finbert_runtime_available ? "Available" : "Unavailable"}</dd></dl></div><div className="settings-panel"><h2>Model registry</h2>{status.isLoading && <LoadingState label="Checking runtime…" />}{status.isError && <ErrorState message={status.error.message} />}{status.data && <dl>{Object.entries(status.data.stock_models_available).map(([horizon, available]) => <><dt key={`${horizon}-label`}>{horizon}-day horizon</dt><dd key={`${horizon}-value`}>{available ? "Artifact available" : "Artifact unavailable"}</dd></>)}</dl>}</div><div className="settings-panel"><h2>Ask & Learn</h2><p>{status.data?.openrouter_configured ? `Primary model: ${status.data.openrouter_model}` : "OpenRouter is not configured. Add the ignored config/openrouter_api_key.txt file or set OPENROUTER_API_KEY."}</p>{status.data?.openrouter_configured && <p>Fallback model: {status.data.openrouter_fallback_model}</p>}</div></section></>;
}

export function HelpPage() {
  return <><PageHeader eyebrow="Support" title="Help" description="Learn how to move through StockSense and interpret its evidence carefully." /><div className="learning-list"><details open><summary>Where should I start?</summary><p>Open Stock Outlook, select a company and prediction horizon, then choose Analyze to load current market and news evidence.</p></details><details><summary>How should I read a trend result?</summary><p>Treat the trend as a model signal rather than a guarantee. Check the observation date, evidence status and probability together.</p></details><details><summary>Why is some information unavailable?</summary><p>Market and news providers can be delayed, rate-limited or missing coverage. StockSense labels unavailable or research-only evidence instead of filling gaps with guesses.</p></details><details><summary>Where can I learn investing terms?</summary><p>Use Ask & Learn for conversational explanations, or open Learning Center for guided notes about probabilities, sentiment and model limitations.</p></details></div></>;
}

export function PlaceholderPage({ title, description }: { title: string; description: string }) {
  return <><PageHeader eyebrow="Workspace" title={title} description={description} /><section className="settings-panel"><h2>{title} are coming next</h2><p>This placeholder is preserved from the current Streamlit application. The migration does not remove the route or change the existing scope.</p></section></>;
}
