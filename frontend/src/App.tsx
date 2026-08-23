import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { AppStateProvider } from "./state/AppState";
import { LoadingState } from "./components/AsyncState";
import { ErrorState } from "./components/AsyncState";
import { useQuery } from "@tanstack/react-query";
import { getSystemStatus } from "./api/client";
import { usePerformanceMonitor } from "./hooks/usePerformanceMonitor";
import "./styles/app.css";

const OverviewPage = lazy(() => import("./pages/OverviewPage").then(({ OverviewPage }) => ({ default: OverviewPage })));
const StockOutlookPage = lazy(() => import("./pages/StockOutlookPage").then(({ StockOutlookPage }) => ({ default: StockOutlookPage })));
const MarketSnapshotPage = lazy(() => import("./pages/MarketSnapshotPage").then(({ MarketSnapshotPage }) => ({ default: MarketSnapshotPage })));
const ComparePage = lazy(() => import("./pages/ComparePage").then(({ ComparePage }) => ({ default: ComparePage })));
const NewsPage = lazy(() => import("./pages/NewsPage").then(({ NewsPage }) => ({ default: NewsPage })));
const AskLearnPage = lazy(() => import("./pages/AskLearnPage").then(({ AskLearnPage }) => ({ default: AskLearnPage })));
const LearningCenterPage = lazy(() => import("./pages/LearningCenterPage").then(({ LearningCenterPage }) => ({ default: LearningCenterPage })));
const ModelPerformancePage = lazy(() => import("./pages/ModelPerformancePage").then(({ ModelPerformancePage }) => ({ default: ModelPerformancePage })));
const SettingsPage = lazy(() => import("./pages/SystemPages").then(({ SettingsPage }) => ({ default: SettingsPage })));
const HelpPage = lazy(() => import("./pages/SystemPages").then(({ HelpPage }) => ({ default: HelpPage })));
const PlaceholderPage = lazy(() => import("./pages/SystemPages").then(({ PlaceholderPage }) => ({ default: PlaceholderPage })));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      gcTime: 30 * 60_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});
const EXPECTED_FEATURE_SCHEMA = "aligned-market-sessions-v2";

function RuntimeGuard({ children }: { children: React.ReactNode }) {
  const status = useQuery({
    queryKey: ["system-status"],
    queryFn: ({ signal }) => getSystemStatus(signal),
    staleTime: 30000,
    retry: false,
    refetchInterval: () => document.hidden ? false : 60000,
    refetchOnWindowFocus: true,
  });
  if (status.isLoading) return <LoadingState label="Checking StockSense runtime…" />;
  if (status.isError) return <ErrorState message={`StockSense could not connect to its backend: ${status.error.message}`} />;
  if (!status.data || status.data.feature_schema_version !== EXPECTED_FEATURE_SCHEMA || !status.data.model_registry_signature) {
    return <main className="runtime-blocker"><h1>Restart StockSense</h1><p>The backend is running an older model or feature schema. Stop and start StockSense, then reload this page.</p></main>;
  }
  return <>{children}</>;
}

export default function App() {
  usePerformanceMonitor();
  return <QueryClientProvider client={queryClient}><AppStateProvider><BrowserRouter><RuntimeGuard><Suspense fallback={<LoadingState label="Loading StockSense…" />}><Routes><Route element={<Layout />}><Route path="/" element={<Navigate to="/overview" replace />} /><Route path="/overview" element={<OverviewPage />} /><Route path="/stock-outlook" element={<StockOutlookPage />} /><Route path="/market-snapshot" element={<MarketSnapshotPage />} /><Route path="/compare" element={<ComparePage />} /><Route path="/news" element={<NewsPage />} /><Route path="/ask-learn" element={<AskLearnPage />} /><Route path="/learning-center" element={<LearningCenterPage />} /><Route path="/model-performance" element={<ModelPerformancePage />} /><Route path="/settings" element={<SettingsPage />} /><Route path="/help" element={<HelpPage />} /><Route path="/alerts" element={<PlaceholderPage title="Alerts" description="A calm place for future reminders about market observations and model evidence." />} /><Route path="/reports" element={<PlaceholderPage title="Reports" description="Review the evidence behind StockSense outputs before drawing conclusions." />} /><Route path="*" element={<Navigate to="/overview" replace />} /></Route></Routes></Suspense></RuntimeGuard></BrowserRouter></AppStateProvider></QueryClientProvider>;
}
