import { BarChart3, BookOpen, ChevronRight, CircleHelp, FileText, Gauge, Globe2, Home, LineChart, MessageCircle, Newspaper, Scale, Settings, Star, TrendingUp } from "lucide-react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

const groups = [
  { label: "Start", links: [{ to: "/overview", label: "Overview", icon: Home }] },
  { label: "Analyze", links: [{ to: "/stock-outlook", label: "Stock Outlook", icon: TrendingUp }, { to: "/market-snapshot", label: "Market Snapshot", icon: LineChart }, { to: "/compare", label: "Compare Tickers", icon: Scale }, { to: "/news", label: "News & Sentiment", icon: Newspaper }] },
  { label: "Learn", links: [{ to: "/ask-learn", label: "Ask & Learn", icon: MessageCircle }, { to: "/learning-center", label: "Learning Center", icon: BookOpen }] },
];

export function Layout() {
  const location = useLocation();
  const isOverview = location.pathname === "/overview" || location.pathname === "/";
  return <div className="app-shell"><aside className="sidebar"><div className="brand"><img src="/stocksense-logo.png" alt="StockSense logo" /><span>StockSense</span></div><nav>{groups.map((group) => <div className="nav-group" key={group.label}><div className="nav-label">{group.label}</div>{group.links.map(({ to, label, icon: Icon }) => <NavLink key={to} to={to} className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}><Icon size={18} /><span>{label}</span></NavLink>)}</div>)}<div className="nav-group utility"><NavLink to="/settings" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}><Settings size={18} /><span>Settings</span></NavLink><NavLink to="/help" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}><CircleHelp size={18} /><span>Help</span></NavLink></div></nav></aside><main className={`main-content ${isOverview ? "overview-content" : ""}`}><Outlet /></main></div>;
}
