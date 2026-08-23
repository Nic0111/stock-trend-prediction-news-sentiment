import { BarChart3, BookOpen, CircleHelp, Home, LineChart, MessageCircle, Newspaper, Scale, Settings, TrendingUp } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { PageFrame } from "./PageFrame";

const groups = [
  { label: "Start", links: [{ to: "/overview", label: "Overview", icon: Home }] },
  { label: "Analyze", links: [{ to: "/market-snapshot", label: "Market Snapshot", icon: LineChart }, { to: "/stock-outlook", label: "Stock Outlook", icon: TrendingUp }, { to: "/news", label: "News & Sentiment", icon: Newspaper }, { to: "/compare", label: "Compare Tickers", icon: Scale }] },
  { label: "Learn", links: [{ to: "/ask-learn", label: "Ask & Learn", icon: MessageCircle }, { to: "/learning-center", label: "Learning Center", icon: BookOpen }, { to: "/model-performance", label: "Model Performance", icon: BarChart3 }] },
];

export function Layout() {
  const location = useLocation();
  const isOverview = location.pathname === "/overview" || location.pathname === "/";
  const [sidebarExpanded, setSidebarExpanded] = useState(false);
  const collapseTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (collapseTimerRef.current !== null) window.clearTimeout(collapseTimerRef.current);
    collapseTimerRef.current = null;
    setSidebarExpanded(false);
  }, [location.pathname]);

  useEffect(() => () => {
    if (collapseTimerRef.current !== null) window.clearTimeout(collapseTimerRef.current);
  }, []);

  const expandSidebar = () => {
    if (!isOverview) return;
    if (collapseTimerRef.current !== null) window.clearTimeout(collapseTimerRef.current);
    collapseTimerRef.current = null;
    setSidebarExpanded(true);
  };

  const scheduleSidebarCollapse = () => {
    if (!isOverview) return;
    if (collapseTimerRef.current !== null) window.clearTimeout(collapseTimerRef.current);
    collapseTimerRef.current = window.setTimeout(() => {
      collapseTimerRef.current = null;
      setSidebarExpanded(false);
    }, 140);
  };

  const collapseIfFocusLeaves = (event: React.FocusEvent<HTMLElement>) => {
    if (!event.currentTarget.contains(event.relatedTarget)) scheduleSidebarCollapse();
  };

  return (
    <div className={`app-shell ${isOverview ? "cinematic-shell" : ""}`}>
      <aside
        className={`sidebar ${isOverview ? "cinematic-sidebar" : ""} ${sidebarExpanded ? "is-expanded" : ""}`}
        onMouseEnter={expandSidebar}
        onMouseLeave={scheduleSidebarCollapse}
        onFocusCapture={expandSidebar}
        onBlurCapture={collapseIfFocusLeaves}
      >
        <div className="brand">
          <img src="/stocksense-logo.png" alt="StockSense logo" />
          <span>StockSense</span>
        </div>
        <nav>
          {groups.map((group) => (
            <div className="nav-group" key={group.label}>
              <div className="nav-label">{group.label}</div>
              {group.links.map(({ to, label, icon: Icon }) => (
                <NavLink key={to} to={to} title={label} aria-label={label} className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
                  <Icon size={18} />
                  <span>{label}</span>
                </NavLink>
              ))}
            </div>
          ))}
          <div className="nav-group utility">
            <NavLink to="/settings" title="Settings" aria-label="Settings" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
              <Settings size={18} /><span>Settings</span>
            </NavLink>
            <NavLink to="/help" title="Help" aria-label="Help" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
              <CircleHelp size={18} /><span>Help</span>
            </NavLink>
          </div>
        </nav>
      </aside>
      <main className={`main-content ${isOverview ? "overview-content" : ""}`}>
        {isOverview ? <Outlet /> : <PageFrame><Outlet /></PageFrame>}
      </main>
    </div>
  );
}
