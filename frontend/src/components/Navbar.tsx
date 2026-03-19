import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const location = useLocation();
  const isHome = location.pathname === '/';

  return (
    <nav className="flex justify-between items-center px-8 py-4 border-b border-border">
      <Link to="/" className="flex items-center gap-2 text-xl font-semibold tracking-tight">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563EB" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M6.5 6.5l11 11" />
          <path d="M21 21l-1-1" />
          <path d="M3 3l1 1" />
          <path d="M18 22l4-4" />
          <path d="M2 6l4-4" />
          <path d="M3 10l7-7" />
          <path d="M14 21l7-7" />
        </svg>
        Mavis
      </Link>

      <div className="flex gap-6 items-center text-sm font-medium">
        <Link
          to="/"
          className={`transition-colors ${isHome ? 'text-text-main' : 'text-text-muted hover:text-text-main'}`}
        >
          Dashboard
        </Link>
        <Link
          to="/workout?type=bicep"
          className={`transition-colors ${!isHome ? 'text-text-main' : 'text-text-muted hover:text-text-main'}`}
        >
          {isHome ? 'Quick Start' : 'Live Workout'}
        </Link>
      </div>
    </nav>
  );
}
