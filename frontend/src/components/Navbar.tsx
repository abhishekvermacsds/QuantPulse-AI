import React from "react";
import { Activity, Cpu, Wifi, WifiOff } from "lucide-react";
import { HealthStatus } from "../types";

interface NavbarProps {
  health: HealthStatus | null;
  loading: boolean;
  onRefresh: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ health, loading, onRefresh }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50 px-6 py-4">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-950/40">
            <Activity className="h-6 w-6 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-bold tracking-tight text-white">QuantPulse AI</h1>
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800/50">
                v0.1.0
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Autonomous Portfolio Risk & Volatility Optimization Engine
            </p>
          </div>
        </div>

        {/* System Diagnostics */}
        <div className="flex items-center space-x-4">
          {health ? (
            <div className="flex items-center space-x-3 bg-slate-900 border border-slate-800 px-3.5 py-1.5 rounded-lg text-xs">
              <div className="flex items-center space-x-1.5 text-emerald-400">
                <Wifi className="h-3.5 w-3.5 pulse-indicator" />
                <span className="font-medium">API Online</span>
              </div>
              <span className="text-slate-600">|</span>
              <div className="flex items-center space-x-1.5 text-slate-300">
                <Cpu className="h-3.5 w-3.5 text-cyan-400" />
                <span className="mono uppercase">{health.device}</span>
              </div>
              <span className="text-slate-600">|</span>
              <div className="text-slate-400 mono">
                {health.memory_mb.toFixed(0)} MB
              </div>
            </div>
          ) : (
            <div className="flex items-center space-x-1.5 bg-rose-950/50 border border-rose-800/40 text-rose-300 px-3 py-1.5 rounded-lg text-xs">
              <WifiOff className="h-3.5 w-3.5" />
              <span>Backend Offline</span>
            </div>
          )}

          <button
            onClick={onRefresh}
            disabled={loading}
            className="px-3.5 py-1.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold tracking-wide transition shadow-sm cursor-pointer"
          >
            {loading ? "Refreshing..." : "Refresh Data"}
          </button>
        </div>
      </div>
    </header>
  );
};
