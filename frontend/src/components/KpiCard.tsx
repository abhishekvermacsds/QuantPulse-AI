import React from "react";
import { LucideIcon } from "lucide-react";

interface KpiCardProps {
  title: string;
  value: string;
  subtext?: string;
  icon: LucideIcon;
  badge?: string;
  badgeColor?: "green" | "red" | "cyan" | "slate";
}

export const KpiCard: React.FC<KpiCardProps> = ({
  title,
  value,
  subtext,
  icon: Icon,
  badge,
  badgeColor = "cyan",
}) => {
  const badgeClasses = {
    green: "bg-emerald-950/70 text-emerald-400 border-emerald-800/40",
    red: "bg-rose-950/70 text-rose-400 border-rose-800/40",
    cyan: "bg-cyan-950/70 text-cyan-400 border-cyan-800/40",
    slate: "bg-slate-800/60 text-slate-300 border-slate-700/50",
  }[badgeColor];

  return (
    <div className="fintech-card p-5 flex flex-col justify-between">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          {title}
        </span>
        <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-cyan-400">
          <Icon className="h-4 w-4" />
        </div>
      </div>

      <div className="flex items-baseline justify-between gap-2">
        <span className="text-2xl font-bold mono tracking-tight text-white">
          {value}
        </span>
        {badge && (
          <span
            className={`text-[11px] font-semibold px-2 py-0.5 rounded border ${badgeClasses}`}
          >
            {badge}
          </span>
        )}
      </div>

      {subtext && (
        <span className="text-xs text-slate-500 mt-2 block">
          {subtext}
        </span>
      )}
    </div>
  );
};
