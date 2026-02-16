/**
 * Reusable gauge visualizations for hardware monitoring.
 */

// ── Circle Gauge ─────────────────────────────────────────────
export function CircleGauge({ value, max, label, unit, color, icon: Icon, sublabel }) {
    const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
    const radius = 36;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (pct / 100) * circumference;

    return (
        <div className="flex flex-col items-center">
            <div className="relative w-24 h-24">
                <svg className="w-24 h-24 -rotate-90" viewBox="0 0 80 80">
                    <circle cx="40" cy="40" r={radius} fill="none" stroke="#f1f3f5" strokeWidth="6" />
                    <circle
                        cx="40" cy="40" r={radius} fill="none"
                        stroke={color} strokeWidth="6" strokeLinecap="round"
                        strokeDasharray={circumference} strokeDashoffset={offset}
                        className="transition-all duration-500 ease-out"
                    />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-lg font-bold text-surface-900">{value != null ? Math.round(value) : '—'}</span>
                    <span className="text-xs text-surface-400">{unit}</span>
                </div>
            </div>
            <div className="flex items-center gap-1 mt-1.5">
                {Icon && <Icon className="w-3 h-3" style={{ color }} />}
                <span className="text-xs font-medium text-surface-600">{label}</span>
            </div>
            {sublabel && <span className="text-xs text-surface-400">{sublabel}</span>}
        </div>
    );
}

// ── Bar Gauge ────────────────────────────────────────────────
export function BarGauge({ value, max, label, color, format }) {
    const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
    return (
        <div className="space-y-1">
            <div className="flex justify-between text-xs">
                <span className="text-surface-600 font-medium">{label}</span>
                <span className="text-surface-500">{format || `${value?.toFixed?.(1) ?? '—'} / ${max?.toFixed?.(1) ?? '—'} GB`}</span>
            </div>
            <div className="h-2.5 bg-surface-100 rounded-full overflow-hidden">
                <div
                    className="h-full rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${pct}%`, backgroundColor: color }}
                />
            </div>
        </div>
    );
}

// ── Shared utilities ─────────────────────────────────────────
export function formatTime(seconds) {
    if (seconds == null || seconds < 0) return '--:--';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}
