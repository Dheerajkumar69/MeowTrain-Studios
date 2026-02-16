import { useState } from 'react';
import { GitCompare, ArrowRight, Loader, X, AlertCircle } from 'lucide-react';
import { trainingAPI } from '../../services/api';

/**
 * RunComparisonPanel — Side-by-side comparison of two training runs.
 * Shows config diff and metric comparison.
 */
export default function RunComparisonPanel({ projectId, runs = [], onClose }) {
    const [selectedRuns, setSelectedRuns] = useState([null, null]);
    const [comparison, setComparison] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const completedRuns = runs.filter(r => r.status === 'completed' || r.status === 'stopped' || r.status === 'failed');

    const handleCompare = async () => {
        if (!selectedRuns[0] || !selectedRuns[1]) return;
        setLoading(true);
        setError(null);
        try {
            const res = await trainingAPI.compareRuns(projectId, selectedRuns);
            setComparison(res.data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to compare runs');
        } finally {
            setLoading(false);
        }
    };

    const canCompare = selectedRuns[0] && selectedRuns[1] && selectedRuns[0] !== selectedRuns[1];

    return (
        <div className="bg-white rounded-2xl border border-surface-100 p-5 space-y-4">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-lg bg-blue-50 flex items-center justify-center">
                        <GitCompare className="w-4 h-4 text-blue-500" />
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold text-surface-900">Compare Training Runs</h3>
                        <p className="text-xs text-surface-500">Side-by-side metrics and config diff</p>
                    </div>
                </div>
                {onClose && (
                    <button
                        onClick={onClose}
                        className="text-surface-400 hover:text-surface-600 transition-colors p-1 rounded-lg hover:bg-surface-50"
                        aria-label="Close comparison panel"
                    >
                        <X className="w-4 h-4" />
                    </button>
                )}
            </div>

            {/* Run selectors */}
            <div className="flex items-end gap-3">
                <RunSelector
                    label="Run A"
                    runs={completedRuns}
                    selected={selectedRuns[0]}
                    onChange={id => setSelectedRuns([id, selectedRuns[1]])}
                    accent="blue"
                />
                <ArrowRight className="w-4 h-4 text-surface-300 mb-2.5 shrink-0" />
                <RunSelector
                    label="Run B"
                    runs={completedRuns}
                    selected={selectedRuns[1]}
                    onChange={id => setSelectedRuns([selectedRuns[0], id])}
                    accent="violet"
                />
                <button
                    onClick={handleCompare}
                    disabled={loading || !canCompare}
                    className="shrink-0 flex items-center gap-1.5 px-4 py-2 bg-blue-50 border border-blue-200 text-blue-700 rounded-xl text-sm font-medium hover:bg-blue-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {loading ? <Loader className="w-3.5 h-3.5 animate-spin" /> : <GitCompare className="w-3.5 h-3.5" />}
                    Compare
                </button>
            </div>

            {error && (
                <div className="bg-danger-400/10 border border-danger-400/30 text-danger-600 text-sm rounded-xl p-3 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0" /> {error}
                </div>
            )}

            {/* Comparison results */}
            {comparison && (
                <div className="space-y-5">
                    {/* Metrics table */}
                    <div>
                        <h4 className="text-xs font-semibold text-surface-500 uppercase tracking-wide mb-2">
                            Metrics Comparison
                        </h4>
                        <div className="border border-surface-200 rounded-xl overflow-hidden">
                            {/* Header row */}
                            <div className="grid grid-cols-3 bg-surface-50 border-b border-surface-200">
                                <div className="px-3 py-2 text-xs text-surface-500 font-medium">Metric</div>
                                <div className="px-3 py-2 text-xs text-blue-600 font-semibold">Run {comparison.runs[0]?.run_id}</div>
                                <div className="px-3 py-2 text-xs text-violet-600 font-semibold">Run {comparison.runs[1]?.run_id}</div>
                            </div>

                            {[
                                { key: 'best_loss', label: 'Best Loss', format: v => v?.toFixed(4) ?? '—' },
                                { key: 'current_loss', label: 'Final Loss', format: v => v?.toFixed(4) ?? '—' },
                                { key: 'validation_loss', label: 'Eval Loss', format: v => v?.toFixed(4) ?? '—' },
                                { key: 'perplexity', label: 'Perplexity', format: v => v?.toFixed(2) ?? '—' },
                                { key: 'tokens_per_sec', label: 'Tokens/sec', format: v => v?.toFixed(1) ?? '—' },
                                { key: 'total_steps', label: 'Total Steps', format: v => v?.toLocaleString() ?? '—' },
                            ].map(({ key, label, format }) => {
                                const a = comparison.runs[0]?.metrics?.[key];
                                const b = comparison.runs[1]?.metrics?.[key];
                                const betterIsLower = ['best_loss', 'current_loss', 'validation_loss', 'perplexity'].includes(key);
                                const aWins = a != null && b != null && (betterIsLower ? a < b : a > b);
                                const bWins = a != null && b != null && (betterIsLower ? b < a : b > a);

                                return (
                                    <div key={key} className="grid grid-cols-3 border-b border-surface-100 last:border-b-0">
                                        <div className="px-3 py-2 text-xs text-surface-600">{label}</div>
                                        <div className={`px-3 py-2 text-sm tabular-nums ${aWins ? 'text-emerald-600 font-bold bg-emerald-50/50' : 'text-surface-800'}`}>
                                            {format(a)} {aWins && '✓'}
                                        </div>
                                        <div className={`px-3 py-2 text-sm tabular-nums ${bWins ? 'text-emerald-600 font-bold bg-emerald-50/50' : 'text-surface-800'}`}>
                                            {format(b)} {bWins && '✓'}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Config diff */}
                    {comparison.config_diff?.length > 0 && (
                        <div>
                            <h4 className="text-xs font-semibold text-surface-500 uppercase tracking-wide mb-2">
                                Configuration Diff
                            </h4>
                            <div className="border border-surface-200 rounded-xl overflow-hidden">
                                {comparison.config_diff.filter(d => d.different).length > 0 ? (
                                    comparison.config_diff.filter(d => d.different).map((diff, i) => (
                                        <div key={i} className="grid grid-cols-3 border-b border-surface-100 last:border-b-0">
                                            <div className="px-3 py-2 text-xs text-amber-600 font-semibold">{diff.key}</div>
                                            <div className="px-3 py-2 text-xs text-blue-600">{String(diff.run_a ?? '—')}</div>
                                            <div className="px-3 py-2 text-xs text-violet-600">{String(diff.run_b ?? '—')}</div>
                                        </div>
                                    ))
                                ) : (
                                    <div className="px-4 py-3 text-sm text-surface-500 text-center">
                                        Configurations are identical
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {!comparison && !loading && (
                <div className="text-sm text-surface-400 text-center py-6">
                    Select two training runs to compare their metrics and configurations
                </div>
            )}
        </div>
    );
}

function RunSelector({ label, runs, selected, onChange, accent }) {
    const colorClasses = accent === 'blue'
        ? { label: 'text-blue-600', ring: 'focus:ring-blue-500/30', border: selected ? 'border-blue-300' : 'border-surface-200' }
        : { label: 'text-violet-600', ring: 'focus:ring-violet-500/30', border: selected ? 'border-violet-300' : 'border-surface-200' };

    return (
        <div className="flex-1">
            <div className={`text-xs font-semibold mb-1 ${colorClasses.label}`}>{label}</div>
            <select
                value={selected || ''}
                onChange={e => onChange(e.target.value ? parseInt(e.target.value) : null)}
                className={`w-full px-3 py-2 border ${colorClasses.border} rounded-xl text-sm bg-white focus:outline-none focus:ring-2 ${colorClasses.ring} transition-colors`}
            >
                <option value="">Select run...</option>
                {runs.map(r => (
                    <option key={r.run_id || r.id} value={r.run_id || r.id}>
                        Run #{r.run_id || r.id} — {r.status} (Loss: {r.best_loss?.toFixed(4) ?? '?'})
                    </option>
                ))}
            </select>
        </div>
    );
}
