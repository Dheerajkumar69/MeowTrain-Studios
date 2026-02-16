import { useState } from 'react';
import { Sparkles, Trash2, AlignLeft, ShieldCheck, Eye, Loader, CheckCircle, AlertCircle, ChevronDown } from 'lucide-react';
import { datasetsAPI } from '../../services/api';

/**
 * DatasetAugmentPanel — Tools for cleaning, deduplicating, and filtering datasets.
 *
 * Shows toggle switches for each operation, runs a preview showing before/after
 * and stats, then optionally applies changes to create a new augmented dataset.
 */
export default function DatasetAugmentPanel({ projectId, onAugmented }) {
    const [options, setOptions] = useState({
        enable_dedup: true,
        enable_clean: true,
        enable_filter: true,
        dedup_threshold: 0.85,
        min_length: 10,
        max_length: 100000,
        strip_urls: false,
        strip_emails: false,
    });

    const [previewResult, setPreviewResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [applying, setApplying] = useState(false);
    const [error, setError] = useState(null);
    const [applied, setApplied] = useState(false);

    const handlePreview = async () => {
        setLoading(true);
        setError(null);
        setApplied(false);
        try {
            const res = await datasetsAPI.augmentPreview(projectId, options);
            setPreviewResult(res.data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to preview augmentation');
        } finally {
            setLoading(false);
        }
    };

    const handleApply = async () => {
        setApplying(true);
        setError(null);
        try {
            const res = await datasetsAPI.augment(projectId, options);
            setPreviewResult(res.data);
            setApplied(true);
            onAugmented?.();
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to apply augmentation');
        } finally {
            setApplying(false);
        }
    };

    const stats = previewResult?.stats;

    return (
        <div className="bg-white rounded-2xl border border-surface-100 p-5 space-y-5">
            {/* Header */}
            <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-violet-50 flex items-center justify-center">
                    <Sparkles className="w-4 h-4 text-violet-500" />
                </div>
                <div>
                    <h3 className="text-sm font-semibold text-surface-900">Dataset Augmentation</h3>
                    <p className="text-xs text-surface-500">Clean, deduplicate, and filter your training data</p>
                </div>
            </div>

            {/* Toggle switches */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <ToggleCard
                    icon={<Trash2 className="w-3.5 h-3.5" />}
                    label="Deduplicate"
                    description="Remove near-duplicate examples"
                    active={options.enable_dedup}
                    onChange={() => setOptions(o => ({ ...o, enable_dedup: !o.enable_dedup }))}
                />
                <ToggleCard
                    icon={<AlignLeft className="w-3.5 h-3.5" />}
                    label="Clean Text"
                    description="Fix HTML, encoding, whitespace"
                    active={options.enable_clean}
                    onChange={() => setOptions(o => ({ ...o, enable_clean: !o.enable_clean }))}
                />
                <ToggleCard
                    icon={<ShieldCheck className="w-3.5 h-3.5" />}
                    label="Quality Filter"
                    description="Remove short/repetitive text"
                    active={options.enable_filter}
                    onChange={() => setOptions(o => ({ ...o, enable_filter: !o.enable_filter }))}
                />
            </div>

            {/* Advanced options */}
            <details className="group">
                <summary className="text-xs text-surface-500 cursor-pointer hover:text-surface-700 transition-colors flex items-center gap-1">
                    <ChevronDown className="w-3 h-3 transition-transform group-open:rotate-180" />
                    Advanced Options
                </summary>
                <div className="grid grid-cols-2 gap-3 mt-3 p-4 bg-surface-50 rounded-xl border border-surface-200">
                    <label className="flex flex-col gap-1 text-xs text-surface-600">
                        <span className="font-medium">Dedup Threshold</span>
                        <input
                            type="number"
                            step="0.05"
                            min="0.5"
                            max="1.0"
                            value={options.dedup_threshold}
                            onChange={e => setOptions(o => ({ ...o, dedup_threshold: parseFloat(e.target.value) || 0.85 }))}
                            className="px-3 py-2 border border-surface-200 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-violet-500/30"
                        />
                    </label>
                    <label className="flex flex-col gap-1 text-xs text-surface-600">
                        <span className="font-medium">Min Length (chars)</span>
                        <input
                            type="number"
                            min="1"
                            value={options.min_length}
                            onChange={e => setOptions(o => ({ ...o, min_length: parseInt(e.target.value) || 10 }))}
                            className="px-3 py-2 border border-surface-200 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-violet-500/30"
                        />
                    </label>
                    <label className="flex items-center gap-2 text-xs text-surface-600 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={options.strip_urls}
                            onChange={e => setOptions(o => ({ ...o, strip_urls: e.target.checked }))}
                            className="rounded border-surface-300 text-violet-500 focus:ring-violet-500"
                        />
                        <span className="font-medium">Strip URLs</span>
                    </label>
                    <label className="flex items-center gap-2 text-xs text-surface-600 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={options.strip_emails}
                            onChange={e => setOptions(o => ({ ...o, strip_emails: e.target.checked }))}
                            className="rounded border-surface-300 text-violet-500 focus:ring-violet-500"
                        />
                        <span className="font-medium">Strip Emails</span>
                    </label>
                </div>
            </details>

            {/* Preview button */}
            <button
                onClick={handlePreview}
                disabled={loading}
                className="w-full py-2.5 bg-violet-50 border border-violet-200 text-violet-700 rounded-xl text-sm font-medium hover:bg-violet-100 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
            >
                {loading ? <Loader className="w-4 h-4 animate-spin" /> : <Eye className="w-4 h-4" />}
                {loading ? 'Analyzing...' : 'Preview Changes'}
            </button>

            {error && (
                <div className="bg-danger-400/10 border border-danger-400/30 text-danger-600 text-sm rounded-xl p-3 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0" /> {error}
                </div>
            )}

            {/* Results */}
            {stats && (
                <div className="space-y-4">
                    {/* Stats grid */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                        <StatCard label="Original" value={stats.original_count} />
                        <StatCard label="After Augmentation" value={stats.final_count} className="text-emerald-600" />
                        <StatCard label="Duplicates Removed" value={stats.duplicates_removed} className="text-amber-600" />
                        <StatCard label="Filtered Out" value={stats.filtered_out} className="text-danger-500" />
                    </div>

                    {/* Reduction bar */}
                    <div className="bg-surface-50 rounded-xl p-4 border border-surface-200">
                        <div className="flex justify-between text-xs text-surface-600 mb-2">
                            <span className="font-medium">Data reduction</span>
                            <span className={stats.reduction_percent > 20 ? 'text-amber-600 font-medium' : 'text-emerald-600 font-medium'}>
                                {stats.reduction_percent}% removed
                            </span>
                        </div>
                        <div className="h-2 bg-surface-200 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-emerald-400 to-emerald-300 rounded-full transition-all duration-500"
                                style={{ width: `${100 - stats.reduction_percent}%` }}
                            />
                        </div>
                    </div>

                    {/* Sample changes */}
                    {previewResult.samples?.length > 0 && (
                        <details className="group">
                            <summary className="text-xs text-surface-500 cursor-pointer hover:text-surface-700 transition-colors flex items-center gap-1">
                                <ChevronDown className="w-3 h-3 transition-transform group-open:rotate-180" />
                                Sample Changes ({previewResult.samples.length})
                            </summary>
                            <div className="mt-2 space-y-2">
                                {previewResult.samples.map((s, i) => (
                                    <div
                                        key={i}
                                        className={`rounded-xl border p-3 text-xs ${s.changed
                                            ? 'border-amber-200 bg-amber-50/50'
                                            : 'border-surface-100 bg-surface-50'
                                            }`}
                                    >
                                        <div className="text-surface-500 mb-1.5 font-medium">
                                            Example {s.index + 1} {s.changed && <span className="text-amber-600">• modified</span>}
                                        </div>
                                        <p className="text-surface-700 whitespace-pre-wrap leading-relaxed line-clamp-3">
                                            {s.after}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </details>
                    )}

                    {/* Apply button */}
                    {!applied ? (
                        <button
                            onClick={handleApply}
                            disabled={applying || stats.final_count === 0}
                            className="w-full py-2.5 bg-emerald-50 border border-emerald-200 text-emerald-700 rounded-xl text-sm font-medium hover:bg-emerald-100 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                        >
                            {applying ? <Loader className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                            {applying ? 'Applying...' : `Apply & Create Augmented Dataset (${stats.final_count} examples)`}
                        </button>
                    ) : (
                        <div className="flex items-center gap-2 text-emerald-600 text-sm font-medium py-2.5 px-4 bg-emerald-50 rounded-xl border border-emerald-200">
                            <CheckCircle className="w-4 h-4" />
                            Augmented dataset created successfully!
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

function ToggleCard({ icon, label, description, active, onChange }) {
    return (
        <button
            onClick={onChange}
            className={`p-3 rounded-xl border text-left transition-all ${active
                ? 'border-violet-300 bg-violet-50'
                : 'border-surface-200 bg-surface-50 hover:border-surface-300'
                }`}
        >
            <div className={`flex items-center gap-1.5 mb-1 ${active ? 'text-violet-600' : 'text-surface-500'}`}>
                {icon}
                <span className="text-xs font-semibold">{label}</span>
            </div>
            <div className="text-xs text-surface-500">{description}</div>
        </button>
    );
}

function StatCard({ label, value, className = 'text-surface-900' }) {
    return (
        <div className="bg-surface-50 rounded-xl border border-surface-200 p-3 text-center">
            <div className={`text-xl font-bold tabular-nums ${className}`}>
                {value?.toLocaleString() ?? '—'}
            </div>
            <div className="text-xs text-surface-500 mt-0.5">{label}</div>
        </div>
    );
}
