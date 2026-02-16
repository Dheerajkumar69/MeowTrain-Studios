import { useState } from 'react';
import { datasetsAPI } from '../../services/api';
import {
    Eye, Loader, FileText, MessageSquare, Type,
    Hash, BarChart3, ChevronDown, ChevronUp, Sparkles, AlertCircle
} from 'lucide-react';

const FORMAT_BADGES = {
    alpaca: { label: 'Alpaca', color: 'bg-blue-50 text-blue-700 border-blue-200' },
    sharegpt: { label: 'ShareGPT', color: 'bg-purple-50 text-purple-700 border-purple-200' },
    openai_messages: { label: 'OpenAI Messages', color: 'bg-emerald-50 text-emerald-700 border-emerald-200' },
    question_answer: { label: 'Q&A', color: 'bg-amber-50 text-amber-700 border-amber-200' },
    prompt_completion: { label: 'Prompt/Completion', color: 'bg-cyan-50 text-cyan-700 border-cyan-200' },
    plain_text: { label: 'Raw Text', color: 'bg-surface-50 text-surface-700 border-surface-200' },
    csv_instruction: { label: 'CSV (Instruction)', color: 'bg-blue-50 text-blue-700 border-blue-200' },
    csv_text: { label: 'CSV (Text)', color: 'bg-surface-50 text-surface-700 border-surface-200' },
    unknown: { label: 'Unknown', color: 'bg-surface-50 text-surface-500 border-surface-200' },
    error: { label: 'Error', color: 'bg-danger-400/10 text-danger-600 border-danger-400/30' },
};

export default function DatasetPreviewPanel({ projectId, selectedModel }) {
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [expandedFile, setExpandedFile] = useState(null);
    const [expandedSample, setExpandedSample] = useState(null);
    const [showTemplated, setShowTemplated] = useState(true);

    const loadPreview = async () => {
        setLoading(true);
        setError('');
        try {
            const res = await datasetsAPI.previewTraining(projectId, {
                base_model: selectedModel || undefined,
                max_tokens: 512,
            });
            setPreview(res.data);
            if (res.data.datasets?.length > 0) {
                setExpandedFile(res.data.datasets[0].dataset_id);
            }
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to load preview');
        } finally {
            setLoading(false);
        }
    };

    if (!preview && !loading) {
        return (
            <div className="bg-white rounded-2xl border border-surface-100 p-6">
                <div className="text-center">
                    <div className="w-12 h-12 rounded-xl bg-violet-50 flex items-center justify-center mx-auto mb-3">
                        <Eye className="w-6 h-6 text-violet-500" />
                    </div>
                    <h3 className="text-sm font-semibold text-surface-800 mb-1">Training Preview</h3>
                    <p className="text-xs text-surface-500 mb-4 max-w-xs mx-auto">
                        See how your data will look during training — format detection, template preview, and token stats.
                    </p>
                    <button
                        onClick={loadPreview}
                        className="px-4 py-2 bg-violet-500 text-white rounded-xl text-sm font-medium hover:bg-violet-600 transition-colors flex items-center gap-2 mx-auto"
                    >
                        <Sparkles className="w-4 h-4" /> Analyse Datasets
                    </button>
                </div>
            </div>
        );
    }

    if (loading) {
        return (
            <div className="bg-white rounded-2xl border border-surface-100 p-6 flex items-center justify-center">
                <Loader className="w-5 h-5 animate-spin text-violet-500 mr-2" />
                <span className="text-sm text-surface-600">Analysing datasets...</span>
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-white rounded-2xl border border-danger-400/30 p-6">
                <div className="flex items-center gap-2 text-danger-600 mb-3">
                    <AlertCircle className="w-4 h-4" />
                    <span className="text-sm font-medium">Preview Error</span>
                </div>
                <p className="text-xs text-surface-600 mb-3">{error}</p>
                <button onClick={loadPreview} className="text-xs text-violet-600 hover:underline">
                    Try again
                </button>
            </div>
        );
    }

    const { datasets, summary } = preview;

    return (
        <div className="space-y-4">
            {/* Summary Stats */}
            <div className="bg-white rounded-2xl border border-surface-100 p-4">
                <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-surface-800 flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-violet-500" /> Training Data Summary
                    </h3>
                    <button
                        onClick={loadPreview}
                        className="text-xs text-violet-600 hover:text-violet-700 flex items-center gap-1"
                    >
                        <Eye className="w-3 h-3" /> Refresh
                    </button>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div className="bg-surface-50 rounded-xl px-3 py-2">
                        <p className="text-xs text-surface-500">Examples</p>
                        <p className="text-lg font-bold text-surface-800">{summary.total_examples.toLocaleString()}</p>
                    </div>
                    <div className="bg-surface-50 rounded-xl px-3 py-2">
                        <p className="text-xs text-surface-500">Total Tokens</p>
                        <p className="text-lg font-bold text-surface-800">{summary.total_tokens.toLocaleString()}</p>
                    </div>
                    <div className="bg-surface-50 rounded-xl px-3 py-2">
                        <p className="text-xs text-surface-500">Avg Tokens</p>
                        <p className="text-lg font-bold text-surface-800">{summary.avg_tokens}</p>
                    </div>
                    <div className="bg-surface-50 rounded-xl px-3 py-2">
                        <p className="text-xs text-surface-500">Instruct / Text</p>
                        <p className="text-lg font-bold text-surface-800">
                            {summary.instruction_examples} / {summary.text_only_examples}
                        </p>
                    </div>
                </div>
            </div>

            {/* Per-Dataset Preview */}
            {datasets.map((ds) => {
                const badge = FORMAT_BADGES[ds.format] || FORMAT_BADGES.unknown;
                const isExpanded = expandedFile === ds.dataset_id;

                return (
                    <div key={ds.dataset_id} className="bg-white rounded-2xl border border-surface-100 overflow-hidden">
                        {/* File Header */}
                        <button
                            onClick={() => setExpandedFile(isExpanded ? null : ds.dataset_id)}
                            className="w-full px-4 py-3 flex items-center justify-between hover:bg-surface-50 transition-colors"
                        >
                            <div className="flex items-center gap-3 min-w-0">
                                <FileText className="w-4 h-4 text-surface-400 shrink-0" />
                                <span className="text-sm font-medium text-surface-800 truncate">{ds.filename}</span>
                                <span className={`text-xs px-2 py-0.5 rounded-full border ${badge.color}`}>
                                    {badge.label}
                                </span>
                                <span className="text-xs text-surface-400">{ds.sample_count} examples</span>
                            </div>
                            {isExpanded ?
                                <ChevronUp className="w-4 h-4 text-surface-400 shrink-0" /> :
                                <ChevronDown className="w-4 h-4 text-surface-400 shrink-0" />
                            }
                        </button>

                        {isExpanded && (
                            <div className="border-t border-surface-100 px-4 py-3 space-y-3">
                                <p className="text-xs text-surface-500">{ds.format_description}</p>

                                {/* Toggle for template view */}
                                {ds.samples?.some(s => s.templated) && (
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => setShowTemplated(false)}
                                            className={`text-xs px-2.5 py-1 rounded-lg transition-colors ${!showTemplated ? 'bg-surface-800 text-white' : 'bg-surface-100 text-surface-600 hover:bg-surface-200'}`}
                                        >
                                            <Type className="w-3 h-3 inline mr-1" />Raw
                                        </button>
                                        <button
                                            onClick={() => setShowTemplated(true)}
                                            className={`text-xs px-2.5 py-1 rounded-lg transition-colors ${showTemplated ? 'bg-violet-500 text-white' : 'bg-surface-100 text-surface-600 hover:bg-surface-200'}`}
                                        >
                                            <Sparkles className="w-3 h-3 inline mr-1" />Templated
                                        </button>
                                    </div>
                                )}

                                {/* Sample Examples */}
                                {ds.samples?.map((sample, idx) => (
                                    <div key={idx} className="border border-surface-100 rounded-xl overflow-hidden">
                                        <button
                                            onClick={() => setExpandedSample(
                                                expandedSample === `${ds.dataset_id}-${idx}` ? null : `${ds.dataset_id}-${idx}`
                                            )}
                                            className="w-full px-3 py-2 flex items-center justify-between bg-surface-50 hover:bg-surface-100 transition-colors"
                                        >
                                            <div className="flex items-center gap-2">
                                                {sample.has_messages ? (
                                                    <MessageSquare className="w-3 h-3 text-violet-500" />
                                                ) : (
                                                    <Type className="w-3 h-3 text-surface-400" />
                                                )}
                                                <span className="text-xs font-medium text-surface-700">
                                                    Example {idx + 1}
                                                </span>
                                                <span className="text-xs text-surface-400 flex items-center gap-1">
                                                    <Hash className="w-3 h-3" />{sample.token_count} tokens
                                                </span>
                                            </div>
                                            {expandedSample === `${ds.dataset_id}-${idx}` ?
                                                <ChevronUp className="w-3 h-3 text-surface-400" /> :
                                                <ChevronDown className="w-3 h-3 text-surface-400" />
                                            }
                                        </button>
                                        {expandedSample === `${ds.dataset_id}-${idx}` && (
                                            <div className="px-3 py-2">
                                                {showTemplated && sample.templated ? (
                                                    <div>
                                                        <p className="text-xs text-violet-600 font-medium mb-1.5 flex items-center gap-1">
                                                            <Sparkles className="w-3 h-3" /> What the model sees:
                                                        </p>
                                                        <pre className="text-xs text-surface-700 bg-violet-50 border border-violet-100 rounded-lg p-3 whitespace-pre-wrap font-mono leading-relaxed overflow-auto max-h-64">
                                                            {sample.templated}
                                                        </pre>
                                                        {sample.templated_tokens && (
                                                            <p className="text-xs text-surface-400 mt-1">{sample.templated_tokens} tokens after template</p>
                                                        )}
                                                    </div>
                                                ) : (
                                                    <pre className="text-xs text-surface-700 bg-surface-50 rounded-lg p-3 whitespace-pre-wrap font-mono leading-relaxed overflow-auto max-h-64">
                                                        {sample.raw}
                                                    </pre>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
}
