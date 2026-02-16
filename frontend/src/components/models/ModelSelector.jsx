import { useState, useEffect, useRef, useCallback } from 'react';
import { modelsAPI } from '../../services/api';
import { Check, Download, AlertTriangle, XCircle, Loader, Info, Search, ExternalLink, X, Trash2, HardDrive } from 'lucide-react';

const COMPAT_CONFIG = {
    compatible: { label: 'Perfect fit', color: 'text-success-600 bg-success-400/10', icon: Check },
    may_be_slow: { label: 'May be slow', color: 'text-amber-600 bg-amber-400/10', icon: AlertTriangle },
    too_large: { label: 'Too large', color: 'text-danger-600 bg-danger-400/10', icon: XCircle },
    unknown: { label: 'Unknown', color: 'text-surface-500 bg-surface-100', icon: Info },
};

export default function ModelSelector({ selectedModel, onSelectModel }) {
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    // Download state: { [modelId]: { status, progress, message, error } }
    const [downloads, setDownloads] = useState({});
    const pollRefs = useRef({});

    // Custom model state
    const [customModelId, setCustomModelId] = useState('');
    const [customLooking, setCustomLooking] = useState(false);
    const [customError, setCustomError] = useState('');
    const [customModel, setCustomModel] = useState(null);
    const debounceRef = useRef(null);

    const loadModels = useCallback(() => {
        modelsAPI.list()
            .then((res) => setModels(res.data || []))
            .catch(() => { setError('Failed to load models. Check backend connection.'); })
            .finally(() => setLoading(false));
    }, []);

    useEffect(() => {
        loadModels();
    }, [loadModels]);

    // Cleanup polling on unmount
    useEffect(() => {
        return () => {
            Object.values(pollRefs.current).forEach(clearInterval);
        };
    }, []);

    // ── Download management ─────────────────────────────────────
    const startDownload = useCallback(async (modelId) => {
        setError('');
        setDownloads((prev) => ({
            ...prev,
            [modelId]: { status: 'starting', progress: 0, message: 'Starting download...' },
        }));
        try {
            const res = await modelsAPI.download(modelId);
            if (res.data?.status === 'cached') {
                // Already cached — just refresh the list
                setDownloads((prev) => {
                    const next = { ...prev };
                    delete next[modelId];
                    return next;
                });
                loadModels();
                return;
            }
            // Start polling for progress
            pollDownloadProgress(modelId);
        } catch (err) {
            setDownloads((prev) => ({
                ...prev,
                [modelId]: {
                    status: 'error',
                    progress: 0,
                    message: err.response?.data?.detail || 'Failed to start download',
                    error: err.response?.data?.detail || 'Failed to start download',
                },
            }));
        }
    }, [loadModels]);

    const pollDownloadProgress = useCallback((modelId) => {
        // Clear any existing poll for this model
        if (pollRefs.current[modelId]) clearInterval(pollRefs.current[modelId]);

        const poll = async () => {
            try {
                const res = await modelsAPI.downloadProgress(modelId);
                const data = res.data;
                setDownloads((prev) => ({
                    ...prev,
                    [modelId]: {
                        status: data.status,
                        progress: data.progress || 0,
                        message: data.message || '',
                        error: data.error,
                        elapsed_seconds: data.elapsed_seconds,
                    },
                }));

                if (data.status === 'completed' || data.status === 'cached') {
                    clearInterval(pollRefs.current[modelId]);
                    delete pollRefs.current[modelId];
                    // Refresh model list to update is_cached
                    loadModels();
                    // Auto-clear after 5s
                    setTimeout(() => {
                        setDownloads((prev) => {
                            const next = { ...prev };
                            delete next[modelId];
                            return next;
                        });
                    }, 5000);
                } else if (data.status === 'error' || data.status === 'cancelled') {
                    clearInterval(pollRefs.current[modelId]);
                    delete pollRefs.current[modelId];
                }
            } catch {
                // Keep polling on transient errors
            }
        };

        poll(); // Initial fetch
        pollRefs.current[modelId] = setInterval(poll, 2000);
    }, [loadModels]);

    const cancelDownload = useCallback(async (modelId) => {
        try {
            await modelsAPI.cancelDownload(modelId);
            if (pollRefs.current[modelId]) {
                clearInterval(pollRefs.current[modelId]);
                delete pollRefs.current[modelId];
            }
            setDownloads((prev) => ({
                ...prev,
                [modelId]: { status: 'cancelled', progress: 0, message: 'Download cancelled.' },
            }));
            setTimeout(() => {
                setDownloads((prev) => {
                    const next = { ...prev };
                    delete next[modelId];
                    return next;
                });
            }, 3000);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to cancel download');
        }
    }, []);

    const deleteCache = useCallback(async (modelId) => {
        try {
            const res = await modelsAPI.deleteCache(modelId);
            setError('');
            loadModels();
            // If deleted model was selected, clear selection
            if (selectedModel?.model_id === modelId) {
                onSelectModel(null);
            }
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to delete cached model');
        }
    }, [loadModels, selectedModel, onSelectModel]);

    const dismissDownload = useCallback((modelId) => {
        setDownloads((prev) => {
            const next = { ...prev };
            delete next[modelId];
            return next;
        });
    }, []);

    const formatElapsed = (secs) => {
        if (!secs) return '';
        const m = Math.floor(secs / 60);
        const s = secs % 60;
        return m > 0 ? `${m}m ${s}s` : `${s}s`;
    };

    const lookupCustomModel = useCallback(async () => {
        const trimmed = customModelId.trim();
        if (!trimmed) return;
        // Validate format: org/model-name
        if (!/^[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+$/.test(trimmed)) {
            setCustomError('Model ID must be in "org/name" format (e.g. meta-llama/Llama-3.2-3B)');
            return;
        }
        setCustomLooking(true);
        setCustomError('');
        setCustomModel(null);
        try {
            const res = await modelsAPI.lookupCustom(trimmed);
            setCustomModel(res.data);
        } catch (err) {
            if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
                setCustomError('Request timed out. Please try again.');
            } else {
                setCustomError(err.response?.data?.detail || 'Model not found on HuggingFace Hub.');
            }
        } finally {
            setCustomLooking(false);
        }
    }, [customModelId]);

    // Debounced lookup on Enter (prevents rapid fire)
    const debouncedLookup = useCallback(() => {
        if (debounceRef.current) clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(lookupCustomModel, 300);
    }, [lookupCustomModel]);

    if (loading) {
        return (
            <div className="flex items-center justify-center py-20">
                <Loader className="w-6 h-6 text-primary-500 animate-spin" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-xl font-bold text-surface-900">Choose a Base Model</h2>
                <p className="text-sm text-surface-500 mt-1">
                    Select from the catalog or use any HuggingFace model.
                </p>
            </div>

            {error && (
                <div className="bg-danger-400/10 border border-danger-400/30 text-danger-600 text-sm rounded-xl p-4 flex items-center gap-2">
                    <XCircle className="w-4 h-4 shrink-0" /> {error}
                </div>
            )}

            {/* Custom Model Input */}
            <div className="bg-gradient-to-r from-primary-50 to-primary-100/50 rounded-2xl border border-primary-200 p-5">
                <h3 className="text-sm font-semibold text-primary-800 mb-1 flex items-center gap-2">
                    🤗 Use Any HuggingFace Model
                </h3>
                <p className="text-xs text-primary-600 mb-3">
                    Enter a model ID from HuggingFace Hub (e.g. <code className="bg-primary-200/50 px-1 rounded">meta-llama/Llama-3.2-3B</code>)
                </p>
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={customModelId}
                        onChange={(e) => setCustomModelId(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && debouncedLookup()}
                        placeholder="org/model-name"
                        className="flex-1 px-4 py-2.5 bg-white border border-primary-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30 placeholder:text-surface-400"
                    />
                    <button
                        onClick={debouncedLookup}
                        disabled={customLooking || !customModelId.trim()}
                        className="px-5 py-2.5 bg-primary-500 text-white rounded-xl text-sm font-medium hover:bg-primary-600 transition-colors disabled:opacity-50 flex items-center gap-2"
                    >
                        {customLooking ? (
                            <Loader className="w-4 h-4 animate-spin" />
                        ) : (
                            <Search className="w-4 h-4" />
                        )}
                        Lookup
                    </button>
                </div>

                {customError && (
                    <div className="mt-3 bg-danger-400/10 border border-danger-400/30 text-danger-600 text-xs rounded-lg p-3 flex items-center gap-2">
                        <XCircle className="w-3.5 h-3.5 shrink-0" /> {customError}
                    </div>
                )}

                {customModel && (
                    <div className={`mt-3 bg-white rounded-xl border-2 p-4 cursor-pointer transition-all ${selectedModel?.model_id === customModel.model_id
                        ? 'border-primary-500 shadow-md ring-2 ring-primary-500/20'
                        : 'border-surface-200 hover:border-primary-300'
                        }`}
                        onClick={() => onSelectModel(customModel)}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelectModel(customModel); } }}
                        role="button"
                        tabIndex={0}
                        aria-pressed={selectedModel?.model_id === customModel.model_id}
                        aria-label={`Select custom model ${customModel.name}`}
                    >
                        <div className="flex items-start justify-between mb-2">
                            <div className="flex items-center gap-2">
                                <span className="text-2xl">{customModel.icon}</span>
                                <div>
                                    <h4 className="font-semibold text-surface-900 text-sm">{customModel.name}</h4>
                                    <p className="text-xs text-surface-500">{customModel.parameters} params • {customModel.size_gb} GB</p>
                                </div>
                            </div>
                            {selectedModel?.model_id === customModel.model_id && (
                                <div className="w-5 h-5 rounded-full bg-primary-500 flex items-center justify-center">
                                    <Check className="w-3 h-3 text-white" />
                                </div>
                            )}
                        </div>
                        <p className="text-xs text-surface-600 mb-2">{customModel.description}</p>
                        <div className="flex items-center justify-between">
                            {(() => {
                                const compat = COMPAT_CONFIG[customModel.compatibility] || COMPAT_CONFIG.unknown;
                                const CompatIcon = compat.icon;
                                return (
                                    <span className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full ${compat.color}`}>
                                        <CompatIcon className="w-3 h-3" /> {compat.label}
                                    </span>
                                );
                            })()}
                            <div className="flex items-center gap-3">
                                {customModel.is_cached ? (
                                    <span className="text-xs text-success-600 font-medium flex items-center gap-1">
                                        <Check className="w-3 h-3" /> Cached
                                    </span>
                                ) : (
                                    <span className="text-xs text-surface-400 flex items-center gap-1">
                                        <Download className="w-3 h-3" /> Needs download
                                    </span>
                                )}
                                <a
                                    href={`https://huggingface.co/${customModel.model_id}`}
                                    target="_blank"
                                    rel="noreferrer"
                                    onClick={(e) => e.stopPropagation()}
                                    className="text-xs text-primary-500 hover:text-primary-700 flex items-center gap-1"
                                >
                                    <ExternalLink className="w-3 h-3" /> View on HuggingFace
                                </a>
                            </div>
                        </div>

                        {/* Download progress for custom model */}
                        {(() => {
                            const dl = downloads[customModel.model_id];
                            if (!dl) return null;
                            const isDownloading = dl.status === 'downloading' || dl.status === 'starting' || dl.status === 'running';
                            const downloadDone = dl.status === 'completed' || dl.status === 'cached';
                            const downloadErr = dl.status === 'error';
                            return (
                                <div className="mt-2 bg-surface-50 rounded-lg p-2.5 border border-surface-100" onClick={(e) => e.stopPropagation()}>
                                    {isDownloading && (
                                        <>
                                            <div className="flex items-center justify-between mb-1">
                                                <span className="text-xs font-medium text-primary-700 flex items-center gap-1">
                                                    <Loader className="w-3 h-3 animate-spin" /> Downloading...
                                                </span>
                                                <button onClick={() => cancelDownload(customModel.model_id)} className="text-surface-400 hover:text-danger-500 transition-colors">
                                                    <X className="w-3.5 h-3.5" />
                                                </button>
                                            </div>
                                            <div className="w-full h-1.5 bg-surface-200 rounded-full overflow-hidden">
                                                <div className="h-full bg-primary-500 rounded-full transition-all duration-500" style={{ width: `${Math.max(dl.progress || 0, 2)}%` }} />
                                            </div>
                                            <p className="text-xs text-surface-500 mt-1 truncate">{dl.message}</p>
                                        </>
                                    )}
                                    {downloadDone && (
                                        <span className="text-xs text-success-600 flex items-center gap-1"><Check className="w-3 h-3" /> Download complete!</span>
                                    )}
                                    {downloadErr && (
                                        <span className="text-xs text-danger-600 flex items-center gap-1"><XCircle className="w-3 h-3" /> {dl.message}</span>
                                    )}
                                </div>
                            );
                        })()}

                        {/* Action buttons for custom model */}
                        <div className="flex gap-2 mt-3" onClick={(e) => e.stopPropagation()}>
                            {!customModel.is_cached && !downloads[customModel.model_id]?.status?.match(/downloading|starting|running/) && (
                                <button
                                    onClick={() => startDownload(customModel.model_id)}
                                    className="flex-1 py-1.5 rounded-lg text-xs font-medium bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-200 flex items-center justify-center gap-1"
                                >
                                    <Download className="w-3 h-3" /> Download ({customModel.size_gb} GB)
                                </button>
                            )}
                            <button
                                onClick={() => onSelectModel(customModel)}
                                className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-all ${selectedModel?.model_id === customModel.model_id
                                    ? 'bg-primary-500 text-white'
                                    : 'bg-surface-50 text-surface-700 hover:bg-primary-50 hover:text-primary-700 border border-surface-200'
                                    }`}
                            >
                                {selectedModel?.model_id === customModel.model_id ? '✓ Selected' : 'Use This Model'}
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {/* Catalog Separator */}
            <div className="flex items-center gap-3">
                <div className="h-px bg-surface-200 flex-1" />
                <span className="text-xs font-medium text-surface-400 uppercase tracking-wide">Or choose from catalog</span>
                <div className="h-px bg-surface-200 flex-1" />
            </div>

            {!error && models.length === 0 && !loading && (
                <div className="text-center py-16">
                    <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-surface-100 mb-4">
                        <Info className="w-7 h-7 text-surface-400" />
                    </div>
                    <h3 className="text-lg font-semibold text-surface-700 mb-1">No Models Available</h3>
                    <p className="text-sm text-surface-500">The model catalog is empty. Check backend configuration.</p>
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {models.map((model) => {
                    const compat = COMPAT_CONFIG[model.compatibility] || COMPAT_CONFIG.unknown;
                    const CompatIcon = compat.icon;
                    const isSelected = selectedModel?.model_id === model.model_id;
                    const dl = downloads[model.model_id];
                    const isDownloading = dl && (dl.status === 'downloading' || dl.status === 'starting' || dl.status === 'running');
                    const downloadDone = dl && (dl.status === 'completed' || dl.status === 'cached');
                    const downloadError = dl && (dl.status === 'error');
                    const downloadCancelled = dl && (dl.status === 'cancelled');

                    return (
                        <div
                            key={model.model_id}
                            onClick={() => onSelectModel(model)}
                            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelectModel(model); } }}
                            role="button"
                            tabIndex={0}
                            aria-pressed={isSelected}
                            aria-label={`Select model ${model.name}, ${model.parameters} parameters, ${compat.label}`}
                            className={`bg-white rounded-2xl border-2 p-5 cursor-pointer transition-all hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-primary-500/50 ${isSelected
                                ? 'border-primary-500 shadow-md ring-2 ring-primary-500/20'
                                : 'border-surface-100 hover:border-primary-200'
                                }`}
                        >
                            {/* Header */}
                            <div className="flex items-start justify-between mb-4">
                                <div className="flex items-center gap-3">
                                    <span className="text-3xl">{model.icon}</span>
                                    <div>
                                        <h3 className="font-semibold text-surface-900">{model.name}</h3>
                                        <p className="text-xs text-surface-500">{model.parameters} parameters</p>
                                    </div>
                                </div>
                                {isSelected && (
                                    <div className="w-6 h-6 rounded-full bg-primary-500 flex items-center justify-center">
                                        <Check className="w-3.5 h-3.5 text-white" />
                                    </div>
                                )}
                            </div>

                            {/* Description */}
                            <p className="text-sm text-surface-600 mb-4 leading-relaxed">{model.description}</p>

                            {/* Specs */}
                            <div className="space-y-2 mb-4">
                                <div className="flex justify-between text-xs">
                                    <span className="text-surface-500">Download size</span>
                                    <span className="font-medium text-surface-700">{model.size_gb} GB</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-surface-500">RAM required</span>
                                    <span className="font-medium text-surface-700">{model.ram_required_gb} GB</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-surface-500">GPU memory</span>
                                    <span className="font-medium text-surface-700">{model.vram_required_gb} GB</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-surface-500">Est. training time</span>
                                    <span className="font-medium text-surface-700">~{model.estimated_train_minutes} min</span>
                                </div>
                            </div>

                            {/* Compatibility Badge + Cache Status */}
                            <div className="flex items-center justify-between">
                                <span className={`inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full ${compat.color}`}>
                                    <CompatIcon className="w-3.5 h-3.5" />
                                    {compat.label}
                                </span>
                                {model.is_cached ? (
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs text-success-600 font-medium flex items-center gap-1">
                                            <Check className="w-3 h-3" /> Cached
                                        </span>
                                        <button
                                            onClick={(e) => { e.stopPropagation(); deleteCache(model.model_id); }}
                                            className="text-surface-400 hover:text-danger-500 transition-colors p-0.5 rounded"
                                            title="Delete cached model"
                                            aria-label={`Delete cached ${model.name}`}
                                        >
                                            <Trash2 className="w-3 h-3" />
                                        </button>
                                    </div>
                                ) : (
                                    <span className="text-xs text-surface-400 flex items-center gap-1">
                                        <Download className="w-3 h-3" /> Needs download
                                    </span>
                                )}
                            </div>

                            {/* Download Progress Bar */}
                            {dl && (
                                <div className="mt-3 bg-surface-50 rounded-xl p-3 border border-surface-100" onClick={(e) => e.stopPropagation()}>
                                    {isDownloading && (
                                        <>
                                            <div className="flex items-center justify-between mb-1.5">
                                                <span className="text-xs font-medium text-primary-700 flex items-center gap-1.5">
                                                    <Loader className="w-3 h-3 animate-spin" />
                                                    Downloading...
                                                </span>
                                                <div className="flex items-center gap-2">
                                                    {dl.elapsed_seconds > 0 && (
                                                        <span className="text-xs text-surface-400">{formatElapsed(dl.elapsed_seconds)}</span>
                                                    )}
                                                    <button
                                                        onClick={() => cancelDownload(model.model_id)}
                                                        className="text-surface-400 hover:text-danger-500 transition-colors"
                                                        title="Cancel download"
                                                    >
                                                        <X className="w-3.5 h-3.5" />
                                                    </button>
                                                </div>
                                            </div>
                                            <div className="w-full h-2 bg-surface-200 rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-gradient-to-r from-primary-400 to-primary-600 rounded-full transition-all duration-500 ease-out"
                                                    style={{ width: `${Math.max(dl.progress || 0, 2)}%` }}
                                                />
                                            </div>
                                            <p className="text-xs text-surface-500 mt-1.5 truncate">{dl.message}</p>
                                        </>
                                    )}
                                    {downloadDone && (
                                        <div className="flex items-center gap-2 text-success-600">
                                            <Check className="w-4 h-4" />
                                            <span className="text-xs font-medium">Download complete!</span>
                                        </div>
                                    )}
                                    {downloadError && (
                                        <div className="flex items-center justify-between">
                                            <span className="text-xs text-danger-600 flex items-center gap-1.5">
                                                <XCircle className="w-3.5 h-3.5" />
                                                {dl.message || 'Download failed'}
                                            </span>
                                            <div className="flex items-center gap-1.5">
                                                <button
                                                    onClick={() => { dismissDownload(model.model_id); startDownload(model.model_id); }}
                                                    className="text-xs text-primary-600 hover:text-primary-800 font-medium transition-colors"
                                                >
                                                    Retry
                                                </button>
                                                <button
                                                    onClick={() => dismissDownload(model.model_id)}
                                                    className="text-surface-400 hover:text-surface-600 transition-colors"
                                                >
                                                    <X className="w-3.5 h-3.5" />
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                    {downloadCancelled && (
                                        <div className="flex items-center gap-2 text-surface-500">
                                            <Info className="w-3.5 h-3.5" />
                                            <span className="text-xs">Download cancelled</span>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Action Buttons */}
                            <div className="flex gap-2 mt-4">
                                {!model.is_cached && !isDownloading && (
                                    <button
                                        onClick={(e) => { e.stopPropagation(); startDownload(model.model_id); }}
                                        disabled={isDownloading}
                                        className="flex-1 py-2 rounded-xl text-sm font-medium transition-all bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-200 flex items-center justify-center gap-1.5"
                                    >
                                        <Download className="w-3.5 h-3.5" />
                                        Download ({model.size_gb} GB)
                                    </button>
                                )}
                                <button
                                    onClick={(e) => { e.stopPropagation(); onSelectModel(model); }}
                                    className={`flex-1 py-2 rounded-xl text-sm font-medium transition-all ${isSelected
                                        ? 'bg-primary-500 text-white'
                                        : 'bg-surface-50 text-surface-700 hover:bg-primary-50 hover:text-primary-700 border border-surface-200'
                                        }`}
                                >
                                    {isSelected ? '✓ Selected' : 'Use This Model'}
                                </button>
                            </div>
                        </div>
                    );
                })}
            </div>

            {selectedModel && (
                <div className="bg-primary-50 border border-primary-200 rounded-xl p-4 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                        <span className="text-2xl">{selectedModel.icon}</span>
                        <div>
                            <p className="text-sm font-medium text-primary-800">
                                Selected: {selectedModel.name} ({selectedModel.parameters})
                            </p>
                            <p className="text-xs text-primary-600">
                                {selectedModel.recommended_hardware}
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        {selectedModel.is_cached ? (
                            <span className="inline-flex items-center gap-1 text-xs font-medium px-2.5 py-1 rounded-full bg-success-400/10 text-success-600">
                                <HardDrive className="w-3 h-3" /> Ready to train
                            </span>
                        ) : (
                            <button
                                onClick={() => startDownload(selectedModel.model_id)}
                                disabled={downloads[selectedModel.model_id]?.status?.match(/downloading|starting|running/)}
                                className="inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-full bg-emerald-500 text-white hover:bg-emerald-600 transition-colors disabled:opacity-50"
                            >
                                {downloads[selectedModel.model_id]?.status?.match(/downloading|starting|running/) ? (
                                    <><Loader className="w-3 h-3 animate-spin" /> Downloading...</>
                                ) : (
                                    <><Download className="w-3 h-3" /> Download Now</>
                                )}
                            </button>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
