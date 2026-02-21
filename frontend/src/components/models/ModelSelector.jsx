import { useState, useEffect, useRef, useCallback } from 'react';
import { modelsAPI } from '../../services/api';
import { Check, Download, AlertTriangle, XCircle, Loader, Info, Search, ExternalLink, X, Trash2, HardDrive, Wifi, WifiOff, HardDriveDownload, RefreshCw, Shield, ChevronDown } from 'lucide-react';

const COMPAT_CONFIG = {
    compatible: { label: 'Perfect fit', color: 'text-success-600 bg-success-400/10', icon: Check },
    may_be_slow: { label: 'May be slow', color: 'text-amber-600 bg-amber-400/10', icon: AlertTriangle },
    too_large: { label: 'Too large', color: 'text-danger-600 bg-danger-400/10', icon: XCircle },
    unknown: { label: 'Unknown', color: 'text-surface-500 bg-surface-100', icon: Info },
};

// Retry helper for transient API failures
async function withRetry(fn, { retries = 2, delay = 1000, onRetry } = {}) {
    let lastErr;
    for (let i = 0; i <= retries; i++) {
        try {
            return await fn();
        } catch (err) {
            lastErr = err;
            const status = err.response?.status;
            // Don't retry on auth/client errors
            if (status && status >= 400 && status < 500 && status !== 429 && status !== 408) throw err;
            if (i < retries) {
                onRetry?.(i + 1, retries + 1);
                await new Promise(r => setTimeout(r, delay * (i + 1)));
            }
        }
    }
    throw lastErr;
}

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

    // HuggingFace Search state
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState([]);
    const [searching, setSearching] = useState(false);
    const [searchError, setSearchError] = useState('');
    const [showSearch, setShowSearch] = useState(false);
    const searchDebounceRef = useRef(null);

    // Preflight / readiness state
    const [preflight, setPreflight] = useState(null);
    const [preflightLoading, setPreflightLoading] = useState(false);

    // Load models with retry
    const loadModels = useCallback(() => {
        withRetry(() => modelsAPI.list(), { retries: 2, delay: 1500 })
            .then((res) => { setModels(res.data || []); setError(''); })
            .catch((err) => {
                const status = err.response?.status;
                if (!err.response) {
                    setError('Cannot reach the server. Check that the backend is running.');
                } else if (status === 502 || status === 503) {
                    setError('Server is starting up. Retrying...');
                    setTimeout(loadModels, 3000);
                } else {
                    setError('Failed to load models. Check backend connection.');
                }
            })
            .finally(() => setLoading(false));
    }, []);

    useEffect(() => {
        loadModels();
    }, [loadModels]);

    // Run preflight check on mount (non-blocking)
    useEffect(() => {
        modelsAPI.preflight()
            .then(res => setPreflight(res.data))
            .catch(() => {}); // Silent — not critical
    }, []);

    // Cleanup polling on unmount
    useEffect(() => {
        return () => {
            Object.values(pollRefs.current).forEach(clearInterval);
        };
    }, []);

    // ── Preflight check before download ─────────────────────────
    const runPreflightForModel = useCallback(async (modelId) => {
        try {
            setPreflightLoading(true);
            const res = await modelsAPI.preflight(modelId);
            setPreflight(res.data);
            return res.data;
        } catch {
            return null;
        } finally {
            setPreflightLoading(false);
        }
    }, []);

    // ── Download management ─────────────────────────────────────
    const startDownload = useCallback(async (modelId) => {
        setError('');

        // Pre-flight check
        const check = await runPreflightForModel(modelId);
        if (check) {
            if (!check.hf_reachable) {
                setError(`Cannot reach HuggingFace Hub: ${check.hf_message || 'Check your internet connection.'}`);
                return;
            }
            if (check.enough_space === false) {
                setError(
                    `Not enough disk space. Need ~${(check.estimated_size_gb + (check.disk_headroom_gb || 2)).toFixed(1)} GB ` +
                    `but only ${check.disk_free_gb?.toFixed(1)} GB free. Delete unused caches to free space.`
                );
                return;
            }
        }

        setDownloads((prev) => ({
            ...prev,
            [modelId]: { status: 'starting', progress: 0, message: 'Starting download...' },
        }));
        try {
            const res = await modelsAPI.download(modelId);
            if (res.data?.status === 'cached') {
                setDownloads((prev) => {
                    const next = { ...prev };
                    delete next[modelId];
                    return next;
                });
                loadModels();
                return;
            }
            pollDownloadProgress(modelId);
        } catch (err) {
            const status = err.response?.status;
            let message = err.response?.data?.detail || 'Failed to start download';
            if (status === 507) {
                message = err.response?.data?.detail || 'Not enough disk space.';
            } else if (!err.response) {
                message = 'Cannot reach the server. Check your connection.';
            }
            setDownloads((prev) => ({
                ...prev,
                [modelId]: { status: 'error', progress: 0, message, error: message },
            }));
        }
    }, [loadModels, runPreflightForModel]);

    const pollDownloadProgress = useCallback((modelId) => {
        if (pollRefs.current[modelId]) clearInterval(pollRefs.current[modelId]);

        let consecutiveErrors = 0;
        const MAX_POLL_ERRORS = 10;

        const poll = async () => {
            try {
                const res = await modelsAPI.downloadProgress(modelId);
                const data = res.data;
                consecutiveErrors = 0; // Reset on success

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
                    loadModels();
                    setTimeout(() => {
                        setDownloads((prev) => {
                            const next = { ...prev };
                            delete next[modelId];
                            return next;
                        });
                    }, 5000);
                } else if (data.status === 'error' || data.status === 'cancelled' || data.status === 'interrupted') {
                    clearInterval(pollRefs.current[modelId]);
                    delete pollRefs.current[modelId];
                }
            } catch {
                consecutiveErrors++;
                if (consecutiveErrors >= MAX_POLL_ERRORS) {
                    clearInterval(pollRefs.current[modelId]);
                    delete pollRefs.current[modelId];
                    setDownloads((prev) => ({
                        ...prev,
                        [modelId]: {
                            status: 'error',
                            progress: 0,
                            message: 'Lost connection to server. Check if the download is still running.',
                            error: 'Connection lost',
                        },
                    }));
                }
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
        if (!window.confirm('Delete this cached model? It will need to be re-downloaded to use.')) return;
        try {
            await modelsAPI.deleteCache(modelId);
            setError('');
            loadModels();
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

    // ── Custom model lookup (exact ID) ──────────────────────────
    const lookupCustomModel = useCallback(async () => {
        const trimmed = customModelId.trim();
        if (!trimmed) return;
        if (!/^[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+$/.test(trimmed)) {
            setCustomError('Model ID must be in "org/name" format (e.g. meta-llama/Llama-3.2-3B)');
            return;
        }
        setCustomLooking(true);
        setCustomError('');
        setCustomModel(null);
        try {
            const res = await withRetry(() => modelsAPI.lookupCustom(trimmed), { retries: 1 });
            setCustomModel(res.data);
        } catch (err) {
            const status = err.response?.status;
            if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
                setCustomError('Request timed out. HuggingFace may be slow — please try again.');
            } else if (status === 429) {
                setCustomError('Too many lookups. Please wait a minute and try again.');
            } else if (status === 502 || status === 504) {
                setCustomError('Cannot reach HuggingFace Hub. Check your internet connection.');
            } else if (status === 403) {
                setCustomError(err.response?.data?.detail || 'This is a gated/private model. Set HF_TOKEN and accept the license.');
            } else if (status === 404) {
                setCustomError(err.response?.data?.detail || 'Model not found. Check the spelling.');
            } else if (!err.response) {
                setCustomError('Cannot reach the server. Check your connection.');
            } else {
                setCustomError(err.response?.data?.detail || 'Model not found on HuggingFace Hub.');
            }
        } finally {
            setCustomLooking(false);
        }
    }, [customModelId]);

    const debouncedLookup = useCallback(() => {
        if (debounceRef.current) clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(lookupCustomModel, 300);
    }, [lookupCustomModel]);

    // ── HuggingFace Hub search ──────────────────────────────────
    const doSearch = useCallback(async (query) => {
        const q = (query || '').trim();
        if (!q || q.length < 2) {
            setSearchResults([]);
            return;
        }
        setSearching(true);
        setSearchError('');
        try {
            const res = await withRetry(() => modelsAPI.searchHF(q), {
                retries: 1,
                delay: 2000,
            });
            setSearchResults(res.data?.results || []);
        } catch (err) {
            const status = err.response?.status;
            if (status === 429) {
                setSearchError('Too many searches. Wait a minute.');
            } else if (!err.response) {
                setSearchError('Cannot reach server.');
            } else {
                setSearchError(err.response?.data?.detail || 'Search failed.');
            }
        } finally {
            setSearching(false);
        }
    }, []);

    const onSearchInput = useCallback((value) => {
        setSearchQuery(value);
        if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
        if (value.trim().length >= 2) {
            searchDebounceRef.current = setTimeout(() => doSearch(value), 500);
        } else {
            setSearchResults([]);
        }
    }, [doSearch]);

    const selectSearchResult = useCallback(async (result) => {
        // Look up full model info
        setCustomModelId(result.model_id);
        setCustomLooking(true);
        setCustomError('');
        setShowSearch(false);
        setSearchQuery('');
        setSearchResults([]);
        try {
            const res = await withRetry(() => modelsAPI.lookupCustom(result.model_id), { retries: 1 });
            setCustomModel(res.data);
        } catch (err) {
            setCustomError(err.response?.data?.detail || 'Failed to load model details.');
        } finally {
            setCustomLooking(false);
        }
    }, []);

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
                    Select from the catalog, search HuggingFace, or enter any model ID.
                </p>
            </div>

            {/* Preflight Status Banner */}
            {preflight && !preflight.ready && (
                <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 space-y-2">
                    {!preflight.hf_reachable && (
                        <div className="flex items-center gap-2 text-sm text-amber-800">
                            <WifiOff className="w-4 h-4 shrink-0" />
                            <span><strong>Offline:</strong> {preflight.hf_message || 'Cannot reach HuggingFace Hub.'}</span>
                        </div>
                    )}
                    {!preflight.disk_ok && (
                        <div className="flex items-center gap-2 text-sm text-amber-800">
                            <HardDriveDownload className="w-4 h-4 shrink-0" />
                            <span><strong>Low disk:</strong> Only {preflight.disk_free_gb?.toFixed(1)} GB free. Models need more space.</span>
                        </div>
                    )}
                    {!preflight.has_hf_token && (
                        <div className="flex items-center gap-2 text-xs text-amber-700">
                            <Shield className="w-3.5 h-3.5 shrink-0" />
                            <span>No HF_TOKEN set — some gated models (Llama, Mistral) won't be accessible.</span>
                        </div>
                    )}
                    <button
                        onClick={() => modelsAPI.preflight().then(r => setPreflight(r.data)).catch(() => {})}
                        className="text-xs text-amber-700 hover:text-amber-900 font-medium flex items-center gap-1"
                    >
                        <RefreshCw className="w-3 h-3" /> Recheck
                    </button>
                </div>
            )}

            {error && (
                <div className="bg-danger-400/10 border border-danger-400/30 text-danger-600 text-sm rounded-xl p-4 flex items-center gap-2">
                    <XCircle className="w-4 h-4 shrink-0" /> {error}
                    <button onClick={() => setError('')} className="ml-auto text-danger-400 hover:text-danger-600"><X className="w-4 h-4" /></button>
                </div>
            )}

            {/* HuggingFace Hub Search */}
            <div className="bg-gradient-to-r from-primary-50 to-primary-100/50 rounded-2xl border border-primary-200 p-5">
                <div className="flex items-center justify-between mb-1">
                    <h3 className="text-sm font-semibold text-primary-800 flex items-center gap-2">
                        🤗 HuggingFace Models
                    </h3>
                    <button
                        onClick={() => setShowSearch(s => !s)}
                        className="text-xs text-primary-600 hover:text-primary-800 font-medium flex items-center gap-1"
                    >
                        <Search className="w-3 h-3" /> {showSearch ? 'Hide Search' : 'Search Hub'}
                        <ChevronDown className={`w-3 h-3 transition-transform ${showSearch ? 'rotate-180' : ''}`} />
                    </button>
                </div>

                {/* Search Box */}
                {showSearch && (
                    <div className="mb-4">
                        <div className="flex gap-2 mt-2">
                            <div className="relative flex-1">
                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-surface-400 pointer-events-none" />
                                <input
                                    type="text"
                                    value={searchQuery}
                                    onChange={(e) => onSearchInput(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && doSearch(searchQuery)}
                                    placeholder="Search models (e.g. 'llama 3B', 'phi code')..."
                                    className="w-full pl-9 pr-4 py-2.5 bg-white border border-primary-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30 placeholder:text-surface-400"
                                />
                                {searching && (
                                    <Loader className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-primary-500 animate-spin" />
                                )}
                            </div>
                        </div>
                        {searchError && (
                            <p className="text-xs text-danger-600 mt-2 flex items-center gap-1">
                                <XCircle className="w-3 h-3" /> {searchError}
                            </p>
                        )}
                        {searchResults.length > 0 && (
                            <div className="mt-3 max-h-60 overflow-y-auto space-y-1.5 scrollbar-thin">
                                {searchResults.map((r) => (
                                    <button
                                        key={r.model_id}
                                        onClick={() => selectSearchResult(r)}
                                        className="w-full text-left px-3 py-2.5 bg-white rounded-lg border border-surface-100 hover:border-primary-300 hover:bg-primary-50 transition-all group"
                                    >
                                        <div className="flex items-center justify-between">
                                            <div className="min-w-0 flex-1">
                                                <p className="text-sm font-medium text-surface-900 truncate group-hover:text-primary-800">{r.model_id}</p>
                                                <div className="flex items-center gap-2 mt-0.5">
                                                    <span className="text-xs text-surface-500">{r.parameters || '?'} params</span>
                                                    {r.size_gb > 0 && <span className="text-xs text-surface-400">• {r.size_gb} GB</span>}
                                                    {r.downloads > 0 && (
                                                        <span className="text-xs text-surface-400">• {r.downloads >= 1e6 ? `${(r.downloads/1e6).toFixed(1)}M` : r.downloads >= 1000 ? `${(r.downloads/1000).toFixed(0)}K` : r.downloads} downloads</span>
                                                    )}
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-2 shrink-0 ml-2">
                                                {r.is_cached && (
                                                    <span className="text-xs text-success-600 font-medium flex items-center gap-0.5">
                                                        <Check className="w-3 h-3" /> Cached
                                                    </span>
                                                )}
                                                <ExternalLink className="w-3.5 h-3.5 text-surface-300 group-hover:text-primary-500" />
                                            </div>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        )}
                        {!searching && searchQuery.length >= 2 && searchResults.length === 0 && !searchError && (
                            <p className="text-xs text-surface-500 mt-2 text-center py-3">No models found for "{searchQuery}"</p>
                        )}
                    </div>
                )}

                {/* Direct Model ID Input */}
                <p className="text-xs text-primary-600 mb-3">
                    {showSearch ? 'Or enter a model ID directly:' : 'Enter a model ID from HuggingFace Hub (e.g.'} <code className="bg-primary-200/50 px-1 rounded">meta-llama/Llama-3.2-3B</code>{showSearch ? '' : ')'}
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
                                {preflight?.disk_free_gb != null && (
                                    <span className="ml-2 text-surface-500">• {preflight.disk_free_gb.toFixed(1)} GB free</span>
                                )}
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
