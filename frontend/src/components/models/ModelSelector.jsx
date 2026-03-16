import { useState, useEffect, useRef, useCallback } from 'react';
import { modelsAPI } from '../../services/api';
import { Check, Download, AlertTriangle, XCircle, Loader, Info, Search, ExternalLink, X, Trash2, HardDrive, Wifi, WifiOff, HardDriveDownload, RefreshCw, Shield, ChevronDown, FolderOpen, MapPin } from 'lucide-react';

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

    // ── Delete confirmation dialog ─────────────────────────────
    // deleteConfirm: null | { modelId, modelName }
    const [deleteConfirm, setDeleteConfirm] = useState(null);

    // ── Download location picker dialog ─────────────────────────
    // downloadDialog: null | { modelId, modelName, sizeGb }
    const [downloadDialog, setDownloadDialog] = useState(null);
    // Available directories (from /models/download-dirs)
    const [downloadDirs, setDownloadDirs] = useState([]);
    // Which dir option is selected: 'default' | 'hf_cache' | 'custom'
    const [dlDirMode, setDlDirMode] = useState('default');
    const [customDlPath, setCustomDlPath] = useState('');



    // Run preflight check on mount (non-blocking)
    useEffect(() => {
        modelsAPI.preflight()
            .then(res => setPreflight(res.data))
            .catch(() => { }); // Silent — not critical
    }, []);

    // Load suggested download directories
    useEffect(() => {
        modelsAPI.downloadDirs()
            .then(res => {
                const dirs = res.data?.dirs || [];
                setDownloadDirs(dirs);
            })
            .catch(() => { }); // Silent — non-critical
    }, []);

    // Cleanup polling on unmount
    useEffect(() => {
        return () => {
            Object.values(pollRefs.current).forEach(clearInterval);
        };
    }, []);

    // ── Resume any in-flight downloads on mount ─────────────────
    // ModelSelector is conditionally rendered (tab-based), so it can remount
    // mid-download. On each mount we probe every known model and any active
    // custom model to see if a download is already running on the backend.
    const pollDownloadProgressRef = useRef(null);
    useEffect(() => {
        // pollDownloadProgress is defined later; we use a ref so this effect
        // can call it without a stale-closure dependency.
        pollDownloadProgressRef.current = (modelId) => {
            if (pollRefs.current[modelId]) clearInterval(pollRefs.current[modelId]);
            let consecutiveErrors = 0;
            const MAX_POLL_ERRORS = 10;
            const poll = async () => {
                try {
                    const res = await modelsAPI.downloadProgress(modelId);
                    const data = res.data;
                    consecutiveErrors = 0;
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
                    // 'cached' is only terminal when progress == 100. This guards against
                    // the race where model exists in HF cache but download is still running.
                    const isTerminalCached = data.status === 'cached' && (data.progress || 0) >= 100;
                    if (data.status === 'completed' || isTerminalCached) {
                        clearInterval(pollRefs.current[modelId]);
                        delete pollRefs.current[modelId];
                        loadModels();
                        setTimeout(() => setDownloads((prev) => { const n = { ...prev }; delete n[modelId]; return n; }), 5000);
                    } else if (data.status === 'error' || data.status === 'cancelled' || data.status === 'interrupted') {
                        clearInterval(pollRefs.current[modelId]);
                        delete pollRefs.current[modelId];
                    }
                    // 'downloading', 'running', 'queued', 'cached' (progress<100) → keep polling
                } catch {
                    consecutiveErrors++;
                    if (consecutiveErrors >= MAX_POLL_ERRORS) {
                        clearInterval(pollRefs.current[modelId]);
                        delete pollRefs.current[modelId];
                    }
                }
            };
            poll();
            pollRefs.current[modelId] = setInterval(poll, 2000);
        };
    });

    // Load models with retry; after loading, probe for any in-flight downloads
    // so that the progress bar reappears after a tab switch (component remount).
    const loadModels = useCallback(() => {
        withRetry(() => modelsAPI.list(), { retries: 2, delay: 1500 })
            .then((res) => {
                const loaded = res.data || [];
                setModels(loaded);
                setError('');
                // Resume polling for any model that is currently downloading
                loaded.forEach((m) => {
                    if (!pollRefs.current[m.model_id]) {
                        modelsAPI.downloadProgress(m.model_id)
                            .then((pr) => {
                                const d = pr.data;
                                if (d.status === 'running' || d.status === 'queued' || d.status === 'downloading') {
                                    setDownloads((prev) => ({
                                        ...prev,
                                        [m.model_id]: {
                                            status: d.status,
                                            progress: d.progress || 0,
                                            message: d.message || '',
                                            error: d.error,
                                            elapsed_seconds: d.elapsed_seconds,
                                        },
                                    }));
                                    // Start live polling
                                    if (pollDownloadProgressRef.current) {
                                        pollDownloadProgressRef.current(m.model_id);
                                    }
                                }
                            })
                            .catch(() => { }); // silent — model has no download task
                    }
                });
            })
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

    // Trigger load on mount
    useEffect(() => {
        loadModels();
    }, [loadModels]);

    const runPreflightForModel = useCallback(async (modelId) => {
        try {
            const res = await modelsAPI.preflight(modelId);
            setPreflight(res.data);
            return res.data;
        } catch (err) {
            console.debug('Preflight check failed:', err.message || err);
            return null;
        }
    }, []);

    // ── Download management ─────────────────────────────────────
    const startDownload = useCallback(async (modelId, downloadPath = null) => {
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
            const res = await modelsAPI.download(modelId, downloadPath);
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

    // Open the "choose save location" dialog before starting a download
    const openDownloadDialog = useCallback((model) => {
        // Reset dialog state
        setDlDirMode('default');
        setCustomDlPath('');
        setDownloadDialog({
            modelId: model.model_id,
            modelName: model.name,
            sizeGb: model.size_gb,
        });
    }, []);

    // Called when user confirms the dialog
    const confirmDownloadDialog = useCallback(() => {
        if (!downloadDialog) return;
        let chosenPath = null;
        if (dlDirMode === 'custom') {
            chosenPath = customDlPath.trim() || null;
        } else if (dlDirMode === 'hf_cache') {
            const hfDir = downloadDirs.find(d => d.id === 'hf_cache');
            chosenPath = hfDir?.path || null;
        }
        // null → server uses default MODELS_DIR
        setDownloadDialog(null);
        startDownload(downloadDialog.modelId, chosenPath);
    }, [downloadDialog, dlDirMode, customDlPath, downloadDirs, startDownload]);

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

                // 'cached' is only terminal when progress == 100. This prevents stopping
                // polling prematurely if the model is in HF cache but still downloading.
                const isTerminalCached = data.status === 'cached' && (data.progress || 0) >= 100;
                if (data.status === 'completed' || isTerminalCached) {
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
                // 'downloading', 'running', 'queued', 'starting' → keep polling
            } catch (pollErr) {
                console.debug('Download poll error:', pollErr.message || pollErr);
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

    const deleteCache = useCallback((modelId, modelName = '') => {
        setDeleteConfirm({ modelId, modelName });
    }, []);

    const confirmDelete = useCallback(async () => {
        if (!deleteConfirm) return;
        const { modelId } = deleteConfirm;
        setDeleteConfirm(null);
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
    }, [deleteConfirm, loadModels, selectedModel, onSelectModel]);

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

    // Parse speed, ETA, and bytes from the backend progress message string.
    // Message format: "Downloading Name: 240/1024 MB (23%) • 5.2 MB/s • ETA 2m"
    const parseDownloadMessage = (message = '') => {
        const result = { speed: null, eta: null, doneMb: null, totalMb: null, pct: null };
        const bytesMatch = message.match(/(\d+)\/(\d+)\s*MB/);
        if (bytesMatch) { result.doneMb = parseInt(bytesMatch[1]); result.totalMb = parseInt(bytesMatch[2]); }
        const pctMatch = message.match(/\((\d+)%\)/);
        if (pctMatch) result.pct = parseInt(pctMatch[1]);
        const speedMatch = message.match(/([\d.]+)\s*MB\/s/);
        if (speedMatch) result.speed = parseFloat(speedMatch[1]);
        const etaMatch = message.match(/ETA\s+([\d]+m(?:\s*[\d]+s)?|[\d]+s)/);
        if (etaMatch) result.eta = etaMatch[1];
        return result;
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
                        onClick={() => modelsAPI.preflight().then(r => setPreflight(r.data)).catch(() => { })}
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
                                                        <span className="text-xs text-surface-400">• {r.downloads >= 1e6 ? `${(r.downloads / 1e6).toFixed(1)}M` : r.downloads >= 1000 ? `${(r.downloads / 1000).toFixed(0)}K` : r.downloads} downloads</span>
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
                            const isDownloading = dl.status === 'downloading' || dl.status === 'starting' || dl.status === 'running' || dl.status === 'queued';
                            const downloadDone = dl.status === 'completed' || dl.status === 'cached';
                            const downloadErr = dl.status === 'error';
                            const { speed, eta, doneMb, totalMb, pct } = parseDownloadMessage(dl.message);
                            const displayPct = pct ?? dl.progress ?? 0;
                            return (
                                <div className="mt-2 rounded-xl border border-surface-200 bg-surface-50 overflow-hidden" onClick={(e) => e.stopPropagation()}>
                                    {isDownloading && (
                                        <>
                                            {/* Progress bar */}
                                            <div className="relative h-2 bg-surface-200">
                                                <div
                                                    className="absolute inset-y-0 left-0 bg-gradient-to-r from-primary-400 via-primary-500 to-emerald-500 transition-all duration-700 ease-out"
                                                    style={{ width: `${Math.max(displayPct, 2)}%` }}
                                                >
                                                    <div className="absolute inset-0 bg-white/20 animate-pulse" />
                                                </div>
                                            </div>
                                            {/* Stats row */}
                                            <div className="px-3 py-2 space-y-1.5">
                                                <div className="flex items-center justify-between">
                                                    <span className="text-xs font-semibold text-primary-700 flex items-center gap-1.5">
                                                        <Loader className="w-3 h-3 animate-spin" />
                                                        {displayPct > 0 ? `${Math.round(displayPct)}%` : 'Starting...'}
                                                    </span>
                                                    <button onClick={() => cancelDownload(customModel.model_id)} className="text-surface-400 hover:text-danger-500 transition-colors" title="Cancel download">
                                                        <X className="w-3.5 h-3.5" />
                                                    </button>
                                                </div>
                                                <div className="flex items-center gap-3 flex-wrap">
                                                    {doneMb != null && totalMb != null && (
                                                        <span className="text-[10px] text-surface-500 font-mono">{doneMb} / {totalMb} MB</span>
                                                    )}
                                                    {speed != null && (
                                                        <span className="text-[10px] font-semibold text-emerald-600 bg-emerald-50 border border-emerald-200 px-1.5 py-0.5 rounded-full">{speed.toFixed(1)} MB/s</span>
                                                    )}
                                                    {eta && (
                                                        <span className="text-[10px] text-amber-700 bg-amber-50 border border-amber-200 px-1.5 py-0.5 rounded-full">ETA {eta}</span>
                                                    )}
                                                </div>
                                            </div>
                                        </>
                                    )}
                                    {downloadDone && (
                                        <div className="px-3 py-2 flex items-center gap-1.5 text-success-600">
                                            <Check className="w-3.5 h-3.5" />
                                            <span className="text-xs font-medium">Download complete!</span>
                                        </div>
                                    )}
                                    {downloadErr && (
                                        <div className="px-3 py-2 flex items-start gap-1.5 text-danger-600">
                                            <XCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                                            <span className="text-xs">{dl.message || 'Download failed'}</span>
                                        </div>
                                    )}
                                </div>
                            );
                        })()}

                        {/* Action buttons for custom model */}
                        <div className="flex gap-2 mt-3" onClick={(e) => e.stopPropagation()}>
                            {!customModel.is_cached && !downloads[customModel.model_id]?.status?.match(/downloading|starting|running/) && (
                                <button
                                    onClick={() => openDownloadDialog(customModel)}
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
                    const isDownloading = dl && (dl.status === 'downloading' || dl.status === 'starting' || dl.status === 'running' || dl.status === 'queued');
                    const downloadDone = dl && (dl.status === 'completed' || (dl.status === 'cached' && (dl.progress || 0) >= 100));
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
                                            onClick={(e) => { e.stopPropagation(); deleteCache(model.model_id, model.name); }}
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
                                <div className="mt-3 rounded-xl border border-surface-200 bg-surface-50 overflow-hidden" onClick={(e) => e.stopPropagation()}>
                                    {isDownloading && (() => {
                                        const { speed, eta, doneMb, totalMb, pct } = parseDownloadMessage(dl.message);
                                        const displayPct = pct ?? dl.progress ?? 0;
                                        return (
                                            <>
                                                {/* Animated gradient progress bar */}
                                                <div className="relative h-3 bg-surface-200">
                                                    <div
                                                        className="absolute inset-y-0 left-0 bg-gradient-to-r from-primary-400 via-primary-500 to-emerald-500 transition-all duration-700 ease-out"
                                                        style={{ width: `${Math.max(displayPct, 2)}%` }}
                                                    >
                                                        {/* Shimmer overlay */}
                                                        <div className="absolute inset-0 overflow-hidden">
                                                            <div className="absolute inset-y-0 -left-full w-1/2 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-slide-shimmer" />
                                                        </div>
                                                    </div>
                                                    {/* Percentage label on bar */}
                                                    {displayPct >= 8 && (
                                                        <span className="absolute inset-y-0 left-2 flex items-center text-[9px] font-bold text-white drop-shadow">
                                                            {Math.round(displayPct)}%
                                                        </span>
                                                    )}
                                                </div>

                                                {/* Stats row */}
                                                <div className="px-3 py-2.5">
                                                    <div className="flex items-center justify-between mb-2">
                                                        <span className="text-xs font-semibold text-primary-700 flex items-center gap-1.5">
                                                            <Loader className="w-3.5 h-3.5 animate-spin" />
                                                            {displayPct > 0 ? `Downloading — ${Math.round(displayPct)}%` : 'Starting download…'}
                                                        </span>
                                                        <div className="flex items-center gap-2">
                                                            {dl.elapsed_seconds > 0 && (
                                                                <span className="text-[10px] text-surface-400">{formatElapsed(dl.elapsed_seconds)}</span>
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

                                                    {/* Metric pills */}
                                                    <div className="flex items-center gap-2 flex-wrap">
                                                        {doneMb != null && totalMb != null && (
                                                            <span className="inline-flex items-center gap-1 text-[10px] text-surface-600 bg-white border border-surface-200 px-2 py-1 rounded-full font-mono">
                                                                <HardDrive className="w-3 h-3 text-surface-400" />
                                                                {doneMb} / {totalMb} MB
                                                            </span>
                                                        )}
                                                        {speed != null && (
                                                            <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-1 rounded-full">
                                                                <Wifi className="w-3 h-3" />
                                                                {speed.toFixed(1)} MB/s
                                                            </span>
                                                        )}
                                                        {eta && (
                                                            <span className="inline-flex items-center gap-1 text-[10px] font-medium text-amber-700 bg-amber-50 border border-amber-200 px-2 py-1 rounded-full">
                                                                ⏱ ETA {eta}
                                                            </span>
                                                        )}
                                                        {!speed && !eta && dl.message && (
                                                            <span className="text-[10px] text-surface-500">{dl.message}</span>
                                                        )}
                                                    </div>
                                                </div>
                                            </>
                                        );
                                    })()}
                                    {downloadDone && (
                                        <div className="px-3 py-2.5 flex items-center gap-2 text-success-600 bg-success-400/5">
                                            <Check className="w-4 h-4" />
                                            <span className="text-xs font-medium">Download complete!</span>
                                        </div>
                                    )}
                                    {downloadError && (
                                        <div className="px-3 py-2.5">
                                            <div className="flex items-start gap-1.5 text-danger-600 mb-1.5">
                                                <XCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                                                <span className="text-xs leading-snug">{dl.message || 'Download failed'}</span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <button
                                                    onClick={() => { dismissDownload(model.model_id); startDownload(model.model_id); }}
                                                    className="text-xs text-primary-600 hover:text-primary-800 font-medium transition-colors"
                                                >
                                                    Retry
                                                </button>
                                                <button
                                                    onClick={() => dismissDownload(model.model_id)}
                                                    className="text-[10px] text-surface-400 hover:text-surface-600 transition-colors"
                                                >
                                                    Dismiss
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                    {downloadCancelled && (
                                        <div className="px-3 py-2.5 flex items-center gap-2 text-surface-500">
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
                                        onClick={(e) => { e.stopPropagation(); openDownloadDialog(model); }}
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
                                onClick={() => openDownloadDialog(selectedModel)}
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

            {/* ── Delete Confirmation Dialog ─────────────────────────── */}
            {deleteConfirm && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
                    onClick={() => setDeleteConfirm(null)}>
                    <div className="bg-white rounded-2xl shadow-2xl w-full max-w-sm p-6 space-y-4"
                        onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-start gap-3">
                            <div className="p-2 rounded-full bg-danger-100 shrink-0">
                                <Trash2 className="w-5 h-5 text-danger-600" />
                            </div>
                            <div>
                                <h3 className="text-base font-bold text-surface-900">Delete cached model?</h3>
                                {deleteConfirm.modelName && (
                                    <p className="text-sm text-surface-600 mt-0.5">
                                        <span className="font-semibold">{deleteConfirm.modelName}</span> will be removed from disk.
                                    </p>
                                )}
                                <p className="text-xs text-surface-400 mt-1">You can re-download it later.</p>
                            </div>
                        </div>
                        <div className="flex gap-3">
                            <button
                                onClick={() => setDeleteConfirm(null)}
                                className="flex-1 py-2.5 rounded-xl text-sm font-medium bg-surface-100 text-surface-700 hover:bg-surface-200 transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={confirmDelete}
                                className="flex-1 py-2.5 rounded-xl text-sm font-medium bg-red-500 text-white hover:bg-red-600 transition-colors flex items-center justify-center gap-2"
                            >
                                <Trash2 className="w-4 h-4" /> Delete
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* ── Download Location Dialog ─────────────────────────────── */}
            {downloadDialog && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
                    onClick={() => setDownloadDialog(null)}>
                    <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-5"
                        onClick={(e) => e.stopPropagation()}>
                        {/* Header */}
                        <div className="flex items-start justify-between">
                            <div>
                                <h3 className="text-base font-bold text-surface-900 flex items-center gap-2">
                                    <FolderOpen className="w-5 h-5 text-primary-500" />
                                    Choose save location
                                </h3>
                                <p className="text-xs text-surface-500 mt-0.5">
                                    Downloading <span className="font-semibold">{downloadDialog.modelName}</span>
                                    {downloadDialog.sizeGb > 0 && <span className="text-surface-400"> ({downloadDialog.sizeGb} GB)</span>}
                                </p>
                            </div>
                            <button onClick={() => setDownloadDialog(null)}
                                className="text-surface-400 hover:text-surface-600 transition-colors mt-0.5">
                                <X className="w-4 h-4" />
                            </button>
                        </div>

                        {/* Location options */}
                        <div className="space-y-2">
                            {/* Default MODELS_DIR */}
                            <label className={`flex items-start gap-3 p-3 rounded-xl border-2 cursor-pointer transition-all
                                ${dlDirMode === 'default' ? 'border-primary-500 bg-primary-50' : 'border-surface-200 hover:border-primary-200'}`}>
                                <input type="radio" className="mt-0.5 accent-primary-500"
                                    checked={dlDirMode === 'default'}
                                    onChange={() => setDlDirMode('default')} />
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm font-medium text-surface-900">Default models folder</span>
                                        {downloadDirs.find(d => d.id === 'models_dir') && (
                                            <span className="text-xs text-surface-400 ml-2 shrink-0">
                                                {downloadDirs.find(d => d.id === 'models_dir').disk_free_gb?.toFixed(1)} GB free
                                            </span>
                                        )}
                                    </div>
                                    <p className="text-xs text-surface-400 mt-0.5 truncate font-mono">
                                        {downloadDirs.find(d => d.id === 'models_dir')?.path || './models/'}
                                    </p>
                                    <span className="text-[10px] font-medium text-primary-600 bg-primary-100 px-1.5 py-0.5 rounded-full mt-1 inline-block">Recommended</span>
                                </div>
                            </label>

                            {/* HF cache */}
                            <label className={`flex items-start gap-3 p-3 rounded-xl border-2 cursor-pointer transition-all
                                ${dlDirMode === 'hf_cache' ? 'border-primary-500 bg-primary-50' : 'border-surface-200 hover:border-primary-200'}`}>
                                <input type="radio" className="mt-0.5 accent-primary-500"
                                    checked={dlDirMode === 'hf_cache'}
                                    onChange={() => setDlDirMode('hf_cache')} />
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm font-medium text-surface-900">HuggingFace cache</span>
                                        {downloadDirs.find(d => d.id === 'hf_cache') && (
                                            <span className="text-xs text-surface-400 ml-2 shrink-0">
                                                {downloadDirs.find(d => d.id === 'hf_cache').disk_free_gb?.toFixed(1)} GB free
                                            </span>
                                        )}
                                    </div>
                                    <p className="text-xs text-surface-400 mt-0.5 truncate font-mono">
                                        {downloadDirs.find(d => d.id === 'hf_cache')?.path || '~/.cache/huggingface/hub'}
                                    </p>
                                </div>
                            </label>

                            {/* Custom path */}
                            <label className={`flex items-start gap-3 p-3 rounded-xl border-2 cursor-pointer transition-all
                                ${dlDirMode === 'custom' ? 'border-primary-500 bg-primary-50' : 'border-surface-200 hover:border-primary-200'}`}>
                                <input type="radio" className="mt-0.5 accent-primary-500"
                                    checked={dlDirMode === 'custom'}
                                    onChange={() => setDlDirMode('custom')} />
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-1.5 mb-1.5">
                                        <MapPin className="w-3.5 h-3.5 text-surface-500" />
                                        <span className="text-sm font-medium text-surface-900">Custom path</span>
                                    </div>
                                    <input
                                        type="text"
                                        value={customDlPath}
                                        onChange={(e) => { setCustomDlPath(e.target.value); setDlDirMode('custom'); }}
                                        onClick={() => setDlDirMode('custom')}
                                        placeholder="/absolute/path/to/directory"
                                        className="w-full px-3 py-2 text-xs font-mono bg-white border border-surface-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500/30 placeholder:text-surface-300"
                                    />
                                    <p className="text-[10px] text-surface-400 mt-1">Must be an absolute path. The directory will be created if it doesn't exist.</p>
                                </div>
                            </label>
                        </div>

                        {/* Validation hint for custom path */}
                        {dlDirMode === 'custom' && customDlPath.trim() && !customDlPath.trim().startsWith('/') && (
                            <div className="flex items-center gap-2 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                                <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                                Path must be absolute (start with /)
                            </div>
                        )}

                        {/* Actions */}
                        <div className="flex gap-3 pt-1">
                            <button
                                onClick={() => setDownloadDialog(null)}
                                className="flex-1 py-2.5 rounded-xl text-sm font-medium bg-surface-100 text-surface-700 hover:bg-surface-200 transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={confirmDownloadDialog}
                                disabled={dlDirMode === 'custom' && (!customDlPath.trim() || !customDlPath.trim().startsWith('/'))}
                                className="flex-1 py-2.5 rounded-xl text-sm font-medium bg-emerald-500 text-white hover:bg-emerald-600 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                            >
                                <Download className="w-4 h-4" />
                                Download {downloadDialog.sizeGb > 0 ? `(${downloadDialog.sizeGb} GB)` : ''}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
