import axios from 'axios';

const api = axios.create({
    baseURL: '/api',
    headers: { 'Content-Type': 'application/json' },
    timeout: 30000,
});

// Attach JWT token to every request
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('meowllm_token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// Handle 401 responses — attempt token refresh before giving up
let _isRefreshing = false;
let _refreshQueue = [];
let _lastNetworkEvent = 0;

function _processQueue(error, token = null) {
    _refreshQueue.forEach(({ resolve, reject }) => {
        if (error) reject(error);
        else resolve(token);
    });
    _refreshQueue = [];
}

api.interceptors.response.use(
    (response) => response,
    async (error) => {
        const originalRequest = error.config;
        const url = originalRequest?.url || '';
        const isAuthEndpoint = url.startsWith('/auth/');

        // Don't attempt refresh for auth endpoints or already retried requests
        if (error.response?.status === 401 && !isAuthEndpoint && !originalRequest._retry) {
            if (_isRefreshing) {
                // Queue this request until the refresh completes
                return new Promise((resolve, reject) => {
                    _refreshQueue.push({ resolve, reject });
                }).then((token) => {
                    originalRequest.headers.Authorization = `Bearer ${token}`;
                    return api(originalRequest);
                });
            }

            originalRequest._retry = true;
            _isRefreshing = true;

            try {
                const res = await api.post('/auth/refresh');
                const newToken = res.data.token;
                localStorage.setItem('meowllm_token', newToken);
                api.defaults.headers.Authorization = `Bearer ${newToken}`;
                _processQueue(null, newToken);
                originalRequest.headers.Authorization = `Bearer ${newToken}`;
                return api(originalRequest);
            } catch (refreshError) {
                _processQueue(refreshError, null);
                localStorage.removeItem('meowllm_token');
                // Dispatch a custom event so AuthContext can clear user state
                window.dispatchEvent(new Event('meowllm:force-logout'));
                return Promise.reject(refreshError);
            } finally {
                _isRefreshing = false;
            }
        }

        // ── Timeout error detection ────────────────────────
        if (error.code === 'ECONNABORTED') {
            const now = Date.now();
            if (!_lastNetworkEvent || now - _lastNetworkEvent > 10_000) {
                _lastNetworkEvent = now;
                window.dispatchEvent(new CustomEvent('meowllm:network-error', {
                    detail: {
                        type: 'timeout',
                        status: 0,
                        message: 'Request timed out. The server may be busy — please try again.',
                    },
                }));
            }
            return Promise.reject(error);
        }

        // ── Network / server error detection ────────────────────────
        // Fires a custom event when the backend is unreachable or returns 502/503/504.
        // Throttled so the event fires at most once every 10 seconds.
        const status = error.response?.status;
        const isNetworkError = !error.response; // no response = offline / CORS / DNS
        const isServerDown = status === 502 || status === 503 || status === 504;
        if (isNetworkError || isServerDown) {
            const now = Date.now();
            if (!_lastNetworkEvent || now - _lastNetworkEvent > 10_000) {
                _lastNetworkEvent = now;
                window.dispatchEvent(new CustomEvent('meowllm:network-error', {
                    detail: {
                        type: isNetworkError ? 'offline' : 'server-error',
                        status: status || 0,
                        message: isNetworkError
                            ? 'Cannot reach the server. Check your connection.'
                            : `Server error (${status}). The backend may be restarting.`,
                    },
                }));
            }
        }

        return Promise.reject(error);
    }
);

// ===== Auth =====
export const authAPI = {
    register: (data) => api.post('/auth/register', data),
    login: (data) => api.post('/auth/login', data),
    guest: () => api.post('/auth/guest'),
    me: () => api.get('/auth/me'),
    updateProfile: (data) => api.patch('/auth/profile', data),
    changePassword: (data) => api.post('/auth/password', data),
    refresh: () => api.post('/auth/refresh'),
    verifyEmail: (token) => api.post('/auth/verify-email', { token }),
    resendVerification: (email) => api.post('/auth/resend-verification', { email }),
    forgotPassword: (email) => api.post('/auth/forgot-password', { email }),
    resetPassword: (token, new_password) => api.post('/auth/reset-password', { token, new_password }),
    deleteAccount: () => api.delete('/auth/account'),
    logoutAll: () => api.post('/auth/logout-all'),
};

// ===== Projects =====
export const projectsAPI = {
    list: (params = {}) => api.get('/projects/', { params }),
    create: (data) => api.post('/projects/', data),
    get: (id) => api.get(`/projects/${id}`),
    update: (id, data) => api.put(`/projects/${id}`, data),
    delete: (id) => api.delete(`/projects/${id}`),
};

// ===== Datasets =====
export const datasetsAPI = {
    upload: (projectId, formData) =>
        api.post(`/projects/${projectId}/datasets/upload`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        }),
    list: (projectId) => api.get(`/projects/${projectId}/datasets/`),
    preview: (projectId, datasetId) =>
        api.get(`/projects/${projectId}/datasets/${datasetId}/preview`),
    delete: (projectId, datasetId) =>
        api.delete(`/projects/${projectId}/datasets/${datasetId}`),
    previewTraining: (projectId, config = {}) =>
        api.post(`/projects/${projectId}/datasets/preview-training`, null, {
            params: { base_model: config.base_model, max_tokens: config.max_tokens },
        }),
    augmentPreview: (projectId, options = {}) =>
        api.post(`/projects/${projectId}/datasets/augment`, { ...options, preview_only: true }),
    augment: (projectId, options = {}) =>
        api.post(`/projects/${projectId}/datasets/augment`, { ...options, preview_only: false }),
};

// ===== Models =====
export const modelsAPI = {
    list: () => api.get('/models/'),
    status: (modelId) => api.get(`/models/${encodeURIComponent(modelId)}/status`),
    // download_path: optional absolute path — if omitted, server uses default models/ folder
    download: (modelId, downloadPath = null) => api.post(
        `/models/${encodeURIComponent(modelId)}/download`,
        downloadPath ? { download_path: downloadPath } : {},
    ),
    downloadDirs: () => api.get('/models/download-dirs'),
    downloadProgress: (modelId) => api.get(`/models/${encodeURIComponent(modelId)}/download/progress`),
    cancelDownload: (modelId) => api.delete(`/models/${encodeURIComponent(modelId)}/download`),
    deleteCache: (modelId) => api.delete(`/models/${encodeURIComponent(modelId)}/cache`),
    exportModel: (projectId) => api.get(`/models/export/${projectId}`, { responseType: 'blob', timeout: 300000 }),
    // Custom model lookup (single model by exact ID)
    lookupCustom: (modelId) => api.post('/models/custom/lookup', null, {
        params: { model_id: modelId },
        timeout: 20000,  // 20s for HF API
    }),
    // Search HuggingFace Hub for models
    searchHF: (query, { limit = 10, sort = 'downloads' } = {}) =>
        api.get('/models/search', {
            params: { q: query, limit, sort },
            timeout: 20000,
        }),
    // Pre-flight check: connectivity, disk space, HF token
    preflight: (modelId = '') => api.get('/models/preflight', {
        params: modelId ? { model_id: modelId } : {},
        timeout: 15000,
    }),
    // Deep-verify cache integrity
    verifyCache: (modelId) => api.post(`/models/${encodeURIComponent(modelId)}/verify-cache`, null, {
        timeout: 30000,
    }),
    // GGUF export for LM Studio
    exportGGUF: (projectId, quantization = 'Q8_0') =>
        api.post(`/models/export/${projectId}/gguf`, null, { params: { quantization } }),
    ggufStatus: (projectId) => api.get(`/models/export/${projectId}/gguf/status`),
    ggufDownload: (projectId) =>
        api.get(`/models/export/${projectId}/gguf/download`, { responseType: 'blob', timeout: 600000 }),
};

// ===== Training =====
export const trainingAPI = {
    configure: (projectId, config) =>
        api.post(`/projects/${projectId}/train/configure`, config),
    start: (projectId) => api.post(`/projects/${projectId}/train/start`),
    pause: (projectId) => api.post(`/projects/${projectId}/train/pause`),
    resume: (projectId) => api.post(`/projects/${projectId}/train/resume`),
    stop: (projectId) => api.post(`/projects/${projectId}/train/stop`),
    status: (projectId) => api.get(`/projects/${projectId}/train/status`),
    history: (projectId, params = {}) =>
        api.get(`/projects/${projectId}/train/history`, { params }),
    compareRuns: (projectId, runIds) =>
        api.get(`/projects/${projectId}/train/compare`, { params: { run_ids: runIds.join(',') } }),
};

// ===== Inference =====
export const inferenceAPI = {
    chat: (projectId, data) => api.post(`/projects/${projectId}/chat`, data),
    /**
     * SSE streaming chat — returns an EventSource-like interface.
     * Calls the /chat/stream endpoint and yields partial tokens via callbacks.
     *
     * @param {number|string} projectId
     * @param {object} data - { prompt, system_prompt, temperature, max_tokens, ... }
     * @param {object} callbacks - { onToken(text), onDone(), onError(err) }
     * @returns {{ abort: () => void }} - controller to cancel the stream
     */
    chatStream: (projectId, data, { onToken, onDone, onError }) => {
        const abortController = new AbortController();
        const token = localStorage.getItem('meowllm_token');

        fetch(`/api/projects/${projectId}/chat/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(token ? { Authorization: `Bearer ${token}` } : {}),
            },
            body: JSON.stringify(data),
            signal: abortController.signal,
        })
            .then(async (response) => {
                if (!response.ok) {
                    const errBody = await response.json().catch(() => ({}));
                    throw new Error(errBody.detail || `HTTP ${response.status}`);
                }
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let doneFired = false;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });

                    // Parse SSE lines from buffer
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || ''; // keep incomplete line in buffer

                    let currentEvent = '';
                    for (const line of lines) {
                        if (line.startsWith('event:')) {
                            currentEvent = line.slice(6).trim();
                        } else if (line.startsWith('data:')) {
                            const dataStr = line.slice(5).trim();
                            if (!dataStr) continue;
                            try {
                                const parsed = JSON.parse(dataStr);
                                if (currentEvent === 'token' && parsed.text != null) {
                                    onToken(parsed.text);
                                } else if (currentEvent === 'done') {
                                    if (!doneFired) { doneFired = true; onDone?.(); }
                                } else if (currentEvent === 'error') {
                                    onError?.(new Error(parsed.detail || 'Stream error'));
                                }
                            } catch (parseErr) {
                                console.debug('SSE parse skip:', parseErr.message);
                                // ignore malformed JSON chunks
                            }
                            currentEvent = '';
                        }
                    }
                }

                // Flush remaining buffer
                if (buffer.trim()) {
                    const remaining = buffer.trim();
                    if (remaining.startsWith('data:')) {
                        try {
                            const parsed = JSON.parse(remaining.slice(5).trim());
                            if (parsed.text != null) onToken(parsed.text);
                        } catch (flushErr) { console.debug('SSE flush parse:', flushErr.message); }
                    }
                }

                if (!doneFired) { doneFired = true; onDone?.(); }
            })
            .catch((err) => {
                if (err.name === 'AbortError') return; // intentional cancel
                onError?.(err);
            });

        return { abort: () => abortController.abort() };
    },
    getContext: (projectId) => api.get(`/projects/${projectId}/chat/context`),
    savePrompt: (projectId, data) => api.post(`/projects/${projectId}/prompts`, data),
    listPrompts: (projectId) => api.get(`/projects/${projectId}/prompts`),
};

// ===== Hardware =====
export const hardwareAPI = {
    status: () => api.get('/hardware/'),
    deviceInfo: () => api.get('/hardware/device'),
    refreshDevice: () => api.post('/hardware/refresh-device'),
};

// ===== LM Studio =====
export const lmstudioAPI = {
    getConfig: () => api.get('/lmstudio/config'),
    setConfig: (data) => api.put('/lmstudio/config', data),
    testConnection: () => api.post('/lmstudio/test'),
    listModels: () => api.get('/lmstudio/models'),
};

// ===== Admin =====
export const adminAPI = {
    listUsers: (params = {}) => api.get('/admin/users', { params }),
    deleteUser: (userId) => api.delete(`/admin/users/${userId}`),
    updateUserRole: (userId, role) => api.patch(`/admin/users/${userId}`, null, { params: { role } }),
    stats: () => api.get('/admin/stats'),
    listCache: () => api.get('/admin/cache'),
    evictCache: (modelName) => api.delete(`/admin/cache/${encodeURIComponent(modelName)}`),
};

// ===== Backup =====
export const backupAPI = {
    export: (projectId) =>
        api.get(`/projects/${projectId}/backup`, { responseType: 'blob', timeout: 300000 }),
    import: (formData) =>
        api.post('/projects/import', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            timeout: 300000,
        }),
};

// ===== Lineage =====
export const lineageAPI = {
    getProjectLineage: (projectId) => api.get(`/projects/${projectId}/lineage`),
    getRunLineage: (projectId, runId) => api.get(`/projects/${projectId}/lineage/runs/${runId}`),
};

export default api;
