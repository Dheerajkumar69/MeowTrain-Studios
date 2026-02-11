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
                return Promise.reject(refreshError);
            } finally {
                _isRefreshing = false;
            }
        }

        // Non-recoverable 401 on auth endpoints — just clear
        if (error.response?.status === 401 && !isAuthEndpoint) {
            localStorage.removeItem('meowllm_token');
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
};

// ===== Models =====
export const modelsAPI = {
    list: () => api.get('/models/'),
    status: (modelId) => api.get(`/models/${encodeURIComponent(modelId)}/status`),
    download: (modelId) => api.post(`/models/${encodeURIComponent(modelId)}/download`),
    cancelDownload: (modelId) => api.delete(`/models/${encodeURIComponent(modelId)}/download`),
    exportModel: (projectId) => api.get(`/models/export/${projectId}`, { responseType: 'blob' }),
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
};

// ===== Inference =====
export const inferenceAPI = {
    chat: (projectId, data) => api.post(`/projects/${projectId}/chat`, data),
    getContext: (projectId) => api.get(`/projects/${projectId}/chat/context`),
    savePrompt: (projectId, data) => api.post(`/projects/${projectId}/prompts`, data),
    listPrompts: (projectId) => api.get(`/projects/${projectId}/prompts`),
};

// ===== Hardware =====
export const hardwareAPI = {
    status: () => api.get('/hardware/'),
};

// ===== LM Studio =====
export const lmstudioAPI = {
    getConfig: () => api.get('/lmstudio/config'),
    setConfig: (data) => api.put('/lmstudio/config', data),
    testConnection: () => api.post('/lmstudio/test'),
    listModels: () => api.get('/lmstudio/models'),
};

export default api;
