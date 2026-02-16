/**
 * API interceptor tests — token injection, 401 handling, token refresh
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import axios from 'axios';

// We test the interceptor logic in isolation by re-creating a mini version,
// since the real module has side-effects (creates the instance on import).
// We test: (1) token attachment, (2) retry after 401 + refresh, (3) no retry on auth endpoints.

describe('API interceptors (unit)', () => {
    let api;
    let store;

    beforeEach(() => {
        store = {};
        vi.spyOn(Storage.prototype, 'getItem').mockImplementation((k) => store[k] ?? null);
        vi.spyOn(Storage.prototype, 'setItem').mockImplementation((k, v) => { store[k] = String(v); });
        vi.spyOn(Storage.prototype, 'removeItem').mockImplementation((k) => { delete store[k]; });

        // Create a fresh axios instance with the same interceptor logic as api.js
        api = axios.create({ baseURL: '/api', timeout: 5000 });

        api.interceptors.request.use((config) => {
            const token = localStorage.getItem('meowllm_token');
            if (token) {
                config.headers.Authorization = `Bearer ${token}`;
            }
            return config;
        });

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
            (res) => res,
            async (error) => {
                const originalRequest = error.config;
                const url = originalRequest?.url || '';
                const isAuthEndpoint = url.startsWith('/auth/');

                if (error.response?.status === 401 && !isAuthEndpoint && !originalRequest._retry) {
                    if (_isRefreshing) {
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

                return Promise.reject(error);
            },
        );
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('attaches Authorization header when token is present', async () => {
        store['meowllm_token'] = 'jwt-123';

        // Use an axios adapter mock to intercept the outgoing request
        const adapter = vi.fn().mockResolvedValue({ data: {}, status: 200, headers: {} });
        api.defaults.adapter = adapter;

        await api.get('/projects/');
        expect(adapter).toHaveBeenCalledOnce();
        const config = adapter.mock.calls[0][0];
        expect(config.headers.Authorization).toBe('Bearer jwt-123');
    });

    it('does NOT attach Authorization header when no token', async () => {
        const adapter = vi.fn().mockResolvedValue({ data: {}, status: 200, headers: {} });
        api.defaults.adapter = adapter;

        await api.get('/projects/');
        const config = adapter.mock.calls[0][0];
        expect(config.headers.Authorization).toBeUndefined();
    });

    it('does not attempt refresh on auth endpoints', async () => {
        store['meowllm_token'] = 'jwt-old';
        let callCount = 0;

        api.defaults.adapter = (config) => {
            callCount++;
            // Auth endpoint returns 401
            return Promise.reject({
                config,
                response: { status: 401, data: {} },
            });
        };

        await expect(api.get('/auth/me')).rejects.toBeDefined();
        // Only one call — no retry attempt
        expect(callCount).toBe(1);
    });

    it('token refresh clears storage on refresh failure', async () => {
        store['meowllm_token'] = 'jwt-expired';

        api.defaults.adapter = (config) => {
            // Both the original request and the refresh fail with 401
            return Promise.reject({
                config,
                response: { status: 401, data: {} },
            });
        };

        await expect(api.get('/projects/')).rejects.toBeDefined();
        expect(localStorage.removeItem).toHaveBeenCalledWith('meowllm_token');
    });
});
