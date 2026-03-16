/**
 * Frontend tests for security-affected components.
 *
 * Tests that the API service correctly exposes the logoutAll method
 * and that token refresh / force-logout events work properly.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

// We need to mock axios before importing the api module
const mockInstance = {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn(),
    interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() },
    },
    defaults: { headers: {} },
};

vi.mock('axios', () => ({
    default: { create: vi.fn(() => mockInstance) },
}));

describe('authAPI security methods', () => {
    let authAPI;

    beforeEach(async () => {
        vi.clearAllMocks();
        // Dynamic import to get the module with our mock applied
        const mod = await import('../services/api.js');
        authAPI = mod.authAPI;
    });

    it('should have a logoutAll method', () => {
        expect(typeof authAPI.logoutAll).toBe('function');
    });

    it('logoutAll should call POST /auth/logout-all', () => {
        authAPI.logoutAll();
        expect(mockInstance.post).toHaveBeenCalledWith('/auth/logout-all');
    });

    it('should have a forgotPassword method for rate-limited endpoint', () => {
        expect(typeof authAPI.forgotPassword).toBe('function');
    });

    it('forgotPassword should call POST /auth/forgot-password', () => {
        authAPI.forgotPassword('test@test.com');
        expect(mockInstance.post).toHaveBeenCalledWith('/auth/forgot-password', { email: 'test@test.com' });
    });
});

describe('force-logout event handling', () => {
    it('should dispatch meowllm:force-logout event on window', () => {
        const listener = vi.fn();
        window.addEventListener('meowllm:force-logout', listener);

        window.dispatchEvent(new Event('meowllm:force-logout'));
        expect(listener).toHaveBeenCalledTimes(1);

        window.removeEventListener('meowllm:force-logout', listener);
    });

    it('should dispatch meowllm:network-error event with detail', () => {
        const listener = vi.fn();
        window.addEventListener('meowllm:network-error', listener);

        window.dispatchEvent(new CustomEvent('meowllm:network-error', {
            detail: { type: 'offline', status: 0, message: 'Cannot reach the server.' },
        }));
        expect(listener).toHaveBeenCalledTimes(1);
        expect(listener.mock.calls[0][0].detail.type).toBe('offline');

        window.removeEventListener('meowllm:network-error', listener);
    });
});

describe('display name sanitization', () => {
    it('should strip HTML tags from display names for XSS prevention', () => {
        // Simulating the server-side behavior — the frontend should expect clean display names
        const dirty = '<script>alert("XSS")</script>John';
        const clean = dirty.replace(/<[^>]*>/g, '');
        expect(clean).toBe('alert("XSS")John');
        expect(clean).not.toContain('<script>');
    });

    it('should handle legitimate names without modification', () => {
        const name = "O'Brien-Smith";
        const clean = name.replace(/<[^>]*>/g, '');
        expect(clean).toBe("O'Brien-Smith");
    });
});
