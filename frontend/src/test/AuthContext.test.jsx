/**
 * AuthContext tests — login / register / guest / logout / error handling
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';

// Manually create mock functions so we can control them
const mockLogin = vi.fn();
const mockRegister = vi.fn();
const mockGuest = vi.fn();
const mockMe = vi.fn();

vi.mock('../services/api', () => ({
    authAPI: {
        login: (...args) => mockLogin(...args),
        register: (...args) => mockRegister(...args),
        guest: (...args) => mockGuest(...args),
        me: (...args) => mockMe(...args),
    },
}));

import { AuthProvider, useAuth } from '../contexts/AuthContext';

// Storage mock
const storageMock = (() => {
    let store = {};
    return {
        getItem: vi.fn((key) => store[key] ?? null),
        setItem: vi.fn((key, val) => { store[key] = String(val); }),
        removeItem: vi.fn((key) => { delete store[key]; }),
        clear: () => { store = {}; },
    };
})();
Object.defineProperty(globalThis, 'localStorage', { value: storageMock, writable: true });

// Helper component that exposes auth context for assertions
function AuthConsumer({ onRender }) {
    const auth = useAuth();
    onRender(auth);
    return (
        <div>
            <span data-testid="user">{auth.user?.email ?? 'none'}</span>
            <span data-testid="loading">{String(auth.loading)}</span>
        </div>
    );
}

function renderWithAuth(onRender = () => {}) {
    return render(
        <AuthProvider>
            <AuthConsumer onRender={onRender} />
        </AuthProvider>,
    );
}

describe('AuthContext', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        storageMock.clear();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('resolves to null user when no token in storage', async () => {
        renderWithAuth();
        // No token → useEffect skips the /me call and sets loading=false immediately
        await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'));
        expect(screen.getByTestId('user').textContent).toBe('none');
        expect(mockMe).not.toHaveBeenCalled();
    });

    it('fetches /me when a token exists in localStorage', async () => {
        storageMock.setItem('meowllm_token', 'existing-jwt');
        mockMe.mockResolvedValue({ data: { email: 'existing@test.com' } });

        renderWithAuth();
        await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'));
        expect(mockMe).toHaveBeenCalledOnce();
        expect(screen.getByTestId('user').textContent).toBe('existing@test.com');
    });

    it('clears token on 401 from /me', async () => {
        storageMock.setItem('meowllm_token', 'expired-jwt');
        mockMe.mockRejectedValue({ response: { status: 401 } });

        renderWithAuth();
        await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'));
        expect(storageMock.removeItem).toHaveBeenCalledWith('meowllm_token');
        expect(screen.getByTestId('user').textContent).toBe('none');
    });

    it('login() sets token + user', async () => {
        mockLogin.mockResolvedValue({
            data: { token: 'jwt-123', user: { email: 'user@test.com' } },
        });

        let authRef;
        renderWithAuth((ctx) => { authRef = ctx; });
        await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'));

        await act(async () => {
            await authRef.login('user@test.com', 'pass');
        });

        expect(storageMock.setItem).toHaveBeenCalledWith('meowllm_token', 'jwt-123');
        expect(screen.getByTestId('user').textContent).toBe('user@test.com');
    });

    it('register() sets token + user', async () => {
        mockRegister.mockResolvedValue({
            data: { token: 'jwt-reg', user: { email: 'new@test.com' } },
        });

        let authRef;
        renderWithAuth((ctx) => { authRef = ctx; });
        await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'));

        await act(async () => {
            await authRef.register('new@test.com', 'pass', 'New User');
        });

        expect(storageMock.setItem).toHaveBeenCalledWith('meowllm_token', 'jwt-reg');
        expect(screen.getByTestId('user').textContent).toBe('new@test.com');
    });

    it('guestLogin() sets token + user', async () => {
        mockGuest.mockResolvedValue({
            data: { token: 'jwt-guest', user: { email: 'guest@meow.local' } },
        });

        let authRef;
        renderWithAuth((ctx) => { authRef = ctx; });
        await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'));

        await act(async () => {
            await authRef.guestLogin();
        });

        expect(storageMock.setItem).toHaveBeenCalledWith('meowllm_token', 'jwt-guest');
        expect(screen.getByTestId('user').textContent).toBe('guest@meow.local');
    });

    it('logout() clears token + sets user to null', async () => {
        storageMock.setItem('meowllm_token', 'jwt-active');
        mockMe.mockResolvedValue({ data: { email: 'active@test.com' } });

        let authRef;
        renderWithAuth((ctx) => { authRef = ctx; });
        await waitFor(() => expect(screen.getByTestId('user').textContent).toBe('active@test.com'));

        act(() => authRef.logout());

        expect(storageMock.removeItem).toHaveBeenCalledWith('meowllm_token');
        expect(screen.getByTestId('user').textContent).toBe('none');
    });

    it('useAuth() throws when used outside AuthProvider', () => {
        // Suppress React error boundary console noise
        const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
        function BadConsumer() {
            useAuth();
            return null;
        }
        expect(() => render(<BadConsumer />)).toThrow('useAuth must be used within AuthProvider');
        spy.mockRestore();
    });
});
