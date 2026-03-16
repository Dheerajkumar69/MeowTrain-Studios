import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { authAPI } from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(() => !!localStorage.getItem('meowllm_token'));

    useEffect(() => {
        const token = localStorage.getItem('meowllm_token');
        if (token) {
            authAPI.me()
                .then((res) => setUser(res.data))
                .catch((err) => {
                    // Only clear token on auth failures, not network errors
                    if (err.response?.status === 401 || err.response?.status === 403) {
                        localStorage.removeItem('meowllm_token');
                        setUser(null);
                    }
                })
                .finally(() => setLoading(false));
        }
    }, []);

    const login = useCallback(async (email, password) => {
        const res = await authAPI.login({ email, password });
        localStorage.setItem('meowllm_token', res.data.token);
        setUser(res.data.user);
        return res.data;
    }, []);

    const register = useCallback(async (email, password, displayName) => {
        const res = await authAPI.register({ email, password, display_name: displayName });
        localStorage.setItem('meowllm_token', res.data.token);
        setUser(res.data.user);
        return res.data;
    }, []);

    const guestLogin = useCallback(async () => {
        const res = await authAPI.guest();
        localStorage.setItem('meowllm_token', res.data.token);
        setUser(res.data.user);
        return res.data;
    }, []);

    const logout = useCallback(() => {
        localStorage.removeItem('meowllm_token');
        setUser(null);
    }, []);

    // Listen for force-logout events from the API interceptor (e.g. token refresh failure)
    useEffect(() => {
        const handleForceLogout = () => {
            setUser(null);
        };
        window.addEventListener('meowllm:force-logout', handleForceLogout);
        return () => window.removeEventListener('meowllm:force-logout', handleForceLogout);
    }, []);

    /** Update user state in-place after profile edits (avoids full refetch). */
    const updateUser = useCallback((updates) => {
        setUser((prev) => (prev ? { ...prev, ...updates } : prev));
    }, []);

    /** Re-fetch user from server (e.g. after token refresh). */
    const refreshUser = useCallback(async () => {
        try {
            const res = await authAPI.me();
            setUser(res.data);
        } catch (err) {
            console.debug('User refresh failed:', err.message || err);
        }
    }, []);

    return (
        <AuthContext.Provider value={{ user, loading, login, register, guestLogin, logout, updateUser, refreshUser }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) throw new Error('useAuth must be used within AuthProvider');
    return context;
}
