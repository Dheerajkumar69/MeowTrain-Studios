import { createContext, useContext, useState, useEffect } from 'react';
import { authAPI } from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

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
        } else {
            setLoading(false);
        }
    }, []);

    const login = async (email, password) => {
        const res = await authAPI.login({ email, password });
        localStorage.setItem('meowllm_token', res.data.token);
        setUser(res.data.user);
        return res.data;
    };

    const register = async (email, password, displayName) => {
        const res = await authAPI.register({ email, password, display_name: displayName });
        localStorage.setItem('meowllm_token', res.data.token);
        setUser(res.data.user);
        return res.data;
    };

    const guestLogin = async () => {
        const res = await authAPI.guest();
        localStorage.setItem('meowllm_token', res.data.token);
        setUser(res.data.user);
        return res.data;
    };

    const logout = () => {
        localStorage.removeItem('meowllm_token');
        setUser(null);
    };

    return (
        <AuthContext.Provider value={{ user, loading, login, register, guestLogin, logout }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) throw new Error('useAuth must be used within AuthProvider');
    return context;
}
