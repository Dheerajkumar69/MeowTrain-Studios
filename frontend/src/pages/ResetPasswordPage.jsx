import { useState } from 'react';
import { useSearchParams, Link, useNavigate } from 'react-router-dom';
import { authAPI } from '../services/api';
import { Cat, Lock, ArrowRight, Eye, EyeOff, CheckCircle, Mail } from 'lucide-react';

export default function ResetPasswordPage() {
    const [searchParams] = useSearchParams();
    const token = searchParams.get('token');

    // If no token, show the "forgot password" request form
    if (!token) {
        return <ForgotPasswordForm />;
    }

    return <ResetPasswordForm token={token} />;
}

function ForgotPasswordForm() {
    const [email, setEmail] = useState('');
    const [submitted, setSubmitted] = useState(false);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            await authAPI.forgotPassword(email);
            setSubmitted(true);
        } catch (err) {
            // Always show success to prevent email enumeration
            setSubmitted(true);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-50 via-surface-0 to-primary-100 p-4">
            <div className="w-full max-w-md animate-fade-in">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 shadow-lg mb-4">
                        <Cat className="w-8 h-8 text-white" />
                    </div>
                    <h1 className="text-3xl font-bold gradient-text">MeowLLM Studio</h1>
                </div>

                <div className="bg-white rounded-2xl shadow-xl border border-surface-100 p-8">
                    {submitted ? (
                        <div className="text-center">
                            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-emerald-50 mb-4">
                                <Mail className="w-8 h-8 text-emerald-500" />
                            </div>
                            <h2 className="text-xl font-semibold text-surface-900 mb-2">Check Your Email</h2>
                            <p className="text-surface-500 text-sm mb-6">
                                If an account exists for {email}, we've sent a password reset link.
                            </p>
                            <Link
                                to="/login"
                                className="text-primary-600 hover:text-primary-700 font-medium text-sm"
                            >
                                ← Back to Login
                            </Link>
                        </div>
                    ) : (
                        <>
                            <h2 className="text-xl font-semibold text-surface-900 mb-2">Forgot Password</h2>
                            <p className="text-surface-500 text-sm mb-6">
                                Enter your email and we'll send you a reset link.
                            </p>
                            {error && (
                                <div className="bg-danger-400/10 border border-danger-400/30 text-danger-600 text-sm rounded-xl p-3 mb-4">
                                    {error}
                                </div>
                            )}
                            <form onSubmit={handleSubmit} className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-surface-700 mb-1.5">Email</label>
                                    <div className="relative">
                                        <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-surface-400" />
                                        <input
                                            type="email"
                                            value={email}
                                            onChange={(e) => setEmail(e.target.value)}
                                            className="w-full pl-10 pr-4 py-2.5 border border-surface-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-500"
                                            placeholder="you@example.com"
                                            required
                                        />
                                    </div>
                                </div>
                                <button
                                    type="submit"
                                    disabled={loading}
                                    className="w-full py-2.5 bg-gradient-to-r from-primary-600 to-primary-500 text-white rounded-xl font-medium text-sm hover:from-primary-700 hover:to-primary-600 transition-all shadow-md disabled:opacity-50"
                                >
                                    {loading ? 'Sending...' : 'Send Reset Link'}
                                </button>
                            </form>
                            <p className="text-center text-sm text-surface-500 mt-4">
                                <Link to="/login" className="text-primary-600 hover:text-primary-700 font-medium">
                                    ← Back to Login
                                </Link>
                            </p>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}

function ResetPasswordForm({ token }) {
    const navigate = useNavigate();
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState(false);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (password.length < 8) {
            setError('Password must be at least 8 characters');
            return;
        }
        setError('');
        setLoading(true);
        try {
            await authAPI.resetPassword(token, password);
            setSuccess(true);
            setTimeout(() => navigate('/login'), 3000);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to reset password. The link may have expired.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-50 via-surface-0 to-primary-100 p-4">
            <div className="w-full max-w-md animate-fade-in">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 shadow-lg mb-4">
                        <Cat className="w-8 h-8 text-white" />
                    </div>
                    <h1 className="text-3xl font-bold gradient-text">MeowLLM Studio</h1>
                </div>

                <div className="bg-white rounded-2xl shadow-xl border border-surface-100 p-8">
                    {success ? (
                        <div className="text-center">
                            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-emerald-50 mb-4">
                                <CheckCircle className="w-8 h-8 text-emerald-500" />
                            </div>
                            <h2 className="text-xl font-semibold text-surface-900 mb-2">Password Reset!</h2>
                            <p className="text-surface-500 text-sm">Redirecting to login...</p>
                        </div>
                    ) : (
                        <>
                            <h2 className="text-xl font-semibold text-surface-900 mb-6">Set New Password</h2>
                            {error && (
                                <div className="bg-danger-400/10 border border-danger-400/30 text-danger-600 text-sm rounded-xl p-3 mb-4">
                                    {error}
                                </div>
                            )}
                            <form onSubmit={handleSubmit} className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-surface-700 mb-1.5">New Password</label>
                                    <div className="relative">
                                        <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-surface-400" />
                                        <input
                                            type={showPassword ? 'text' : 'password'}
                                            value={password}
                                            onChange={(e) => setPassword(e.target.value)}
                                            className="w-full pl-10 pr-10 py-2.5 border border-surface-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-500"
                                            placeholder="At least 8 characters"
                                            required
                                            minLength={8}
                                            maxLength={128}
                                        />
                                        <button
                                            type="button"
                                            onClick={() => setShowPassword(!showPassword)}
                                            className="absolute right-3 top-1/2 -translate-y-1/2 text-surface-400 hover:text-surface-600"
                                        >
                                            {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                        </button>
                                    </div>
                                </div>
                                <button
                                    type="submit"
                                    disabled={loading}
                                    className="w-full py-2.5 bg-gradient-to-r from-primary-600 to-primary-500 text-white rounded-xl font-medium text-sm hover:from-primary-700 hover:to-primary-600 transition-all shadow-md disabled:opacity-50 flex items-center justify-center gap-2"
                                >
                                    {loading ? 'Resetting...' : <>Reset Password <ArrowRight className="w-4 h-4" /></>}
                                </button>
                            </form>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
