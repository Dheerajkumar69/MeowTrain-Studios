import { useState, useEffect } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { authAPI } from '../services/api';
import { Cat, CheckCircle, AlertTriangle, Loader2 } from 'lucide-react';

export default function VerifyEmailPage() {
    const [searchParams] = useSearchParams();
    const token = searchParams.get('token');
    const [status, setStatus] = useState(() => token ? 'verifying' : 'error');
    const [message, setMessage] = useState(() => token ? '' : 'No verification token found. Please check your email link.');

    useEffect(() => {
        if (!token) return;

        const verify = async () => {
            try {
                const res = await authAPI.verifyEmail(token);
                setStatus('success');
                setMessage(res.data?.detail || 'Email verified successfully!');
            } catch (err) {
                setStatus('error');
                setMessage(err.response?.data?.detail || 'Verification failed. The link may have expired.');
            }
        };
        verify();
    }, [token]);

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-50 via-surface-0 to-primary-100 p-4">
            <div className="w-full max-w-md animate-fade-in">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 shadow-lg mb-4">
                        <Cat className="w-8 h-8 text-white" />
                    </div>
                    <h1 className="text-3xl font-bold gradient-text">MeowLLM Studio</h1>
                </div>

                <div className="bg-white rounded-2xl shadow-xl border border-surface-100 p-8 text-center">
                    {status === 'verifying' && (
                        <>
                            <Loader2 className="w-12 h-12 text-primary-500 animate-spin mx-auto mb-4" />
                            <h2 className="text-xl font-semibold text-surface-900 mb-2">Verifying your email...</h2>
                            <p className="text-surface-500 text-sm">Please wait a moment.</p>
                        </>
                    )}

                    {status === 'success' && (
                        <>
                            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-emerald-50 mb-4">
                                <CheckCircle className="w-8 h-8 text-emerald-500" />
                            </div>
                            <h2 className="text-xl font-semibold text-surface-900 mb-2">Email Verified!</h2>
                            <p className="text-surface-500 text-sm mb-6">{message}</p>
                            <Link
                                to="/login"
                                className="inline-flex items-center gap-2 px-6 py-2.5 bg-gradient-to-r from-primary-600 to-primary-500 text-white rounded-xl font-medium text-sm shadow-md hover:shadow-lg"
                            >
                                Sign In
                            </Link>
                        </>
                    )}

                    {status === 'error' && (
                        <>
                            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-danger-400/10 mb-4">
                                <AlertTriangle className="w-8 h-8 text-danger-500" />
                            </div>
                            <h2 className="text-xl font-semibold text-surface-900 mb-2">Verification Failed</h2>
                            <p className="text-surface-500 text-sm mb-6">{message}</p>
                            <Link
                                to="/login"
                                className="inline-flex items-center gap-2 px-6 py-2.5 bg-surface-100 text-surface-700 rounded-xl font-medium text-sm hover:bg-surface-200"
                            >
                                Back to Login
                            </Link>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
