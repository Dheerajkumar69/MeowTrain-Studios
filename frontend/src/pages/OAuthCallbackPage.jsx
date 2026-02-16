import { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Cat, Loader2, AlertTriangle } from 'lucide-react';

/**
 * OAuth callback handler.
 *
 * After OAuth (Google/GitHub), the backend redirects here with ?token=JWT.
 * We store the token and redirect to the dashboard.
 */
export default function OAuthCallbackPage() {
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const { refreshUser } = useAuth();
    const [error, setError] = useState('');

    useEffect(() => {
        const token = searchParams.get('token');
        const err = searchParams.get('error');

        if (err) {
            setError(err);
            return;
        }

        if (!token) {
            setError('No authentication token received.');
            return;
        }

        // Store token and redirect
        localStorage.setItem('meowllm_token', token);
        refreshUser().then(() => {
            navigate('/', { replace: true });
        }).catch(() => {
            navigate('/', { replace: true });
        });
    }, [searchParams, navigate, refreshUser]);

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
                    {error ? (
                        <>
                            <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-danger-400/10 mb-4">
                                <AlertTriangle className="w-6 h-6 text-danger-500" />
                            </div>
                            <h2 className="text-lg font-semibold text-surface-900 mb-2">Authentication Failed</h2>
                            <p className="text-surface-500 text-sm mb-4">{error}</p>
                            <button
                                onClick={() => navigate('/login')}
                                className="text-primary-600 hover:text-primary-700 font-medium text-sm"
                            >
                                ← Back to Login
                            </button>
                        </>
                    ) : (
                        <>
                            <Loader2 className="w-10 h-10 text-primary-500 animate-spin mx-auto mb-4" />
                            <h2 className="text-lg font-semibold text-surface-900 mb-2">Signing you in...</h2>
                            <p className="text-surface-500 text-sm">Please wait a moment.</p>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
