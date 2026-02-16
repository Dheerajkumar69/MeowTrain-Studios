import { lazy, Suspense, useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ErrorBoundary } from './components/ErrorBoundary';
import { ToastProvider } from './components/Toast';

// Route-level code splitting for smaller initial bundle
const LoginPage = lazy(() => import('./pages/LoginPage'));
const RegisterPage = lazy(() => import('./pages/RegisterPage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const ProjectPage = lazy(() => import('./pages/ProjectPage'));
const VerifyEmailPage = lazy(() => import('./pages/VerifyEmailPage'));
const ResetPasswordPage = lazy(() => import('./pages/ResetPasswordPage'));
const OAuthCallbackPage = lazy(() => import('./pages/OAuthCallbackPage'));

function LazyFallback() {
  return (
    <div className="min-h-screen bg-surface-50 flex items-center justify-center">
      <div className="w-8 h-8 border-3 border-primary-200 border-t-primary-600 rounded-full animate-spin" />
    </div>
  );
}

function NetworkBanner() {
  const [msg, setMsg] = useState(null);
  useEffect(() => {
    const handler = (e) => {
      setMsg(e.detail?.message || 'Connection issue detected.');
      setTimeout(() => setMsg(null), 15000);
    };
    window.addEventListener('meowllm:network-error', handler);
    return () => window.removeEventListener('meowllm:network-error', handler);
  }, []);
  if (!msg) return null;
  return (
    <div className="fixed top-0 inset-x-0 z-[100] bg-amber-500 text-white text-center text-sm py-2 px-4 flex items-center justify-center gap-2 animate-fade-in">
      <span>⚠️ {msg}</span>
      <button onClick={() => setMsg(null)} className="ml-2 text-white/80 hover:text-white font-bold">✕</button>
    </div>
  );
}

function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();
  if (loading) {
    return (
      <div className="min-h-screen bg-surface-50 flex items-center justify-center">
        <div className="w-8 h-8 border-3 border-primary-200 border-t-primary-600 rounded-full animate-spin" />
      </div>
    );
  }
  return user ? children : <Navigate to="/login" />;
}

function PublicRoute({ children }) {
  const { user, loading } = useAuth();
  if (loading) return null;
  return user ? <Navigate to="/" /> : children;
}

function NotFoundPage() {
  return (
    <div className="min-h-screen bg-surface-50 flex items-center justify-center">
      <div className="text-center max-w-md">
        <p className="text-6xl font-bold text-surface-300 mb-4">404</p>
        <h2 className="text-xl font-semibold text-surface-800 mb-2">Page Not Found</h2>
        <p className="text-surface-500 text-sm mb-6">The page you're looking for doesn't exist or has been moved.</p>
        <a href="/" className="inline-flex items-center gap-2 px-4 py-2.5 bg-primary-600 text-white rounded-xl text-sm font-medium hover:bg-primary-700 transition-colors">
          Go to Dashboard
        </a>
      </div>
    </div>
  );
}

function AppRoutes() {
  return (
    <Suspense fallback={<LazyFallback />}>
      <Routes>
        <Route path="/login" element={<PublicRoute><ErrorBoundary><LoginPage /></ErrorBoundary></PublicRoute>} />
        <Route path="/register" element={<PublicRoute><ErrorBoundary><RegisterPage /></ErrorBoundary></PublicRoute>} />
        <Route path="/verify-email" element={<ErrorBoundary><VerifyEmailPage /></ErrorBoundary>} />
        <Route path="/reset-password" element={<ErrorBoundary><ResetPasswordPage /></ErrorBoundary>} />
        <Route path="/forgot-password" element={<ErrorBoundary><ResetPasswordPage /></ErrorBoundary>} />
        <Route path="/oauth/callback" element={<ErrorBoundary><OAuthCallbackPage /></ErrorBoundary>} />
        <Route path="/" element={<ProtectedRoute><ErrorBoundary><DashboardPage /></ErrorBoundary></ProtectedRoute>} />
        <Route path="/project/:id" element={<ProtectedRoute><ErrorBoundary><ProjectPage /></ErrorBoundary></ProtectedRoute>} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </Suspense>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <NetworkBanner />
      <ErrorBoundary>
        <ToastProvider>
          <AuthProvider>
            <AppRoutes />
          </AuthProvider>
        </ToastProvider>
      </ErrorBoundary>
    </BrowserRouter>
  );
}
