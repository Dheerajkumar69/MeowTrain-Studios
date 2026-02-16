import React from 'react';
import { Cat, AlertTriangle, RefreshCw } from 'lucide-react';

export class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, info) {
        console.error('[ErrorBoundary] Caught error:', error, info?.componentStack);
    }

    render() {
        if (this.state.hasError) {
            const compact = this.props.compact;
            if (compact) {
                return (
                    <div className="flex flex-col items-center justify-center p-6 min-h-[200px]" role="alert">
                        <div className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-danger-400/10 mb-3">
                            <AlertTriangle className="w-5 h-5 text-danger-500" aria-hidden="true" />
                        </div>
                        <p className="text-sm font-medium text-surface-800 mb-1">
                            {this.props.label ? `${this.props.label} crashed` : 'This section crashed'}
                        </p>
                        <p className="text-xs text-surface-500 mb-3 text-center max-w-xs">
                            {this.state.error?.message || 'An unexpected error occurred.'}
                        </p>
                        <button
                            onClick={() => this.setState({ hasError: false, error: null })}
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-primary-600 text-white rounded-lg text-xs font-medium hover:bg-primary-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500/50"
                        >
                            <RefreshCw className="w-3 h-3" aria-hidden="true" />
                            Retry
                        </button>
                    </div>
                );
            }

            return (
                <div className="min-h-[400px] flex items-center justify-center p-8" role="alert">
                    <div className="text-center max-w-md">
                        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-danger-400/10 mb-4">
                            <AlertTriangle className="w-8 h-8 text-danger-500" aria-hidden="true" />
                        </div>
                        <h2 className="text-xl font-semibold text-surface-900 mb-2">
                            Something went wrong
                        </h2>
                        <p className="text-surface-500 text-sm mb-6">
                            {this.state.error?.message || 'An unexpected error occurred.'}
                        </p>
                        <button
                            onClick={() => this.setState({ hasError: false, error: null })}
                            className="inline-flex items-center gap-2 px-4 py-2.5 bg-primary-600 text-white rounded-xl text-sm font-medium hover:bg-primary-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500/50"
                        >
                            <RefreshCw className="w-4 h-4" aria-hidden="true" />
                            Try Again
                        </button>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
