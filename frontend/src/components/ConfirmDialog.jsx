import { useEffect, useRef, useCallback } from 'react';
import { AlertTriangle, X } from 'lucide-react';

export default function ConfirmDialog({ open, title, message, confirmLabel = 'Confirm', cancelLabel = 'Cancel', variant = 'danger', onConfirm, onCancel }) {
    const dialogRef = useRef(null);
    const previousFocusRef = useRef(null);

    // Focus trap: cycle Tab/Shift+Tab among focusable elements inside the dialog
    const handleKeyDown = useCallback((e) => {
        if (e.key === 'Escape') {
            onCancel();
            return;
        }
        if (e.key !== 'Tab') return;

        const focusable = dialogRef.current?.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (!focusable || focusable.length === 0) return;

        const first = focusable[0];
        const last = focusable[focusable.length - 1];

        if (e.shiftKey) {
            if (document.activeElement === first) {
                e.preventDefault();
                last.focus();
            }
        } else {
            if (document.activeElement === last) {
                e.preventDefault();
                first.focus();
            }
        }
    }, [onCancel]);

    // Save + restore focus, and auto-focus the dialog on open
    useEffect(() => {
        if (!open) return;
        previousFocusRef.current = document.activeElement;

        // Small delay to let the DOM paint the dialog before focusing
        const raf = requestAnimationFrame(() => {
            const confirm = dialogRef.current?.querySelector('[data-autofocus]');
            confirm?.focus();
        });

        return () => {
            cancelAnimationFrame(raf);
            // Restore focus to the element that opened the dialog
            previousFocusRef.current?.focus?.();
        };
    }, [open]);

    if (!open) return null;

    const btnColors = variant === 'danger'
        ? 'bg-red-600 hover:bg-red-700 text-white'
        : 'bg-primary-600 hover:bg-primary-700 text-white';

    return (
        <div
            className="fixed inset-0 z-[9998] flex items-center justify-center bg-black/50 backdrop-blur-sm animate-fade-in"
            onClick={onCancel}
            role="dialog"
            aria-modal="true"
            aria-labelledby="confirm-dialog-title"
            aria-describedby="confirm-dialog-message"
            onKeyDown={handleKeyDown}
        >
            <div
                ref={dialogRef}
                className="bg-white rounded-2xl shadow-2xl max-w-md w-full mx-4 p-6 animate-scale-in"
                onClick={e => e.stopPropagation()}
            >
                <div className="flex items-start gap-4 mb-4">
                    {variant === 'danger' && (
                        <div className="p-2 bg-red-100 rounded-xl">
                            <AlertTriangle className="w-6 h-6 text-red-600" aria-hidden="true" />
                        </div>
                    )}
                    <div className="flex-1">
                        <h3 id="confirm-dialog-title" className="text-lg font-semibold text-surface-900">{title}</h3>
                        <p id="confirm-dialog-message" className="text-sm text-surface-500 mt-1">{message}</p>
                    </div>
                    <button onClick={onCancel} className="p-1 rounded-lg hover:bg-surface-100 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500/50" aria-label="Close dialog">
                        <X className="w-5 h-5 text-surface-400" aria-hidden="true" />
                    </button>
                </div>
                <div className="flex justify-end gap-3">
                    <button
                        onClick={onCancel}
                        className="px-4 py-2.5 text-sm font-medium text-surface-700 bg-surface-100 rounded-xl hover:bg-surface-200 transition-colors focus:outline-none focus:ring-2 focus:ring-surface-400/50"
                    >
                        {cancelLabel}
                    </button>
                    <button
                        onClick={onConfirm}
                        data-autofocus
                        className={`px-4 py-2.5 text-sm font-medium rounded-xl transition-colors focus:outline-none focus:ring-2 focus:ring-offset-1 ${btnColors}`}
                    >
                        {confirmLabel}
                    </button>
                </div>
            </div>
        </div>
    );
}
