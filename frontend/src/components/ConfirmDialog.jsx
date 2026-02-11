import { AlertTriangle, X } from 'lucide-react';

export default function ConfirmDialog({ open, title, message, confirmLabel = 'Confirm', cancelLabel = 'Cancel', variant = 'danger', onConfirm, onCancel }) {
    if (!open) return null;

    const btnColors = variant === 'danger'
        ? 'bg-red-600 hover:bg-red-700 text-white'
        : 'bg-primary-600 hover:bg-primary-700 text-white';

    return (
        <div className="fixed inset-0 z-[9998] flex items-center justify-center bg-black/50 backdrop-blur-sm animate-fade-in" onClick={onCancel}>
            <div
                className="bg-white rounded-2xl shadow-2xl max-w-md w-full mx-4 p-6 animate-scale-in"
                onClick={e => e.stopPropagation()}
            >
                <div className="flex items-start gap-4 mb-4">
                    {variant === 'danger' && (
                        <div className="p-2 bg-red-100 rounded-xl">
                            <AlertTriangle className="w-6 h-6 text-red-600" />
                        </div>
                    )}
                    <div className="flex-1">
                        <h3 className="text-lg font-semibold text-surface-900">{title}</h3>
                        <p className="text-sm text-surface-500 mt-1">{message}</p>
                    </div>
                    <button onClick={onCancel} className="p-1 rounded-lg hover:bg-surface-100 transition-colors">
                        <X className="w-5 h-5 text-surface-400" />
                    </button>
                </div>
                <div className="flex justify-end gap-3">
                    <button
                        onClick={onCancel}
                        className="px-4 py-2.5 text-sm font-medium text-surface-700 bg-surface-100 rounded-xl hover:bg-surface-200 transition-colors"
                    >
                        {cancelLabel}
                    </button>
                    <button
                        onClick={onConfirm}
                        className={`px-4 py-2.5 text-sm font-medium rounded-xl transition-colors ${btnColors}`}
                    >
                        {confirmLabel}
                    </button>
                </div>
            </div>
        </div>
    );
}
