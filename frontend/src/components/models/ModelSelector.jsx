import { useState, useEffect } from 'react';
import { modelsAPI } from '../../services/api';
import { Check, Download, AlertTriangle, XCircle, Loader, Info } from 'lucide-react';

const COMPAT_CONFIG = {
    compatible: { label: 'Perfect fit', color: 'text-success-600 bg-success-400/10', icon: Check },
    may_be_slow: { label: 'May be slow', color: 'text-amber-600 bg-amber-400/10', icon: AlertTriangle },
    too_large: { label: 'Too large', color: 'text-danger-600 bg-danger-400/10', icon: XCircle },
    unknown: { label: 'Unknown', color: 'text-surface-500 bg-surface-100', icon: Info },
};

export default function ModelSelector({ selectedModel, onSelectModel }) {
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        modelsAPI.list()
            .then((res) => setModels(res.data || []))
            .catch((err) => { console.error(err); setError('Failed to load models. Check backend connection.'); })
            .finally(() => setLoading(false));
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center py-20">
                <Loader className="w-6 h-6 text-primary-500 animate-spin" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-xl font-bold text-surface-900">Choose a Base Model</h2>
                <p className="text-sm text-surface-500 mt-1">
                    Select the AI model to start from. Compatibility is checked against your hardware.
                </p>
            </div>

            {error && (
                <div className="bg-danger-400/10 border border-danger-400/30 text-danger-600 text-sm rounded-xl p-4 flex items-center gap-2">
                    <XCircle className="w-4 h-4 shrink-0" /> {error}
                </div>
            )}

            {!error && models.length === 0 && !loading && (
                <div className="text-center py-16">
                    <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-surface-100 mb-4">
                        <Info className="w-7 h-7 text-surface-400" />
                    </div>
                    <h3 className="text-lg font-semibold text-surface-700 mb-1">No Models Available</h3>
                    <p className="text-sm text-surface-500">The model catalog is empty. Check backend configuration.</p>
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {models.map((model) => {
                    const compat = COMPAT_CONFIG[model.compatibility] || COMPAT_CONFIG.unknown;
                    const CompatIcon = compat.icon;
                    const isSelected = selectedModel?.model_id === model.model_id;

                    return (
                        <div
                            key={model.model_id}
                            onClick={() => onSelectModel(model)}
                            className={`bg-white rounded-2xl border-2 p-5 cursor-pointer transition-all hover:shadow-lg ${isSelected
                                ? 'border-primary-500 shadow-md ring-2 ring-primary-500/20'
                                : 'border-surface-100 hover:border-primary-200'
                                }`}
                        >
                            {/* Header */}
                            <div className="flex items-start justify-between mb-4">
                                <div className="flex items-center gap-3">
                                    <span className="text-3xl">{model.icon}</span>
                                    <div>
                                        <h3 className="font-semibold text-surface-900">{model.name}</h3>
                                        <p className="text-xs text-surface-500">{model.parameters} parameters</p>
                                    </div>
                                </div>
                                {isSelected && (
                                    <div className="w-6 h-6 rounded-full bg-primary-500 flex items-center justify-center">
                                        <Check className="w-3.5 h-3.5 text-white" />
                                    </div>
                                )}
                            </div>

                            {/* Description */}
                            <p className="text-sm text-surface-600 mb-4 leading-relaxed">{model.description}</p>

                            {/* Specs */}
                            <div className="space-y-2 mb-4">
                                <div className="flex justify-between text-xs">
                                    <span className="text-surface-500">Download size</span>
                                    <span className="font-medium text-surface-700">{model.size_gb} GB</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-surface-500">RAM required</span>
                                    <span className="font-medium text-surface-700">{model.ram_required_gb} GB</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-surface-500">GPU memory</span>
                                    <span className="font-medium text-surface-700">{model.vram_required_gb} GB</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-surface-500">Est. training time</span>
                                    <span className="font-medium text-surface-700">~{model.estimated_train_minutes} min</span>
                                </div>
                            </div>

                            {/* Compatibility Badge */}
                            <div className="flex items-center justify-between">
                                <span className={`inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full ${compat.color}`}>
                                    <CompatIcon className="w-3.5 h-3.5" />
                                    {compat.label}
                                </span>
                                {model.is_cached ? (
                                    <span className="text-xs text-success-600 font-medium flex items-center gap-1">
                                        <Check className="w-3 h-3" /> Cached
                                    </span>
                                ) : (
                                    <span className="text-xs text-surface-400 flex items-center gap-1">
                                        <Download className="w-3 h-3" /> Needs download
                                    </span>
                                )}
                            </div>

                            {/* Select Button */}
                            <button
                                onClick={(e) => { e.stopPropagation(); onSelectModel(model); }}
                                className={`w-full mt-4 py-2 rounded-xl text-sm font-medium transition-all ${isSelected
                                    ? 'bg-primary-500 text-white'
                                    : 'bg-surface-50 text-surface-700 hover:bg-primary-50 hover:text-primary-700 border border-surface-200'
                                    }`}
                            >
                                {isSelected ? '✓ Selected' : 'Use This Model'}
                            </button>
                        </div>
                    );
                })}
            </div>

            {selectedModel && (
                <div className="bg-primary-50 border border-primary-200 rounded-xl p-4 flex items-center gap-3">
                    <span className="text-2xl">{selectedModel.icon}</span>
                    <div>
                        <p className="text-sm font-medium text-primary-800">
                            Selected: {selectedModel.name} ({selectedModel.parameters})
                        </p>
                        <p className="text-xs text-primary-600">
                            {selectedModel.recommended_hardware}
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
}
