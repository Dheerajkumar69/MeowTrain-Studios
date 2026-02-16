/**
 * ConfigForm — training configuration panel (method, epochs, LR, advanced settings, etc.).
 */
import { useState } from 'react';
import {
    Zap, Info, ChevronDown, ChevronUp, Loader,
    CheckCircle, AlertCircle, Cpu, RotateCcw, GitCompare, Activity,
} from 'lucide-react';
import { BarGauge } from './Gauges';
import RunComparisonPanel from './RunComparisonPanel';
import { trainingAPI } from '../../services/api';
import { useToast } from '../Toast';

const METHOD_INFO = {
    lora: { label: 'LoRA', desc: 'Fast & memory-efficient. Recommended for most users.' },
    qlora: { label: 'QLoRA', desc: 'Compressed LoRA. Best for GPUs with <6GB VRAM.' },
    full: { label: 'Full Fine-tune', desc: 'Trains all params. Best quality, needs lots of VRAM.' },
    dpo: { label: 'DPO', desc: 'Alignment with preference pairs (chosen/rejected).' },
    orpo: { label: 'ORPO', desc: 'Memory-efficient alignment. No reference model needed.' },
};

const LR_PRESETS = {
    conservative: { value: 1e-4, label: 'Conservative', desc: 'Slower but safer learning' },
    balanced: { value: 2e-4, label: 'Balanced', desc: 'Good default for most tasks' },
    aggressive: { value: 5e-4, label: 'Aggressive', desc: 'Faster learning, risk of instability' },
};

function Tip({ text }) {
    return (
        <div className="group relative inline-block ml-1">
            <Info className="w-3.5 h-3.5 text-surface-400 cursor-help" />
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-surface-800 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none w-48 text-center z-10">
                {text}
                <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-surface-800" />
            </div>
        </div>
    );
}

export default function ConfigForm({
    config, setConfig, selectedModel, readyDatasets, hardware,
    projectId, error, setError, actionPending, onStartTraining,
}) {
    const toast = useToast();
    const [lrPreset, setLrPreset] = useState('balanced');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [showComparison, setShowComparison] = useState(false);
    const [runHistory, setRunHistory] = useState([]);

    return (
        <div className="space-y-6 animate-fade-in">
            <div>
                <h2 className="text-xl font-bold text-surface-900">Training Configuration</h2>
                <p className="text-sm text-surface-500 mt-1">Adjust how your model learns. Safe defaults are pre-selected.</p>
            </div>

            {error && (
                <div className="bg-danger-400/10 border border-danger-400/30 text-danger-600 text-sm rounded-xl p-3 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0" /> {error}
                    <button onClick={() => setError('')} className="ml-auto text-danger-400 hover:text-danger-600">✕</button>
                </div>
            )}

            {/* Live Hardware Preview (even before training) */}
            {hardware && (
                <div className="bg-white rounded-2xl border border-surface-100 p-5">
                    <h3 className="text-sm font-semibold text-surface-700 mb-3 flex items-center gap-2">
                        <Activity className="w-4 h-4 text-primary-500" /> Current Hardware
                        <span className="text-xs text-surface-400 font-normal ml-auto animate-pulse">● Live</span>
                    </h3>
                    <div className="grid grid-cols-2 gap-3">
                        <BarGauge
                            value={hardware.cpu_usage_percent} max={100}
                            label={`CPU (${hardware.cpu_cores} cores)`} color="#10b981"
                            format={`${hardware.cpu_usage_percent?.toFixed(0) ?? '—'}%`}
                        />
                        <BarGauge
                            value={hardware.ram_used_gb} max={hardware.ram_total_gb}
                            label="RAM" color="#3b82f6"
                            format={`${hardware.ram_used_gb?.toFixed(1) ?? '—'} / ${hardware.ram_total_gb?.toFixed(1) ?? '—'} GB`}
                        />
                        {hardware.gpu_available && (
                            <>
                                <BarGauge
                                    value={hardware.gpu_usage_percent} max={100}
                                    label={hardware.gpu_name?.replace('NVIDIA ', '') || 'GPU'} color="#8b5cf6"
                                    format={`${hardware.gpu_usage_percent ?? '—'}% • ${hardware.gpu_temp_celsius ?? '—'}°C`}
                                />
                                <BarGauge
                                    value={hardware.gpu_vram_used_gb} max={hardware.gpu_vram_total_gb}
                                    label="VRAM" color="#a855f7"
                                    format={`${hardware.gpu_vram_used_gb?.toFixed(1) ?? '—'} / ${hardware.gpu_vram_total_gb?.toFixed(1) ?? '—'} GB`}
                                />
                            </>
                        )}
                    </div>
                </div>
            )}

            <div className="bg-white rounded-2xl border border-surface-100 p-6 space-y-6">
                {/* Training Method */}
                <div>
                    <label className="text-sm font-semibold text-surface-700 flex items-center">
                        Training Method <Tip text="LoRA/QLoRA/Full for supervised fine-tuning. DPO/ORPO for alignment with preference data." />
                    </label>
                    <div className="text-xs text-surface-400 mt-1 mb-2">SFT Methods</div>
                    <div className="grid grid-cols-3 gap-3">
                        {Object.entries(METHOD_INFO).filter(([k]) => ['lora', 'qlora', 'full'].includes(k)).map(([key, info]) => (
                            <button
                                key={key}
                                onClick={() => setConfig({ ...config, method: key })}
                                className={`p-3 rounded-xl border text-left transition-all ${config.method === key
                                    ? 'border-primary-500 bg-primary-50 shadow-sm'
                                    : 'border-surface-200 hover:border-surface-300'
                                    }`}
                            >
                                <p className={`text-sm font-medium ${config.method === key ? 'text-primary-700' : 'text-surface-700'}`}>
                                    {info.label}
                                </p>
                                <p className="text-xs text-surface-500 mt-1">{info.desc}</p>
                            </button>
                        ))}
                    </div>
                    <div className="text-xs text-surface-400 mt-3 mb-2">Alignment Methods <span className="text-amber-500">(requires preference pair datasets)</span></div>
                    <div className="grid grid-cols-2 gap-3">
                        {Object.entries(METHOD_INFO).filter(([k]) => ['dpo', 'orpo'].includes(k)).map(([key, info]) => (
                            <button
                                key={key}
                                onClick={() => setConfig({ ...config, method: key })}
                                className={`p-3 rounded-xl border text-left transition-all ${config.method === key
                                    ? 'border-violet-500 bg-violet-50 shadow-sm'
                                    : 'border-surface-200 hover:border-surface-300'
                                    }`}
                            >
                                <p className={`text-sm font-medium ${config.method === key ? 'text-violet-700' : 'text-surface-700'}`}>
                                    {info.label}
                                </p>
                                <p className="text-xs text-surface-500 mt-1">{info.desc}</p>
                            </button>
                        ))}
                    </div>
                    {config.method === 'dpo' && (
                        <div className="mt-3 p-3 bg-violet-50 rounded-xl border border-violet-100">
                            <label className="text-xs font-medium text-violet-700">DPO Beta (KL penalty): {config.dpo_beta}</label>
                            <input type="range" min={0.01} max={1.0} step={0.01} value={config.dpo_beta}
                                onChange={(e) => setConfig({ ...config, dpo_beta: parseFloat(e.target.value) })}
                                className="w-full mt-1 accent-violet-500" />
                        </div>
                    )}
                    {config.method === 'orpo' && (
                        <div className="mt-3 p-3 bg-violet-50 rounded-xl border border-violet-100">
                            <label className="text-xs font-medium text-violet-700">ORPO Alpha (odds ratio weight): {config.orpo_alpha}</label>
                            <input type="range" min={0.1} max={10.0} step={0.1} value={config.orpo_alpha}
                                onChange={(e) => setConfig({ ...config, orpo_alpha: parseFloat(e.target.value) })}
                                className="w-full mt-1 accent-violet-500" />
                        </div>
                    )}
                </div>

                {/* Epochs */}
                <div>
                    <label className="text-sm font-semibold text-surface-700 flex items-center">
                        Epochs: {config.epochs} <Tip text="How many times the model sees all your data. More = better quality but takes longer." />
                    </label>
                    <input
                        type="range" min={1} max={20} value={config.epochs}
                        onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                        className="w-full mt-2 accent-primary-500"
                    />
                    <div className="flex justify-between text-xs text-surface-400 mt-1">
                        <span>1 (Quick)</span><span>10</span><span>20 (Thorough)</span>
                    </div>
                </div>

                {/* Learning Rate */}
                <div>
                    <label className="text-sm font-semibold text-surface-700 flex items-center">
                        Learning Speed <Tip text="How fast the model learns. Slower is safer but takes longer." />
                    </label>
                    <div className="grid grid-cols-3 gap-3 mt-3">
                        {Object.entries(LR_PRESETS).map(([key, preset]) => (
                            <button
                                key={key}
                                onClick={() => { setLrPreset(key); setConfig({ ...config, learning_rate: preset.value }); }}
                                className={`p-3 rounded-xl border text-center transition-all ${lrPreset === key
                                    ? 'border-primary-500 bg-primary-50 shadow-sm'
                                    : 'border-surface-200 hover:border-surface-300'
                                    }`}
                            >
                                <p className={`text-sm font-medium ${lrPreset === key ? 'text-primary-700' : 'text-surface-700'}`}>{preset.label}</p>
                                <p className="text-xs text-surface-500 mt-1">{preset.desc}</p>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Max Tokens */}
                <div>
                    <label className="text-sm font-semibold text-surface-700 flex items-center">
                        Max Tokens: {config.max_tokens} <Tip text="Maximum length of each training example. Longer = more context but slower." />
                    </label>
                    <input
                        type="range" min={64} max={2048} step={64} value={config.max_tokens}
                        onChange={(e) => setConfig({ ...config, max_tokens: parseInt(e.target.value) })}
                        className="w-full mt-2 accent-primary-500"
                    />
                    <div className="flex justify-between text-xs text-surface-400 mt-1">
                        <span>64</span><span>512</span><span>2048</span>
                    </div>
                </div>

                {/* Data Split */}
                <div>
                    <label className="text-sm font-semibold text-surface-700 flex items-center">
                        Data Split: {Math.round(config.train_split * 100)}% train / {Math.round((1 - config.train_split) * 100)}% validation
                        <Tip text="How much data is used for training vs. checking quality. 90/10 is standard." />
                    </label>
                    <input
                        type="range" min={50} max={99} value={config.train_split * 100}
                        onChange={(e) => setConfig({ ...config, train_split: parseInt(e.target.value) / 100 })}
                        className="w-full mt-2 accent-primary-500"
                    />
                </div>

                {/* Advanced Toggle */}
                <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex items-center gap-2 text-sm text-surface-500 hover:text-surface-700 transition-colors"
                >
                    {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
                </button>

                {showAdvanced && (
                    <div className="space-y-4 p-4 bg-surface-50 rounded-xl border border-surface-200 animate-fade-in">
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="text-xs font-medium text-surface-600">Batch Size</label>
                                <input type="number" min={1} max={64} value={config.batch_size}
                                    onChange={(e) => setConfig({ ...config, batch_size: Math.max(1, parseInt(e.target.value) || 4) })}
                                    className="w-full mt-1 px-3 py-2 border border-surface-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30"
                                />
                            </div>
                            <div>
                                <label className="text-xs font-medium text-surface-600">Gradient Accumulation</label>
                                <input type="number" min={1} max={32} value={config.gradient_accumulation_steps}
                                    onChange={(e) => setConfig({ ...config, gradient_accumulation_steps: Math.max(1, parseInt(e.target.value) || 4) })}
                                    className="w-full mt-1 px-3 py-2 border border-surface-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30"
                                />
                            </div>
                            <div>
                                <label className="text-xs font-medium text-surface-600">LoRA Rank</label>
                                <input type="number" min={4} max={128} value={config.lora_rank}
                                    onChange={(e) => setConfig({ ...config, lora_rank: Math.max(4, parseInt(e.target.value) || 16) })}
                                    className="w-full mt-1 px-3 py-2 border border-surface-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30"
                                />
                            </div>
                            <div>
                                <label className="text-xs font-medium text-surface-600">LoRA Alpha</label>
                                <input type="number" min={4} max={256} value={config.lora_alpha}
                                    onChange={(e) => setConfig({ ...config, lora_alpha: Math.max(4, parseInt(e.target.value) || 32) })}
                                    className="w-full mt-1 px-3 py-2 border border-surface-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30"
                                />
                            </div>
                            <div>
                                <label className="text-xs font-medium text-surface-600">LoRA Dropout</label>
                                <input type="number" min={0} max={0.5} step={0.01} value={config.lora_dropout}
                                    onChange={(e) => setConfig({ ...config, lora_dropout: Math.min(0.5, Math.max(0, parseFloat(e.target.value) || 0.05)) })}
                                    className="w-full mt-1 px-3 py-2 border border-surface-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30"
                                />
                            </div>
                            <div>
                                <label className="text-xs font-medium text-surface-600">Warmup Steps</label>
                                <input type="number" min={0} max={1000} value={config.warmup_steps}
                                    onChange={(e) => setConfig({ ...config, warmup_steps: Math.max(0, parseInt(e.target.value) || 10) })}
                                    className="w-full mt-1 px-3 py-2 border border-surface-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30"
                                />
                            </div>
                        </div>

                        {/* Best-practice training settings */}
                        <div className="border-t border-surface-200 pt-4 mt-4">
                            <p className="text-xs font-semibold text-surface-500 uppercase tracking-wide mb-3">Optimization Settings</p>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="text-xs font-medium text-surface-600">Weight Decay</label>
                                    <input type="number" min={0} max={1} step={0.005} value={config.weight_decay}
                                        onChange={(e) => setConfig({ ...config, weight_decay: Math.min(1, Math.max(0, parseFloat(e.target.value) || 0.01)) })}
                                        className="w-full mt-1 px-3 py-2 border border-surface-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30"
                                    />
                                </div>
                                <div>
                                    <label className="text-xs font-medium text-surface-600">LR Scheduler</label>
                                    <select value={config.lr_scheduler_type}
                                        onChange={(e) => setConfig({ ...config, lr_scheduler_type: e.target.value })}
                                        className="w-full mt-1 px-3 py-2 border border-surface-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30 bg-white"
                                    >
                                        <option value="cosine">Cosine (recommended)</option>
                                        <option value="linear">Linear</option>
                                        <option value="constant">Constant</option>
                                        <option value="cosine_with_restarts">Cosine with Restarts</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="text-xs font-medium text-surface-600">Early Stopping Patience</label>
                                    <input type="number" min={0} max={20} value={config.early_stopping_patience}
                                        onChange={(e) => setConfig({ ...config, early_stopping_patience: Math.max(0, parseInt(e.target.value) || 3) })}
                                        className="w-full mt-1 px-3 py-2 border border-surface-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30"
                                    />
                                    <p className="text-xs text-surface-400 mt-0.5">0 = disabled, 3 = stop after 3 evals without improvement</p>
                                </div>
                                <div>
                                    <label className="text-xs font-medium text-surface-600">Eval Steps</label>
                                    <input type="number" min={10} max={1000} value={config.eval_steps}
                                        onChange={(e) => setConfig({ ...config, eval_steps: Math.max(10, parseInt(e.target.value) || 50) })}
                                        className="w-full mt-1 px-3 py-2 border border-surface-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30"
                                    />
                                </div>
                            </div>
                            <div className="flex items-center gap-3 mt-3">
                                <label className="flex items-center gap-2 cursor-pointer">
                                    <input type="checkbox" checked={config.gradient_checkpointing}
                                        onChange={(e) => setConfig({ ...config, gradient_checkpointing: e.target.checked })}
                                        className="rounded border-surface-300 text-primary-500 focus:ring-primary-500"
                                    />
                                    <span className="text-xs font-medium text-surface-600">Gradient Checkpointing</span>
                                    <span className="text-xs text-surface-400">(saves ~40% VRAM)</span>
                                </label>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Status Summary */}
            <div className="bg-white rounded-2xl border border-surface-100 p-5">
                <h3 className="text-sm font-semibold text-surface-700 mb-3">Ready to Train?</h3>
                <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm">
                        {readyDatasets.length > 0 ? (
                            <CheckCircle className="w-4 h-4 text-emerald-500" />
                        ) : (
                            <AlertCircle className="w-4 h-4 text-danger-500" />
                        )}
                        <span className={readyDatasets.length > 0 ? 'text-surface-700' : 'text-danger-600'}>
                            {readyDatasets.length} dataset(s) ready
                        </span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                        {selectedModel ? (
                            <CheckCircle className="w-4 h-4 text-emerald-500" />
                        ) : (
                            <AlertCircle className="w-4 h-4 text-danger-500" />
                        )}
                        <span className={selectedModel ? 'text-surface-700' : 'text-danger-600'}>
                            {selectedModel ? `Model: ${selectedModel.name}` : 'No model selected'}
                        </span>
                    </div>
                    {selectedModel && !selectedModel.is_cached && (
                        <div className="flex items-center gap-2 text-sm">
                            <AlertCircle className="w-4 h-4 text-amber-500" />
                            <span className="text-amber-600">
                                Model not downloaded yet — it will be downloaded automatically when training starts, or you can pre-download it from the Model tab.
                            </span>
                        </div>
                    )}
                    {selectedModel && selectedModel.is_cached && (
                        <div className="flex items-center gap-2 text-sm">
                            <CheckCircle className="w-4 h-4 text-emerald-500" />
                            <span className="text-surface-700">Model downloaded & cached</span>
                        </div>
                    )}
                </div>
            </div>

            {/* Multi-GPU & Resume Options */}
            <div className="bg-white rounded-2xl border border-surface-100 p-5 space-y-3">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Cpu className="w-4 h-4 text-blue-500" />
                        <span className="text-sm font-medium text-surface-700">Multi-GPU (DeepSpeed)</span>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" checked={config.multi_gpu}
                            onChange={(e) => setConfig({ ...config, multi_gpu: e.target.checked })}
                            className="sr-only peer" />
                        <div className="w-9 h-5 bg-surface-200 peer-focus:ring-2 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-surface-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-500"></div>
                    </label>
                </div>
                {config.multi_gpu && (
                    <div className="flex gap-2 ml-6">
                        {[2, 3].map(stage => (
                            <button key={stage}
                                onClick={() => setConfig({ ...config, deepspeed_stage: stage })}
                                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${config.deepspeed_stage === stage
                                    ? 'bg-blue-100 text-blue-700 border border-blue-200'
                                    : 'bg-surface-50 text-surface-500 border border-surface-200'
                                    }`}
                            >
                                ZeRO-{stage} {stage === 2 ? '(Recommended)' : '(Large Models)'}
                            </button>
                        ))}
                    </div>
                )}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <RotateCcw className="w-4 h-4 text-emerald-500" />
                        <span className="text-sm font-medium text-surface-700">Resume from Checkpoint</span>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" checked={config.resume_from_checkpoint}
                            onChange={(e) => setConfig({ ...config, resume_from_checkpoint: e.target.checked })}
                            className="sr-only peer" />
                        <div className="w-9 h-5 bg-surface-200 peer-focus:ring-2 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-surface-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-emerald-500"></div>
                    </label>
                </div>
            </div>

            {/* Start Button */}
            <button
                onClick={onStartTraining}
                disabled={!selectedModel || readyDatasets.length === 0 || actionPending === 'starting'}
                aria-busy={actionPending === 'starting'}
                className="w-full py-3 bg-gradient-to-r from-primary-600 to-primary-500 text-white rounded-2xl font-semibold text-sm shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 focus:outline-none focus:ring-2 focus:ring-primary-500/50"
            >
                {actionPending === 'starting' ? (
                    <><Loader className="w-5 h-5 animate-spin" aria-hidden="true" /> Starting…</>
                ) : (
                    <><Zap className="w-5 h-5" aria-hidden="true" /> {config.resume_from_checkpoint ? 'Resume Training' : 'Start Training'}</>
                )}
            </button>

            {/* Compare Runs */}
            <button
                onClick={async () => {
                    try {
                        const res = await trainingAPI.history(projectId);
                        setRunHistory(res.data.runs || []);
                        setShowComparison(!showComparison);
                    } catch (e) { toast.error('Failed to load run history'); }
                }}
                className="w-full py-2.5 bg-surface-50 border border-surface-200 text-surface-600 rounded-2xl text-sm font-medium hover:bg-surface-100 transition-all flex items-center justify-center gap-2"
            >
                <GitCompare className="w-4 h-4" /> Compare Training Runs
            </button>

            {showComparison && (
                <RunComparisonPanel
                    projectId={projectId}
                    runs={runHistory}
                    onClose={() => setShowComparison(false)}
                />
            )}
        </div>
    );
}
