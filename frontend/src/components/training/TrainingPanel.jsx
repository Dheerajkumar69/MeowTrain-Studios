import { useState, useEffect, useRef, useCallback } from 'react';
import { trainingAPI, hardwareAPI } from '../../services/api';
import { createWebSocket } from '../../services/ws';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import {
    Play, Pause, Square, Zap, Info, ChevronDown, ChevronUp,
    Loader, CheckCircle, AlertCircle, Clock, TrendingDown,
    Cpu, MemoryStick, Thermometer, Activity, Gauge, Timer, Flame
} from 'lucide-react';

const METHOD_INFO = {
    lora: { label: 'LoRA', desc: 'Adds small trainable layers on top of the model. Fast, memory-efficient, and works on most hardware.' },
    qlora: { label: 'QLoRA', desc: 'Like LoRA but the model is compressed to use less memory. Best for GPUs with less than 6GB.' },
    full: { label: 'Full Fine-tune', desc: 'Trains all parameters. Best quality but needs a lot of RAM/VRAM. For advanced users.' },
};

const LR_PRESETS = {
    conservative: { value: 1e-4, label: 'Conservative', desc: 'Slower but safer learning' },
    balanced: { value: 2e-4, label: 'Balanced', desc: 'Good default for most tasks' },
    aggressive: { value: 5e-4, label: 'Aggressive', desc: 'Faster learning, risk of instability' },
};

function formatTime(seconds) {
    if (seconds == null || seconds < 0) return '--:--';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

// ── Gauge Component ──────────────────────────────────────────────
function CircleGauge({ value, max, label, unit, color, icon: Icon, sublabel }) {
    const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
    const radius = 36;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (pct / 100) * circumference;

    return (
        <div className="flex flex-col items-center">
            <div className="relative w-24 h-24">
                <svg className="w-24 h-24 -rotate-90" viewBox="0 0 80 80">
                    <circle cx="40" cy="40" r={radius} fill="none" stroke="#f1f3f5" strokeWidth="6" />
                    <circle
                        cx="40" cy="40" r={radius} fill="none"
                        stroke={color} strokeWidth="6" strokeLinecap="round"
                        strokeDasharray={circumference} strokeDashoffset={offset}
                        className="transition-all duration-500 ease-out"
                    />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-lg font-bold text-surface-900">{value != null ? Math.round(value) : '—'}</span>
                    <span className="text-xs text-surface-400">{unit}</span>
                </div>
            </div>
            <div className="flex items-center gap-1 mt-1.5">
                {Icon && <Icon className="w-3 h-3" style={{ color }} />}
                <span className="text-xs font-medium text-surface-600">{label}</span>
            </div>
            {sublabel && <span className="text-xs text-surface-400">{sublabel}</span>}
        </div>
    );
}

// ── Bar Gauge ────────────────────────────────────────────────────
function BarGauge({ value, max, label, color, format }) {
    const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
    return (
        <div className="space-y-1">
            <div className="flex justify-between text-xs">
                <span className="text-surface-600 font-medium">{label}</span>
                <span className="text-surface-500">{format || `${value?.toFixed?.(1) ?? '—'} / ${max?.toFixed?.(1) ?? '—'} GB`}</span>
            </div>
            <div className="h-2.5 bg-surface-100 rounded-full overflow-hidden">
                <div
                    className="h-full rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${pct}%`, backgroundColor: color }}
                />
            </div>
        </div>
    );
}

export default function TrainingPanel({ projectId, selectedModel, datasets, project, onProjectUpdate }) {
    const [config, setConfig] = useState({
        method: 'lora',
        epochs: 3,
        batch_size: 4,
        learning_rate: 2e-4,
        max_tokens: 512,
        train_split: 0.9,
        lora_rank: 16,
        lora_alpha: 32,
        lora_dropout: 0.05,
        warmup_steps: 10,
        gradient_accumulation_steps: 4,
    });
    const [lrPreset, setLrPreset] = useState('balanced');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [training, setTraining] = useState(false);
    const [status, setStatus] = useState(null);
    const [hardware, setHardware] = useState(null);
    const [lossHistory, setLossHistory] = useState([]);
    const [error, setError] = useState('');
    const [wsError, setWsError] = useState(false);
    const wsRef = useRef(null);
    const hwPollRef = useRef(null);

    const readyDatasets = datasets?.filter((d) => d.status === 'ready') || [];

    // Poll hardware even when not training (for the idle view)
    useEffect(() => {
        const poll = async () => {
            try {
                const res = await hardwareAPI.status();
                setHardware(res.data);
            } catch { /* silent */ }
        };
        poll();
        hwPollRef.current = setInterval(poll, 500);
        return () => clearInterval(hwPollRef.current);
    }, []);

    // Connect WebSocket when training
    useEffect(() => {
        if (!training) return;
        // Stop REST polling when WS is active (WS sends hardware data)
        clearInterval(hwPollRef.current);

        const wsHandle = createWebSocket(
            projectId,
            (data) => {
                setWsError(false);
                setStatus(data);
                if (data.hardware) setHardware(data.hardware);
                if (data.current_loss != null) {
                    setLossHistory((prev) => [...prev.slice(-199), { step: data.current_step, loss: data.current_loss }]);
                }
                if (data.status === 'completed' || data.status === 'error' || data.status === 'stopped') {
                    setTraining(false);
                    onProjectUpdate?.();
                }
            },
            (err) => {
                console.error('WS error:', err);
                setWsError(true);
            },
            () => console.log('WS closed')
        );
        wsRef.current = wsHandle;

        return () => {
            wsHandle.close();
            // Resume REST polling
            const poll = async () => {
                try {
                    const res = await hardwareAPI.status();
                    setHardware(res.data);
                } catch { /* silent */ }
            };
            hwPollRef.current = setInterval(poll, 500);
        };
    }, [training, projectId]);

    const startTraining = async () => {
        if (!selectedModel) { setError('Please select a model first (Model tab).'); return; }
        if (readyDatasets.length === 0) { setError('Please upload training data first (Data tab).'); return; }
        setError('');
        try {
            await trainingAPI.configure(projectId, { ...config, base_model: selectedModel.model_id });
            await trainingAPI.start(projectId);
            setTraining(true);
            setLossHistory([]);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to start training');
        }
    };

    const pauseTraining = async () => { try { await trainingAPI.pause(projectId); } catch (err) { setError(err.response?.data?.detail || 'Pause failed'); } };
    const resumeTraining = async () => { try { await trainingAPI.resume(projectId); } catch (err) { setError(err.response?.data?.detail || 'Resume failed'); } };
    const stopTraining = async () => {
        if (!confirm('Stop training? Progress will be saved as a checkpoint.')) return;
        try { await trainingAPI.stop(projectId); setTraining(false); } catch (err) { setError(err.response?.data?.detail || 'Stop failed'); }
    };

    const Tip = ({ text }) => (
        <div className="group relative inline-block ml-1">
            <Info className="w-3.5 h-3.5 text-surface-400 cursor-help" />
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-surface-800 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none w-48 text-center z-10">
                {text}
                <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-surface-800" />
            </div>
        </div>
    );

    // ─── TRAINING ACTIVE VIEW ────────────────────────────────────
    if (training || status?.status === 'running' || status?.status === 'paused') {
        const progress = status?.total_steps > 0
            ? ((status.current_step / status.total_steps) * 100)
            : 0;
        const hw = hardware || {};
        const gpuAvail = hw.gpu_available;

        return (
            <div className="space-y-4 animate-fade-in">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-xl font-bold text-surface-900 flex items-center gap-2">
                            {status?.status === 'paused' ? (
                                <><Pause className="w-5 h-5 text-amber-500" /> Training Paused</>
                            ) : (
                                <><Loader className="w-5 h-5 text-primary-500 animate-spin" /> Training in Progress</>
                            )}
                        </h2>
                        <p className="text-sm text-surface-500 mt-1">
                            {selectedModel?.name} • Epoch {status?.current_epoch || 0} / {status?.total_epochs || 0}
                        </p>
                    </div>
                    <div className="flex gap-2">
                        {status?.status === 'paused' ? (
                            <button onClick={resumeTraining} className="flex items-center gap-2 px-4 py-2 bg-primary-500 text-white rounded-xl text-sm font-medium hover:bg-primary-600 transition-colors">
                                <Play className="w-4 h-4" /> Resume
                            </button>
                        ) : (
                            <button onClick={pauseTraining} className="flex items-center gap-2 px-4 py-2 bg-amber-500 text-white rounded-xl text-sm font-medium hover:bg-amber-600 transition-colors">
                                <Pause className="w-4 h-4" /> Pause
                            </button>
                        )}
                        <button onClick={stopTraining} className="flex items-center gap-2 px-4 py-2 bg-danger-500 text-white rounded-xl text-sm font-medium hover:bg-danger-600 transition-colors">
                            <Square className="w-4 h-4" /> Stop
                        </button>
                    </div>
                </div>

                {wsError && (
                    <div className="bg-amber-50 border border-amber-200 text-amber-700 text-xs rounded-xl px-3 py-2 flex items-center gap-2">
                        <AlertCircle className="w-3.5 h-3.5" /> WebSocket reconnecting... Stats may be delayed.
                    </div>
                )}

                {/* Progress Bar */}
                <div className="bg-white rounded-2xl border border-surface-100 p-5">
                    <div className="flex justify-between text-sm mb-2">
                        <span className="text-surface-600 font-medium">Overall Progress</span>
                        <span className="font-bold text-primary-600">{progress.toFixed(1)}%</span>
                    </div>
                    <div className="h-4 bg-surface-100 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 rounded-full transition-all duration-500 relative overflow-hidden"
                            style={{ width: `${Math.max(progress, 0.5)}%` }}
                        >
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
                        </div>
                    </div>
                    <div className="grid grid-cols-4 gap-4 mt-4">
                        <div className="text-center">
                            <p className="text-lg font-bold text-surface-900">{status?.current_step || 0}</p>
                            <p className="text-xs text-surface-500">Step / {status?.total_steps || '—'}</p>
                        </div>
                        <div className="text-center">
                            <p className="text-lg font-bold text-surface-900">{status?.current_loss?.toFixed(4) || '—'}</p>
                            <p className="text-xs text-surface-500">Current Loss</p>
                        </div>
                        <div className="text-center">
                            <p className="text-lg font-bold text-emerald-600">{status?.best_loss?.toFixed(4) || '—'}</p>
                            <p className="text-xs text-surface-500">Best Loss</p>
                        </div>
                        <div className="text-center">
                            <p className="text-lg font-bold text-primary-600">{status?.tokens_per_sec?.toFixed(1) || '0'}</p>
                            <p className="text-xs text-surface-500">Tokens/sec</p>
                        </div>
                    </div>
                    <div className="flex justify-between text-xs text-surface-400 mt-3 pt-3 border-t border-surface-100">
                        <span className="flex items-center gap-1"><Timer className="w-3 h-3" /> Elapsed: {formatTime(status?.elapsed_seconds)}</span>
                        <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> ETA: {formatTime(status?.eta_seconds)}</span>
                    </div>
                </div>

                {/* Hardware Gauges */}
                <div className="bg-white rounded-2xl border border-surface-100 p-5">
                    <h3 className="text-sm font-semibold text-surface-700 mb-4 flex items-center gap-2">
                        <Activity className="w-4 h-4 text-primary-500" /> Live System Monitor
                        <span className="text-xs text-surface-400 font-normal ml-auto">Refreshing every 0.5s</span>
                    </h3>

                    {/* Circle Gauges */}
                    <div className="flex justify-around mb-5">
                        <CircleGauge
                            value={hw.cpu_usage_percent} max={100}
                            label="CPU" unit="%" color="#10b981" icon={Cpu}
                            sublabel={`${hw.cpu_cores || '—'} cores`}
                        />
                        {gpuAvail && (
                            <CircleGauge
                                value={hw.gpu_usage_percent} max={100}
                                label="GPU" unit="%" color="#8b5cf6" icon={Gauge}
                                sublabel={hw.gpu_name?.replace('NVIDIA ', '') || ''}
                            />
                        )}
                        {gpuAvail && (
                            <CircleGauge
                                value={hw.gpu_temp_celsius} max={100}
                                label="Temp" unit="°C" color={hw.gpu_temp_celsius > 80 ? '#ef4444' : hw.gpu_temp_celsius > 65 ? '#f59e0b' : '#10b981'}
                                icon={Thermometer}
                            />
                        )}
                        {gpuAvail && (
                            <CircleGauge
                                value={hw.gpu_power_watts} max={hw.gpu_power_limit_watts || 100}
                                label="Power" unit="W" color="#f59e0b" icon={Flame}
                                sublabel={`/ ${hw.gpu_power_limit_watts || '—'} W`}
                            />
                        )}
                    </div>

                    {/* Bar Gauges */}
                    <div className="space-y-3">
                        <BarGauge
                            value={hw.ram_used_gb} max={hw.ram_total_gb}
                            label="RAM" color="#3b82f6"
                            format={`${hw.ram_used_gb?.toFixed?.(1) ?? '—'} / ${hw.ram_total_gb?.toFixed?.(1) ?? '—'} GB (${hw.ram_usage_percent?.toFixed?.(0) ?? '—'}%)`}
                        />
                        {gpuAvail && (
                            <BarGauge
                                value={hw.gpu_vram_used_gb} max={hw.gpu_vram_total_gb}
                                label="VRAM" color="#8b5cf6"
                                format={`${hw.gpu_vram_used_gb?.toFixed?.(1) ?? '—'} / ${hw.gpu_vram_total_gb?.toFixed?.(1) ?? '—'} GB`}
                            />
                        )}
                        <BarGauge
                            value={(hw.disk_total_gb || 0) - (hw.disk_free_gb || 0)} max={hw.disk_total_gb || 1}
                            label="Disk" color="#f59e0b"
                            format={`${hw.disk_free_gb?.toFixed?.(0) ?? '—'} GB free / ${hw.disk_total_gb?.toFixed?.(0) ?? '—'} GB`}
                        />
                    </div>
                </div>

                {/* Loss Chart */}
                {lossHistory.length > 1 && (
                    <div className="bg-white rounded-2xl border border-surface-100 p-5">
                        <h3 className="text-sm font-semibold text-surface-700 mb-4 flex items-center gap-2">
                            <TrendingDown className="w-4 h-4" /> Training Loss
                        </h3>
                        <ResponsiveContainer width="100%" height={200}>
                            <AreaChart data={lossHistory}>
                                <defs>
                                    <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#5c7cfa" stopOpacity={0.15} />
                                        <stop offset="95%" stopColor="#5c7cfa" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f3f5" />
                                <XAxis dataKey="step" tick={{ fontSize: 10 }} stroke="#adb5bd" />
                                <YAxis tick={{ fontSize: 10 }} stroke="#adb5bd" />
                                <Tooltip contentStyle={{ borderRadius: 10, border: '1px solid #e9ecef', fontSize: 12, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} />
                                <Area type="monotone" dataKey="loss" stroke="#5c7cfa" strokeWidth={2} fill="url(#lossGrad)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                )}
            </div>
        );
    }

    // ─── CONFIGURATION VIEW ──────────────────────────────────────
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
                        Training Method <Tip text="How the model is trained. LoRA is recommended for most users." />
                    </label>
                    <div className="grid grid-cols-3 gap-3 mt-3">
                        {Object.entries(METHOD_INFO).map(([key, info]) => (
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
                </div>
            </div>

            {/* Start Button */}
            <button
                onClick={startTraining}
                disabled={!selectedModel || readyDatasets.length === 0}
                className="w-full py-3 bg-gradient-to-r from-primary-600 to-primary-500 text-white rounded-2xl font-semibold text-sm shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
                <Zap className="w-5 h-5" /> Start Training
            </button>
        </div>
    );
}
