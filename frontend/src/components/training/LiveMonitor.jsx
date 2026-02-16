/**
 * LiveMonitor — renders the training-active view with progress,
 * hardware gauges, and loss chart.
 */
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import {
    Play, Pause, Square, Loader, AlertCircle, Clock,
    TrendingDown, Cpu, Gauge, Thermometer, Activity, Timer, Flame,
} from 'lucide-react';
import { CircleGauge, BarGauge, formatTime } from './Gauges';

export default function LiveMonitor({
    status, hardware, lossHistory, selectedModel,
    wsError, actionPending,
    onPause, onResume, onStop,
}) {
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
                        <button onClick={onResume} disabled={!!actionPending} aria-busy={actionPending === 'resuming'} className="flex items-center gap-2 px-4 py-2 bg-primary-500 text-white rounded-xl text-sm font-medium hover:bg-primary-600 transition-colors disabled:opacity-60 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-primary-500/50">
                            {actionPending === 'resuming' ? <Loader className="w-4 h-4 animate-spin" aria-hidden="true" /> : <Play className="w-4 h-4" aria-hidden="true" />} {actionPending === 'resuming' ? 'Resuming…' : 'Resume'}
                        </button>
                    ) : (
                        <button onClick={onPause} disabled={!!actionPending} aria-busy={actionPending === 'pausing'} className="flex items-center gap-2 px-4 py-2 bg-amber-500 text-white rounded-xl text-sm font-medium hover:bg-amber-600 transition-colors disabled:opacity-60 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-amber-500/50">
                            {actionPending === 'pausing' ? <Loader className="w-4 h-4 animate-spin" aria-hidden="true" /> : <Pause className="w-4 h-4" aria-hidden="true" />} {actionPending === 'pausing' ? 'Pausing…' : 'Pause'}
                        </button>
                    )}
                    <button onClick={onStop} disabled={!!actionPending} className="flex items-center gap-2 px-4 py-2 bg-danger-500 text-white rounded-xl text-sm font-medium hover:bg-danger-600 transition-colors disabled:opacity-60 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-danger-500/50">
                        {actionPending === 'stopping' ? <Loader className="w-4 h-4 animate-spin" aria-hidden="true" /> : <Square className="w-4 h-4" aria-hidden="true" />} {actionPending === 'stopping' ? 'Stopping…' : 'Stop'}
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
                <div className="grid grid-cols-3 md:grid-cols-6 gap-4 mt-4">
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
                        <p className="text-lg font-bold text-amber-600">{status?.validation_loss?.toFixed(4) || '—'}</p>
                        <p className="text-xs text-surface-500">Eval Loss</p>
                    </div>
                    <div className="text-center">
                        <p className="text-lg font-bold text-violet-600">{status?.perplexity?.toFixed(1) || '—'}</p>
                        <p className="text-xs text-surface-500">Perplexity</p>
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
                    <span className="text-xs text-surface-400 font-normal ml-auto">Refreshing via WebSocket</span>
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
