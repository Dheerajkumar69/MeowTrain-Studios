/**
 * SettingsPanel — side panel with LM Studio, system prompt,
 * temperature, max tokens, streaming toggle, context, and templates.
 */
import { useState } from 'react';
import {
    Thermometer, Save, BookOpen, FileText,
    Server, Wifi, WifiOff, RefreshCw, Zap,
    Loader, Radio,
} from 'lucide-react';
import { inferenceAPI, lmstudioAPI } from '../../services/api';

export default function SettingsPanel({
    projectId, systemPrompt, setSystemPrompt,
    temperature, setTemperature,
    maxTokens, setMaxTokens,
    useStreaming, setUseStreaming,
    readyDatasets, selectedContextIds, toggleContext,
    showContext, showTemplates, showLMStudio,
    lmsConfig, setLmsConfig,
    lmsModels, setLmsModels,
    selectedLmsModel, setSelectedLmsModel,
    lmsConnected, setLmsConnected,
    input,
}) {
    const [lmsTesting, setLmsTesting] = useState(false);
    const [lmsError, setLmsError] = useState('');
    const [templates, setTemplates] = useState([]);
    const [templateName, setTemplateName] = useState('');
    const [templatesLoaded, setTemplatesLoaded] = useState(false);

    // Lazy load templates the first time panel shows
    if (showTemplates && !templatesLoaded) {
        setTemplatesLoaded(true);
        inferenceAPI.listPrompts(projectId).then((res) => setTemplates(res.data)).catch(() => {});
    }

    const loadLmsModels = async () => {
        try {
            const res = await lmstudioAPI.listModels();
            if (res.data.models) {
                setLmsModels(res.data.models);
                setLmsConnected(true);
                if (res.data.models.length > 0 && !selectedLmsModel) {
                    setSelectedLmsModel(res.data.models[0].model_id);
                }
            }
        } catch {
            setLmsConnected(false);
            setLmsModels([]);
        }
    };

    const testLmsConnection = async () => {
        setLmsTesting(true);
        setLmsError('');
        try {
            await lmstudioAPI.setConfig(lmsConfig);
            const res = await lmstudioAPI.testConnection();
            if (res.data.connected) {
                setLmsConnected(true);
                setLmsModels(res.data.models || []);
                if (res.data.models?.length > 0) {
                    setSelectedLmsModel(res.data.models[0].model_id);
                }
                const updated = { ...lmsConfig, enabled: true };
                setLmsConfig(updated);
                await lmstudioAPI.setConfig({ enabled: true });
            } else {
                setLmsConnected(false);
                setLmsError(res.data.error || 'Connection failed');
            }
        } catch (err) {
            setLmsConnected(false);
            setLmsError(err.response?.data?.detail || 'Connection test failed');
        } finally {
            setLmsTesting(false);
        }
    };

    const disconnectLms = async () => {
        setLmsConfig({ ...lmsConfig, enabled: false });
        setLmsConnected(false);
        setLmsModels([]);
        setSelectedLmsModel(null);
        await lmstudioAPI.setConfig({ enabled: false });
    };

    const saveTemplate = async () => {
        if (!templateName.trim()) return;
        try {
            const res = await inferenceAPI.savePrompt(projectId, {
                name: templateName,
                system_prompt: systemPrompt,
                user_prompt: input,
                temperature,
            });
            setTemplates((prev) => [...prev, res.data]);
            setTemplateName('');
        } catch { /* silently ignore */ }
    };

    const loadTemplate = (t) => {
        setSystemPrompt(t.system_prompt);
        setTemperature(t.temperature);
    };

    const tempLabel = temperature < 0.3 ? 'Precise' : temperature < 0.8 ? 'Balanced' : temperature < 1.3 ? 'Creative' : 'Wild';

    return (
        <div className="w-full lg:w-72 space-y-4 shrink-0 overflow-auto" role="complementary" aria-label="Chat settings">
            {/* LM Studio Panel */}
            {showLMStudio && (
                <div className="bg-white rounded-2xl border border-violet-200 p-4 shadow-sm">
                    <div className="flex items-center justify-between mb-3">
                        <label className="text-xs font-semibold text-violet-700 flex items-center gap-1.5">
                            <Server className="w-3.5 h-3.5" /> LM Studio
                        </label>
                        {lmsConnected ? (
                            <span className="text-xs flex items-center gap-1 text-emerald-600">
                                <Wifi className="w-3 h-3" /> Connected
                            </span>
                        ) : (
                            <span className="text-xs flex items-center gap-1 text-surface-400">
                                <WifiOff className="w-3 h-3" /> Disconnected
                            </span>
                        )}
                    </div>

                    {!lmsConnected ? (
                        <div className="space-y-3">
                            <div>
                                <label className="text-xs text-surface-600 block mb-1">Host</label>
                                <input
                                    type="text"
                                    value={lmsConfig.host}
                                    onChange={(e) => setLmsConfig({ ...lmsConfig, host: e.target.value })}
                                    placeholder="http://localhost"
                                    className="w-full px-3 py-2 border border-surface-200 rounded-lg text-xs focus:outline-none focus:ring-2 focus:ring-violet-500/30"
                                />
                            </div>
                            <div>
                                <label className="text-xs text-surface-600 block mb-1">Port</label>
                                <input
                                    type="number"
                                    value={lmsConfig.port}
                                    onChange={(e) => setLmsConfig({ ...lmsConfig, port: parseInt(e.target.value) || 1234 })}
                                    className="w-full px-3 py-2 border border-surface-200 rounded-lg text-xs focus:outline-none focus:ring-2 focus:ring-violet-500/30"
                                />
                            </div>
                            {lmsError && (
                                <p className="text-xs text-danger-600 bg-danger-400/10 px-2 py-1.5 rounded-lg">{lmsError}</p>
                            )}
                            <button
                                onClick={testLmsConnection}
                                disabled={lmsTesting}
                                className="w-full py-2 bg-violet-500 text-white rounded-xl text-xs font-medium hover:bg-violet-600 transition-colors disabled:opacity-50 flex items-center justify-center gap-1.5"
                            >
                                {lmsTesting ? (
                                    <><Loader className="w-3 h-3 animate-spin" /> Testing...</>
                                ) : (
                                    <><Wifi className="w-3 h-3" /> Connect</>
                                )}
                            </button>
                            <p className="text-xs text-surface-400 leading-relaxed">
                                Make sure LM Studio is running with the local server enabled (default port: 1234).
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            <div>
                                <label className="text-xs text-surface-600 block mb-1">Active Model</label>
                                <select
                                    value={selectedLmsModel || ''}
                                    onChange={(e) => setSelectedLmsModel(e.target.value)}
                                    className="w-full px-3 py-2 border border-surface-200 rounded-lg text-xs bg-white focus:outline-none focus:ring-2 focus:ring-violet-500/30"
                                >
                                    {lmsModels.map((m) => (
                                        <option key={m.model_id} value={m.model_id}>
                                            {m.name || m.model_id}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <div className="space-y-1.5">
                                {lmsModels.map((m) => (
                                    <div
                                        key={m.model_id}
                                        onClick={() => setSelectedLmsModel(m.model_id)}
                                        className={`flex items-center gap-2 px-2.5 py-2 rounded-lg text-xs cursor-pointer transition-colors ${selectedLmsModel === m.model_id
                                            ? 'bg-violet-50 border border-violet-200 text-violet-700'
                                            : 'bg-surface-50 border border-surface-100 text-surface-700 hover:bg-violet-50'
                                            }`}
                                    >
                                        <Zap className={`w-3 h-3 ${selectedLmsModel === m.model_id ? 'text-violet-500' : 'text-surface-400'}`} />
                                        <span className="truncate font-medium">{m.name || m.model_id}</span>
                                    </div>
                                ))}
                            </div>

                            <div className="flex gap-2">
                                <button
                                    onClick={loadLmsModels}
                                    className="flex-1 py-1.5 text-xs text-violet-600 bg-violet-50 rounded-lg hover:bg-violet-100 transition-colors flex items-center justify-center gap-1"
                                >
                                    <RefreshCw className="w-3 h-3" /> Refresh
                                </button>
                                <button
                                    onClick={disconnectLms}
                                    className="flex-1 py-1.5 text-xs text-danger-600 bg-danger-400/10 rounded-lg hover:bg-danger-400/20 transition-colors"
                                >
                                    Disconnect
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* System Prompt */}
            <div className="bg-white rounded-2xl border border-surface-100 p-4">
                <label className="text-xs font-semibold text-surface-600 block mb-2">System Prompt</label>
                <textarea
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    rows={3}
                    maxLength={2000}
                    className="w-full px-3 py-2 border border-surface-200 rounded-xl text-xs focus:outline-none focus:ring-2 focus:ring-primary-500/30 resize-none"
                />
            </div>

            {/* Temperature */}
            <div className="bg-white rounded-2xl border border-surface-100 p-4">
                <label className="text-xs font-semibold text-surface-600 flex items-center gap-2 mb-2">
                    <Thermometer className="w-3.5 h-3.5" /> Temperature: {temperature.toFixed(1)} ({tempLabel})
                </label>
                <input
                    type="range" min={0} max={2} step={0.1} value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="w-full accent-primary-500"
                />
                <div className="flex justify-between text-xs text-surface-400 mt-1">
                    <span>Precise</span><span>Creative</span>
                </div>
            </div>

            {/* Max Tokens */}
            <div className="bg-white rounded-2xl border border-surface-100 p-4">
                <label className="text-xs font-semibold text-surface-600 mb-2 block">Max Response Tokens: {maxTokens}</label>
                <input
                    type="range" min={64} max={2048} step={64} value={maxTokens}
                    onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                    className="w-full accent-primary-500"
                    aria-label={`Max response tokens: ${maxTokens}`}
                />
            </div>

            {/* Streaming Toggle */}
            <div className="bg-white rounded-2xl border border-surface-100 p-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Radio className="w-3.5 h-3.5 text-primary-500" aria-hidden="true" />
                        <span className="text-xs font-semibold text-surface-600">Stream Response</span>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input
                            type="checkbox"
                            checked={useStreaming}
                            onChange={(e) => setUseStreaming(e.target.checked)}
                            className="sr-only peer"
                            aria-label="Toggle streaming responses"
                        />
                        <div className="w-9 h-5 bg-surface-200 peer-focus:ring-2 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-surface-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-primary-500" />
                    </label>
                </div>
                <p className="text-xs text-surface-400 mt-1.5">See tokens as they're generated</p>
            </div>

            {/* Context Panel */}
            {showContext && readyDatasets.length > 0 && (
                <div className="bg-white rounded-2xl border border-surface-100 p-4">
                    <label className="text-xs font-semibold text-surface-600 mb-2 block">Context Documents</label>
                    <div className="space-y-2">
                        {readyDatasets.map((ds) => (
                            <label key={ds.id} className="flex items-center gap-2 text-xs cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={selectedContextIds.includes(ds.id)}
                                    onChange={() => toggleContext(ds.id)}
                                    className="accent-primary-500 rounded"
                                />
                                <FileText className="w-3 h-3 text-surface-400" />
                                <span className="text-surface-700 truncate">{ds.original_name}</span>
                            </label>
                        ))}
                    </div>
                </div>
            )}

            {/* Templates */}
            {showTemplates && (
                <div className="bg-white rounded-2xl border border-surface-100 p-4">
                    <label className="text-xs font-semibold text-surface-600 mb-2 block">Prompt Templates</label>
                    {templates.length > 0 && (
                        <div className="space-y-2 mb-3">
                            {templates.map((t) => (
                                <button
                                    key={t.id}
                                    onClick={() => loadTemplate(t)}
                                    className="w-full text-left px-3 py-2 rounded-lg text-xs bg-surface-50 hover:bg-primary-50 border border-surface-100 transition-colors"
                                >
                                    <p className="font-medium text-surface-700">{t.name}</p>
                                </button>
                            ))}
                        </div>
                    )}
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={templateName}
                            onChange={(e) => setTemplateName(e.target.value)}
                            placeholder="Template name"
                            maxLength={100}
                            className="flex-1 px-2 py-1.5 border border-surface-200 rounded-lg text-xs"
                        />
                        <button
                            onClick={saveTemplate}
                            disabled={!templateName.trim()}
                            className="px-2 py-1.5 bg-primary-500 text-white rounded-lg text-xs disabled:opacity-50"
                        >
                            <Save className="w-3 h-3" />
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
