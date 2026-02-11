import { useState, useEffect, useRef } from 'react';
import { inferenceAPI, lmstudioAPI } from '../../services/api';
import {
    Send, Loader, Bot, User, Thermometer, Save,
    BookOpen, FileText, Trash2, Bookmark, Server,
    Wifi, WifiOff, RefreshCw, Zap, Settings
} from 'lucide-react';

export default function PlaygroundPanel({ projectId, project, datasets }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [systemPrompt, setSystemPrompt] = useState('You are a helpful assistant.');
    const [temperature, setTemperature] = useState(0.7);
    const [maxTokens, setMaxTokens] = useState(512);
    const [loading, setLoading] = useState(false);
    const [showContext, setShowContext] = useState(false);
    const [selectedContextIds, setSelectedContextIds] = useState([]);
    const [templates, setTemplates] = useState([]);
    const [showTemplates, setShowTemplates] = useState(false);
    const [templateName, setTemplateName] = useState('');
    const messagesEndRef = useRef(null);

    // LM Studio state
    const [showLMStudio, setShowLMStudio] = useState(false);
    const [lmsConfig, setLmsConfig] = useState({ host: 'http://localhost', port: 1234, enabled: false });
    const [lmsModels, setLmsModels] = useState([]);
    const [selectedLmsModel, setSelectedLmsModel] = useState(null);
    const [lmsConnected, setLmsConnected] = useState(false);
    const [lmsTesting, setLmsTesting] = useState(false);
    const [lmsError, setLmsError] = useState('');

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    useEffect(() => {
        inferenceAPI.listPrompts(projectId).then((res) => setTemplates(res.data)).catch(console.error);
        // Load LM Studio config
        lmstudioAPI.getConfig().then((res) => {
            setLmsConfig(res.data);
            if (res.data.enabled) {
                loadLmsModels();
            }
        }).catch(console.error);
    }, [projectId]);

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
            // Save config first
            await lmstudioAPI.setConfig(lmsConfig);
            const res = await lmstudioAPI.testConnection();
            if (res.data.connected) {
                setLmsConnected(true);
                setLmsModels(res.data.models || []);
                if (res.data.models?.length > 0) {
                    setSelectedLmsModel(res.data.models[0].model_id);
                }
                // Enable and save
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

    const sendMessage = async () => {
        if (!input.trim() || loading) return;
        const userMsg = { role: 'user', content: input };
        setMessages((prev) => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const payload = {
                prompt: input,
                system_prompt: systemPrompt,
                temperature,
                max_tokens: maxTokens,
                context_dataset_ids: selectedContextIds,
            };

            // Add LM Studio model if connected
            if (lmsConnected && selectedLmsModel && lmsConfig.enabled) {
                payload.lmstudio_model = selectedLmsModel;
            }

            const res = await inferenceAPI.chat(projectId, payload);
            setMessages((prev) => [
                ...prev,
                {
                    role: 'assistant',
                    content: res.data.response,
                    tokens: res.data.tokens_used,
                    time: res.data.generation_time_ms,
                    model: res.data.model_used,
                },
            ]);
        } catch (err) {
            setMessages((prev) => [
                ...prev,
                { role: 'assistant', content: 'Error: ' + (err.response?.data?.detail || err.message), isError: true },
            ]);
        } finally {
            setLoading(false);
        }
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
            setShowTemplates(false);
        } catch (err) {
            console.error('Failed to save template:', err);
        }
    };

    const loadTemplate = (t) => {
        setSystemPrompt(t.system_prompt);
        setInput(t.user_prompt);
        setTemperature(t.temperature);
        setShowTemplates(false);
    };

    const toggleContext = (id) => {
        setSelectedContextIds((prev) =>
            prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
        );
    };

    const tempLabel = temperature < 0.3 ? 'Precise' : temperature < 0.8 ? 'Balanced' : temperature < 1.3 ? 'Creative' : 'Wild';
    const readyDatasets = (datasets || []).filter((d) => d.status === 'ready');

    return (
        <div className="flex gap-4 h-[calc(100vh-12rem)]">
            {/* Chat Panel */}
            <div className="flex-1 flex flex-col bg-white rounded-2xl border border-surface-100 overflow-hidden">
                {/* Chat Header */}
                <div className="px-5 py-3 border-b border-surface-100 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Bot className="w-5 h-5 text-primary-500" />
                        <span className="font-semibold text-surface-900 text-sm">Playground</span>
                        {lmsConnected && selectedLmsModel && (
                            <span className="flex items-center gap-1 text-xs px-2 py-0.5 bg-emerald-50 text-emerald-700 rounded-full border border-emerald-200">
                                <Zap className="w-3 h-3" /> {selectedLmsModel.split('/').pop()}
                            </span>
                        )}
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setShowLMStudio(!showLMStudio)}
                            className={`flex items-center gap-1 text-xs px-2 py-1 rounded-lg transition-colors ${showLMStudio ? 'bg-violet-50 text-violet-600' :
                                lmsConnected ? 'text-emerald-600 hover:bg-emerald-50' :
                                    'text-surface-500 hover:text-violet-600 hover:bg-violet-50'
                                }`}
                        >
                            <Server className="w-3.5 h-3.5" /> LM Studio
                            {lmsConnected && <Wifi className="w-3 h-3 text-emerald-500" />}
                        </button>
                        <button
                            onClick={() => setShowTemplates(!showTemplates)}
                            className="flex items-center gap-1 text-xs text-surface-500 hover:text-primary-600 transition-colors px-2 py-1 rounded-lg hover:bg-primary-50"
                        >
                            <Bookmark className="w-3.5 h-3.5" /> Templates
                        </button>
                        <button
                            onClick={() => setShowContext(!showContext)}
                            className={`flex items-center gap-1 text-xs px-2 py-1 rounded-lg transition-colors ${showContext ? 'bg-primary-50 text-primary-600' : 'text-surface-500 hover:text-primary-600 hover:bg-primary-50'
                                }`}
                        >
                            <BookOpen className="w-3.5 h-3.5" /> Context
                        </button>
                        <button
                            onClick={() => setMessages([])}
                            className="flex items-center gap-1 text-xs text-surface-400 hover:text-danger-500 transition-colors px-2 py-1 rounded-lg hover:bg-danger-400/10"
                        >
                            <Trash2 className="w-3.5 h-3.5" /> Clear
                        </button>
                    </div>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-auto p-5 space-y-4">
                    {messages.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full text-center">
                            <div className="w-16 h-16 rounded-2xl bg-primary-50 flex items-center justify-center mb-4">
                                <Bot className="w-8 h-8 text-primary-400" />
                            </div>
                            <h3 className="text-lg font-semibold text-surface-700 mb-1">Test Your Model</h3>
                            <p className="text-sm text-surface-500 max-w-sm mb-4">
                                Start a conversation to test how your model responds.
                            </p>
                            {!lmsConnected && (
                                <button
                                    onClick={() => setShowLMStudio(true)}
                                    className="flex items-center gap-2 px-4 py-2 bg-violet-50 text-violet-700 rounded-xl text-sm border border-violet-200 hover:bg-violet-100 transition-colors"
                                >
                                    <Server className="w-4 h-4" /> Connect LM Studio for real responses
                                </button>
                            )}
                        </div>
                    )}
                    {messages.map((msg, i) => (
                        <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                            {msg.role === 'assistant' && (
                                <div className="w-8 h-8 rounded-lg bg-primary-100 flex items-center justify-center shrink-0">
                                    <Bot className="w-4 h-4 text-primary-600" />
                                </div>
                            )}
                            <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${msg.role === 'user'
                                ? 'bg-primary-500 text-white'
                                : msg.isError
                                    ? 'bg-danger-400/10 text-danger-600 border border-danger-400/30'
                                    : 'bg-surface-50 text-surface-800 border border-surface-100'
                                }`}>
                                <p className="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                                {(msg.tokens || msg.model) && (
                                    <p className={`text-xs mt-2 ${msg.role === 'user' ? 'text-white/60' : 'text-surface-400'}`}>
                                        {msg.model && <span className="font-medium">{msg.model.split('/').pop()}</span>}
                                        {msg.model && msg.tokens ? ' • ' : ''}
                                        {msg.tokens && `${msg.tokens} tokens`}
                                        {msg.time ? ` • ${msg.time.toFixed(0)}ms` : ''}
                                    </p>
                                )}
                            </div>
                            {msg.role === 'user' && (
                                <div className="w-8 h-8 rounded-lg bg-primary-500 flex items-center justify-center shrink-0">
                                    <User className="w-4 h-4 text-white" />
                                </div>
                            )}
                        </div>
                    ))}
                    {loading && (
                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-lg bg-primary-100 flex items-center justify-center">
                                <Bot className="w-4 h-4 text-primary-600" />
                            </div>
                            <div className="bg-surface-50 rounded-2xl px-4 py-3 border border-surface-100">
                                <div className="flex gap-1">
                                    <div className="w-2 h-2 rounded-full bg-surface-400 animate-bounce" style={{ animationDelay: '0s' }} />
                                    <div className="w-2 h-2 rounded-full bg-surface-400 animate-bounce" style={{ animationDelay: '0.15s' }} />
                                    <div className="w-2 h-2 rounded-full bg-surface-400 animate-bounce" style={{ animationDelay: '0.3s' }} />
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                <div className="px-5 py-3 border-t border-surface-100">
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                            placeholder={lmsConnected ? `Chat with ${selectedLmsModel?.split('/').pop() || 'LM Studio'}...` : 'Type a message...'}
                            className="flex-1 px-4 py-2.5 border border-surface-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-500"
                            disabled={loading}
                        />
                        <button
                            onClick={sendMessage}
                            disabled={!input.trim() || loading}
                            className="px-4 py-2.5 bg-primary-500 text-white rounded-xl hover:bg-primary-600 transition-colors disabled:opacity-50"
                        >
                            <Send className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            </div>

            {/* Side Panel */}
            <div className="w-72 space-y-4 shrink-0 overflow-auto">
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
                                {/* Model Selector */}
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

                                {/* Model List */}
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
                    />
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
        </div>
    );
}
