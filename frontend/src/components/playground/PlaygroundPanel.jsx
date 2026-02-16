/**
 * PlaygroundPanel — thin orchestrator that delegates to:
 *   • ChatView      – message list + empty state
 *   • ChatInput     – input field + send button
 *   • SettingsPanel – LM Studio, system prompt, temp, tokens, streaming, context, templates
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { inferenceAPI, lmstudioAPI } from '../../services/api';
import {
    Bot, Bookmark, BookOpen, Trash2, Server, Wifi, Zap,
    AlertTriangle, Brain,
} from 'lucide-react';
import ChatView from './ChatView';
import ChatInput from './ChatInput';
import SettingsPanel from './SettingsPanel';

export default function PlaygroundPanel({ projectId, project, datasets }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [systemPrompt, setSystemPrompt] = useState('You are a helpful assistant.');
    const [temperature, setTemperature] = useState(0.7);
    const [maxTokens, setMaxTokens] = useState(512);
    const [loading, setLoading] = useState(false);
    const [showContext, setShowContext] = useState(false);
    const [selectedContextIds, setSelectedContextIds] = useState([]);
    const [showTemplates, setShowTemplates] = useState(false);
    const [useStreaming, setUseStreaming] = useState(true);
    const messagesEndRef = useRef(null);
    const streamAbortRef = useRef(null);

    // LM Studio state
    const [showLMStudio, setShowLMStudio] = useState(false);
    const [lmsConfig, setLmsConfig] = useState({ host: 'http://localhost', port: 1234, enabled: false });
    const [lmsModels, setLmsModels] = useState([]);
    const [selectedLmsModel, setSelectedLmsModel] = useState(null);
    const [lmsConnected, setLmsConnected] = useState(false);

    // Model status
    const [modelInfo, setModelInfo] = useState(null);
    const [modelChecking, setModelChecking] = useState(true);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Abort any active stream on unmount
    useEffect(() => {
        return () => {
            if (streamAbortRef.current) {
                streamAbortRef.current();
                streamAbortRef.current = null;
            }
        };
    }, []);

    useEffect(() => {
        // Load LM Studio config
        lmstudioAPI.getConfig().then((res) => {
            setLmsConfig(res.data);
            if (res.data.enabled) {
                lmstudioAPI.listModels().then((r) => {
                    if (r.data.models) {
                        setLmsModels(r.data.models);
                        setLmsConnected(true);
                        if (r.data.models.length > 0) setSelectedLmsModel(r.data.models[0].model_id);
                    }
                }).catch(() => {});
            }
        }).catch(() => {});

        // Check model availability
        setModelChecking(true);
        inferenceAPI.getContext(projectId).then((res) => {
            setModelInfo(res.data?.model || null);
        }).catch(() => setModelInfo(null))
          .finally(() => setModelChecking(false));
    }, [projectId]);

    const sendMessage = async () => {
        if (!input.trim() || loading) return;
        const userMsg = { role: 'user', content: input };
        setMessages((prev) => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        const payload = {
            prompt: input,
            system_prompt: systemPrompt,
            temperature,
            max_tokens: maxTokens,
            context_dataset_ids: selectedContextIds,
        };

        if (lmsConnected && selectedLmsModel && lmsConfig.enabled) {
            payload.lmstudio_model = selectedLmsModel;
        }

        const canStream = useStreaming && !(lmsConnected && selectedLmsModel && lmsConfig.enabled);

        if (canStream) {
            const assistantIdx = { current: -1 };
            setMessages((prev) => {
                assistantIdx.current = prev.length;
                return [...prev, { role: 'assistant', content: '', streaming: true }];
            });

            const { abort } = inferenceAPI.chatStream(
                projectId,
                payload,
                {
                    onToken: (text) => {
                        setMessages((prev) => {
                            const updated = [...prev];
                            const msg = updated[assistantIdx.current];
                            if (msg) updated[assistantIdx.current] = { ...msg, content: msg.content + text };
                            return updated;
                        });
                    },
                    onDone: () => {
                        setMessages((prev) => {
                            const updated = [...prev];
                            const msg = updated[assistantIdx.current];
                            if (msg) updated[assistantIdx.current] = { ...msg, streaming: false };
                            return updated;
                        });
                        setLoading(false);
                        streamAbortRef.current = null;
                    },
                    onError: (err) => {
                        setMessages((prev) => {
                            const updated = [...prev];
                            const msg = updated[assistantIdx.current];
                            if (msg && !msg.content) {
                                updated[assistantIdx.current] = {
                                    role: 'assistant',
                                    content: 'Error: ' + (err.message || 'Stream failed'),
                                    isError: true, streaming: false,
                                };
                            } else if (msg) {
                                updated[assistantIdx.current] = { ...msg, streaming: false };
                            }
                            return updated;
                        });
                        setLoading(false);
                        streamAbortRef.current = null;
                    },
                },
            );
            streamAbortRef.current = abort;
        } else {
            try {
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
        }
    };

    const cancelStream = useCallback(() => {
        if (streamAbortRef.current) {
            streamAbortRef.current();
            streamAbortRef.current = null;
            setLoading(false);
            setMessages((prev) => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last?.streaming) updated[updated.length - 1] = { ...last, streaming: false };
                return updated;
            });
        }
    }, []);

    const toggleContext = (id) => {
        setSelectedContextIds((prev) =>
            prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
        );
    };

    const readyDatasets = (datasets || []).filter((d) => d.status === 'ready');

    return (
        <div className="flex flex-col lg:flex-row gap-4 h-auto lg:h-[calc(100vh-12rem)]">
            {/* Chat Panel */}
            <div className="flex-1 flex flex-col bg-white rounded-2xl border border-surface-100 overflow-hidden min-h-[400px] lg:min-h-0" role="region" aria-label="Chat conversation">
                {/* Chat Header */}
                <div className="px-5 py-3 border-b border-surface-100 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Bot className="w-5 h-5 text-primary-500" />
                        <span className="font-semibold text-surface-900 text-sm">Playground</span>
                        {lmsConnected && selectedLmsModel && lmsConfig.enabled ? (
                            <span className="flex items-center gap-1 text-xs px-2 py-0.5 bg-violet-50 text-violet-700 rounded-full border border-violet-200">
                                <Zap className="w-3 h-3" /> {selectedLmsModel.split('/').pop()}
                            </span>
                        ) : modelInfo ? (
                            <span className="flex items-center gap-1 text-xs px-2 py-0.5 bg-emerald-50 text-emerald-700 rounded-full border border-emerald-200">
                                <Brain className="w-3 h-3" /> Fine-tuned
                            </span>
                        ) : (
                            <span className="flex items-center gap-1 text-xs px-2 py-0.5 bg-surface-50 text-surface-400 rounded-full border border-surface-200">
                                <AlertTriangle className="w-3 h-3" /> No model
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

                <ChatView
                    messages={messages}
                    loading={loading}
                    streamAbortRef={streamAbortRef}
                    cancelStream={cancelStream}
                    modelInfo={modelInfo}
                    modelChecking={modelChecking}
                    lmsConnected={lmsConnected}
                    selectedLmsModel={selectedLmsModel}
                    lmsConfig={lmsConfig}
                    onShowLMStudio={setShowLMStudio}
                    messagesEndRef={messagesEndRef}
                />

                <ChatInput
                    input={input}
                    setInput={setInput}
                    loading={loading}
                    onSend={sendMessage}
                    lmsConnected={lmsConnected}
                    selectedLmsModel={selectedLmsModel}
                />
            </div>

            <SettingsPanel
                projectId={projectId}
                systemPrompt={systemPrompt}
                setSystemPrompt={setSystemPrompt}
                temperature={temperature}
                setTemperature={setTemperature}
                maxTokens={maxTokens}
                setMaxTokens={setMaxTokens}
                useStreaming={useStreaming}
                setUseStreaming={setUseStreaming}
                readyDatasets={readyDatasets}
                selectedContextIds={selectedContextIds}
                toggleContext={toggleContext}
                showContext={showContext}
                showTemplates={showTemplates}
                showLMStudio={showLMStudio}
                lmsConfig={lmsConfig}
                setLmsConfig={setLmsConfig}
                lmsModels={lmsModels}
                setLmsModels={setLmsModels}
                selectedLmsModel={selectedLmsModel}
                setSelectedLmsModel={setSelectedLmsModel}
                lmsConnected={lmsConnected}
                setLmsConnected={setLmsConnected}
                input={input}
            />
        </div>
    );
}
