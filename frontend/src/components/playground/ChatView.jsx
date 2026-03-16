/**
 * ChatView — message list and empty state for the playground.
 */
import {
    Bot, User, Loader, AlertTriangle, CheckCircle, Zap, Brain, Server,
} from 'lucide-react';

export default function ChatView({
    messages, loading, cancelStream,
    modelInfo, modelChecking, lmsConnected, selectedLmsModel,
    onShowLMStudio, messagesEndRef,
}) {
    return (
        <div className="flex-1 overflow-auto p-5 space-y-4">
            {messages.length === 0 && (
                <EmptyState
                    modelInfo={modelInfo}
                    modelChecking={modelChecking}
                    lmsConnected={lmsConnected}
                    selectedLmsModel={selectedLmsModel}
                    onShowLMStudio={onShowLMStudio}
                />
            )}
            {messages.map((msg, i) => (
                <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                    {msg.role === 'assistant' && (
                        <div className="w-8 h-8 rounded-lg bg-primary-100 flex items-center justify-center shrink-0">
                            <Bot className="w-4 h-4 text-primary-600" aria-hidden="true" />
                        </div>
                    )}
                    <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${msg.role === 'user'
                        ? 'bg-primary-500 text-white'
                        : msg.isError
                            ? 'bg-danger-400/10 text-danger-600 border border-danger-400/30'
                            : 'bg-surface-50 text-surface-800 border border-surface-100'
                        }`}>
                        <p className="text-sm whitespace-pre-wrap leading-relaxed">
                            {msg.content}
                            {msg.streaming && <span className="inline-block w-2 h-4 bg-primary-500 ml-0.5 animate-pulse rounded-sm" aria-hidden="true" />}
                        </p>
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
                            <User className="w-4 h-4 text-white" aria-hidden="true" />
                        </div>
                    )}
                </div>
            ))}
            {loading && (
                <div className="flex gap-3 items-end">
                    <div className="w-8 h-8 rounded-lg bg-primary-100 flex items-center justify-center">
                        <Bot className="w-4 h-4 text-primary-600" aria-hidden="true" />
                    </div>
                    {cancelStream ? (
                        <button
                            onClick={cancelStream}
                            className="text-xs text-surface-500 hover:text-danger-500 bg-surface-50 border border-surface-200 rounded-xl px-3 py-1.5 transition-colors focus:outline-none focus:ring-2 focus:ring-danger-500/50"
                            aria-label="Stop generating"
                        >
                            ■ Stop generating
                        </button>
                    ) : (
                        <div className="bg-surface-50 rounded-2xl px-4 py-3 border border-surface-100">
                            <div className="flex gap-1" aria-label="Generating response">
                                <div className="w-2 h-2 rounded-full bg-surface-400 animate-bounce" style={{ animationDelay: '0s' }} />
                                <div className="w-2 h-2 rounded-full bg-surface-400 animate-bounce" style={{ animationDelay: '0.15s' }} />
                                <div className="w-2 h-2 rounded-full bg-surface-400 animate-bounce" style={{ animationDelay: '0.3s' }} />
                            </div>
                        </div>
                    )}
                </div>
            )}
            <div ref={messagesEndRef} />
        </div>
    );
}

function EmptyState({ modelInfo, modelChecking, lmsConnected, selectedLmsModel, onShowLMStudio }) {
    return (
        <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 rounded-2xl bg-primary-50 flex items-center justify-center mb-4">
                <Bot className="w-8 h-8 text-primary-400" />
            </div>
            <h3 className="text-lg font-semibold text-surface-700 mb-1">Test Your Model</h3>
            <p className="text-sm text-surface-500 max-w-sm mb-4">
                Start a conversation to test how your model responds.
            </p>

            <div className="space-y-2 mb-4 w-full max-w-xs">
                {modelChecking ? (
                    <div className="flex items-center justify-center gap-2 px-3 py-2 bg-surface-50 rounded-xl text-xs text-surface-500">
                        <Loader className="w-3 h-3 animate-spin" /> Checking model...
                    </div>
                ) : modelInfo ? (
                    <div className="flex items-center gap-2 px-3 py-2 bg-emerald-50 border border-emerald-200 rounded-xl text-xs text-emerald-700">
                        <CheckCircle className="w-3.5 h-3.5" />
                        <span className="font-medium">Fine-tuned model ready</span>
                        {modelInfo.is_loaded && <span className="ml-auto text-emerald-500">Loaded</span>}
                    </div>
                ) : (
                    <div className="flex items-center gap-2 px-3 py-2 bg-amber-50 border border-amber-200 rounded-xl text-xs text-amber-700">
                        <AlertTriangle className="w-3.5 h-3.5" />
                        <span>No fine-tuned model — train one in the Train tab</span>
                    </div>
                )}
                {lmsConnected ? (
                    <div className="flex items-center gap-2 px-3 py-2 bg-violet-50 border border-violet-200 rounded-xl text-xs text-violet-700">
                        <Zap className="w-3.5 h-3.5" />
                        <span className="font-medium">LM Studio connected</span>
                        <span className="ml-auto text-violet-500 truncate max-w-[120px]">{selectedLmsModel?.split('/').pop()}</span>
                    </div>
                ) : (
                    <button
                        onClick={() => onShowLMStudio(true)}
                        className="w-full flex items-center gap-2 px-3 py-2 bg-violet-50 text-violet-700 rounded-xl text-xs border border-violet-200 hover:bg-violet-100 transition-colors"
                    >
                        <Server className="w-3.5 h-3.5" /> Connect LM Studio for inference
                    </button>
                )}
            </div>

            {!modelInfo && !lmsConnected && (
                <p className="text-xs text-surface-400 max-w-xs">
                    You need either a fine-tuned model or LM Studio to start chatting.
                </p>
            )}
        </div>
    );
}
