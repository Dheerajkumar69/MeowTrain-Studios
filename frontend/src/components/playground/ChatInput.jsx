/**
 * ChatInput — message input field and send button.
 */
import { Send } from 'lucide-react';

export default function ChatInput({
    input, setInput, loading, onSend,
    lmsConnected, selectedLmsModel,
}) {
    return (
        <div className="px-5 py-3 border-t border-surface-100">
            <div className="flex gap-2">
                <label htmlFor="chat-input" className="sr-only">Type a message</label>
                <input
                    id="chat-input"
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && onSend()}
                    placeholder={lmsConnected ? `Chat with ${selectedLmsModel?.split('/').pop() || 'LM Studio'}...` : 'Type a message...'}
                    maxLength={4000}
                    className="flex-1 px-4 py-2.5 border border-surface-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-500"
                    disabled={loading}
                    aria-describedby="chat-status"
                />
                <button
                    onClick={onSend}
                    disabled={!input.trim() || loading}
                    className="px-4 py-2.5 bg-primary-500 text-white rounded-xl hover:bg-primary-600 transition-colors disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-primary-500/50"
                    aria-label="Send message"
                >
                    <Send className="w-4 h-4" aria-hidden="true" />
                </button>
            </div>
            <span id="chat-status" className="sr-only">{loading ? 'Generating response...' : 'Ready'}</span>
        </div>
    );
}
