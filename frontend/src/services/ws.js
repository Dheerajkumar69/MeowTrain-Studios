/**
 * WebSocket service with auto-reconnect and exponential backoff.
 * Auth via first message instead of URL query param (security).
 */

const MIN_RECONNECT_DELAY = 1000;
const MAX_RECONNECT_DELAY = 10000;
const MAX_RECONNECT_ATTEMPTS = 20;

export function createWebSocket(projectId, onMessage, onError, onClose) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    // No token in URL — auth is sent as first message after connect
    const url = `${protocol}//${host}/api/projects/${projectId}/train/ws`;

    let ws = null;
    let reconnectAttempts = 0;
    let reconnectDelay = MIN_RECONNECT_DELAY;
    let reconnectTimer = null;
    let intentionallyClosed = false;

    function connect() {
        try {
            ws = new WebSocket(url);
        } catch (_err) {
            scheduleReconnect();
            return null;
        }

        ws.onopen = () => {
            reconnectAttempts = 0;
            reconnectDelay = MIN_RECONNECT_DELAY;
            // Send auth token as first message (re-read from storage on each
            // reconnect so we always use the freshest token — fixes M6)
            const token = localStorage.getItem('meowllm_token');
            if (token) {
                ws.send(JSON.stringify({ type: 'auth', token }));
            }
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                // Server may send an auth_error message
                if (data.type === 'auth_error') {
                    if (onError) onError(new Error(data.reason || 'Auth failed'));
                    return;
                }
                onMessage(data);
            } catch (_err) {
                // silently ignore parse errors
            }
        };

        ws.onerror = (event) => {
            if (onError) onError(event);
        };

        ws.onclose = (event) => {
            if (!intentionallyClosed) {
                scheduleReconnect();
            }
            if (onClose) onClose(event);
        };

        return ws;
    }

    function scheduleReconnect() {
        if (intentionallyClosed) return;
        if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
            if (onError) onError(new Error('Max reconnect attempts reached'));
            return;
        }

        reconnectAttempts++;
        const jitter = Math.random() * 500;
        const delay = Math.min(reconnectDelay + jitter, MAX_RECONNECT_DELAY);

        reconnectTimer = setTimeout(() => {
            reconnectDelay = Math.min(reconnectDelay * 1.5, MAX_RECONNECT_DELAY);
            connect();
        }, delay);
    }

    connect();

    return {
        close() {
            intentionallyClosed = true;
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }
            if (ws && ws.readyState !== WebSocket.CLOSED) {
                ws.close();
            }
        },
        get readyState() {
            return ws ? ws.readyState : WebSocket.CLOSED;
        },
    };
}
