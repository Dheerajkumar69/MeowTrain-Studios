/**
 * ExportPanel — GGUF export section for LM Studio.
 */
import { useState, useEffect, useRef } from 'react';
import { Loader, CheckCircle, AlertCircle, Download } from 'lucide-react';
import { modelsAPI } from '../../services/api';

export default function ExportPanel({ projectId, setError }) {
    const [ggufStatus, setGgufStatus] = useState(null);
    const [ggufExporting, setGgufExporting] = useState(false);
    const ggufPollRef = useRef(null);

    // Cleanup poll on unmount
    useEffect(() => {
        return () => {
            if (ggufPollRef.current) {
                clearInterval(ggufPollRef.current);
                ggufPollRef.current = null;
            }
        };
    }, []);

    const exportGGUF = async (quantization = 'Q8_0') => {
        setGgufExporting(true);
        setGgufStatus(null);
        try {
            await modelsAPI.exportGGUF(projectId, quantization);
            if (ggufPollRef.current) clearInterval(ggufPollRef.current);
            const MAX_POLL_MS = 30 * 60 * 1000; // 30 minutes
            const pollStart = Date.now();
            ggufPollRef.current = setInterval(async () => {
                // Timeout guard: stop polling after 30 minutes
                if (Date.now() - pollStart > MAX_POLL_MS) {
                    clearInterval(ggufPollRef.current);
                    ggufPollRef.current = null;
                    setGgufExporting(false);
                    setGgufStatus({ step: 'error', message: 'Export timed out after 30 minutes. The model may be too large. Check server logs.' });
                    return;
                }
                try {
                    const res = await modelsAPI.ggufStatus(projectId);
                    setGgufStatus(res.data);
                    if (res.data.step === 'completed' || res.data.step === 'error') {
                        clearInterval(ggufPollRef.current);
                        ggufPollRef.current = null;
                        setGgufExporting(false);
                    }
                } catch (pollErr) {
                    clearInterval(ggufPollRef.current);
                    ggufPollRef.current = null;
                    setGgufExporting(false);
                }
            }, 2000);
        } catch (err) {
            setError(err.response?.data?.detail || 'GGUF export failed');
            setGgufExporting(false);
        }
    };

    const downloadGGUF = async () => {
        try {
            const res = await modelsAPI.ggufDownload(projectId);
            const url = window.URL.createObjectURL(new Blob([res.data]));
            const a = document.createElement('a');
            a.href = url;
            a.download = `meowllm-model.gguf`;
            a.click();
            window.URL.revokeObjectURL(url);
        } catch (err) {
            setError('GGUF download failed');
        }
    };

    return (
        <div className="bg-gradient-to-r from-violet-50 to-purple-50 rounded-2xl border border-violet-200 p-5">
            <h3 className="text-sm font-semibold text-violet-800 mb-1 flex items-center gap-2">
                🚀 Export for LM Studio
            </h3>
            <p className="text-xs text-violet-600 mb-3">
                Convert your trained model to GGUF format so you can load it in LM Studio and chat with it.
            </p>

            {ggufStatus?.step === 'completed' ? (
                <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm text-emerald-600">
                        <CheckCircle className="w-4 h-4" />
                        <span className="font-medium">{ggufStatus.message}</span>
                        {ggufStatus.gguf_size_mb && <span className="text-xs text-surface-500">({ggufStatus.gguf_size_mb} MB)</span>}
                    </div>
                    <button
                        onClick={downloadGGUF}
                        className="px-4 py-2 bg-violet-500 text-white rounded-xl text-sm font-medium hover:bg-violet-600 transition-colors flex items-center gap-2"
                    >
                        <Download className="w-4 h-4" /> Download .gguf file
                    </button>
                </div>
            ) : ggufExporting ? (
                <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm text-violet-700">
                        <Loader className="w-4 h-4 animate-spin" />
                        <span>{ggufStatus?.message || 'Converting...'}</span>
                    </div>
                    {ggufStatus?.progress > 0 && (
                        <div className="h-2 bg-violet-100 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-violet-500 rounded-full transition-all duration-500"
                                style={{ width: `${ggufStatus.progress}%` }}
                            />
                        </div>
                    )}
                </div>
            ) : (
                <div className="flex gap-2">
                    <button
                        onClick={() => exportGGUF('Q8_0')}
                        className="px-4 py-2 bg-violet-500 text-white rounded-xl text-sm font-medium hover:bg-violet-600 transition-colors"
                    >
                        Export Q8 (Best Quality)
                    </button>
                    <button
                        onClick={() => exportGGUF('Q4_K_M')}
                        className="px-4 py-2 bg-violet-400 text-white rounded-xl text-sm font-medium hover:bg-violet-500 transition-colors"
                    >
                        Export Q4 (Smaller/Faster)
                    </button>
                </div>
            )}

            {ggufStatus?.step === 'error' && (
                <div className="mt-2 text-xs text-danger-600 flex items-center gap-1">
                    <AlertCircle className="w-3.5 h-3.5" /> {ggufStatus.message}
                </div>
            )}
        </div>
    );
}
