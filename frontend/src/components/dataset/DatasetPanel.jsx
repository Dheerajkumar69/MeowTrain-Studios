import { useState, useRef } from 'react';
import { datasetsAPI } from '../../services/api';
import {
    Upload, FileText, File, Trash2, Eye, X,
    Hash, Layers, CheckCircle, AlertCircle, Loader
} from 'lucide-react';

export default function DatasetPanel({ projectId, datasets, setDatasets }) {
    const [uploading, setUploading] = useState(false);
    const [dragOver, setDragOver] = useState(false);
    const [preview, setPreview] = useState(null);
    const [previewLoading, setPreviewLoading] = useState(false);
    const fileInputRef = useRef(null);

    const handleFiles = async (files) => {
        setUploading(true);
        for (const file of files) {
            try {
                const formData = new FormData();
                formData.append('file', file);
                const res = await datasetsAPI.upload(projectId, formData);
                setDatasets((prev) => [res.data, ...prev]);
            } catch (err) {
                alert(`Failed to upload ${file.name}: ${err.response?.data?.detail || err.message}`);
            }
        }
        setUploading(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        handleFiles(Array.from(e.dataTransfer.files));
    };

    const handlePreview = async (dataset) => {
        setPreviewLoading(true);
        try {
            const res = await datasetsAPI.preview(projectId, dataset.id);
            setPreview(res.data);
        } catch (err) {
            alert('Failed to load preview');
        } finally {
            setPreviewLoading(false);
        }
    };

    const handleDelete = async (datasetId) => {
        if (!confirm('Delete this dataset?')) return;
        try {
            await datasetsAPI.delete(projectId, datasetId);
            setDatasets((prev) => prev.filter((d) => d.id !== datasetId));
            if (preview?.id === datasetId) setPreview(null);
        } catch (err) {
            alert('Failed to delete');
        }
    };

    const formatSize = (bytes) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    const totalTokens = datasets.reduce((sum, d) => sum + (d.token_count || 0), 0);

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h2 className="text-xl font-bold text-surface-900">Training Data</h2>
                <p className="text-sm text-surface-500 mt-1">Upload files to teach your model. Supported: .txt, .pdf, .md, .csv, .json, .docx</p>
            </div>

            {/* Drop Zone */}
            <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all ${dragOver
                    ? 'border-primary-400 bg-primary-50'
                    : 'border-surface-200 hover:border-primary-300 hover:bg-surface-50'
                    }`}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".txt,.pdf,.md,.csv,.json,.docx"
                    onChange={(e) => handleFiles(Array.from(e.target.files))}
                    className="hidden"
                />
                {uploading ? (
                    <div className="flex flex-col items-center gap-3">
                        <Loader className="w-10 h-10 text-primary-500 animate-spin" />
                        <p className="text-sm text-surface-600">Uploading & processing...</p>
                    </div>
                ) : (
                    <div className="flex flex-col items-center gap-3">
                        <div className="w-14 h-14 rounded-2xl bg-primary-50 flex items-center justify-center">
                            <Upload className="w-7 h-7 text-primary-500" />
                        </div>
                        <div>
                            <p className="text-sm font-medium text-surface-700">Drop files here or click to browse</p>
                            <p className="text-xs text-surface-400 mt-1">Text, PDF, Markdown, CSV, JSON, Word documents</p>
                        </div>
                    </div>
                )}
            </div>

            {/* Stats */}
            {datasets.length > 0 && (
                <div className="grid grid-cols-3 gap-4">
                    <div className="bg-white rounded-xl border border-surface-100 p-4 text-center">
                        <p className="text-2xl font-bold text-surface-900">{datasets.length}</p>
                        <p className="text-xs text-surface-500 mt-1">Files</p>
                    </div>
                    <div className="bg-white rounded-xl border border-surface-100 p-4 text-center">
                        <p className="text-2xl font-bold text-primary-600">{totalTokens.toLocaleString()}</p>
                        <p className="text-xs text-surface-500 mt-1">Estimated Tokens</p>
                    </div>
                    <div className="bg-white rounded-xl border border-surface-100 p-4 text-center">
                        <p className="text-2xl font-bold text-surface-900">{datasets.reduce((s, d) => s + (d.chunk_count || 0), 0)}</p>
                        <p className="text-xs text-surface-500 mt-1">Chunks</p>
                    </div>
                </div>
            )}

            {/* File List */}
            {datasets.length > 0 && (
                <div className="bg-white rounded-2xl border border-surface-100 overflow-hidden">
                    <div className="px-5 py-3 border-b border-surface-100 bg-surface-50">
                        <p className="text-sm font-medium text-surface-700">Uploaded Files</p>
                    </div>
                    <div className="divide-y divide-surface-100">
                        {datasets.map((ds) => (
                            <div key={ds.id} className="px-5 py-3 flex items-center gap-4 hover:bg-surface-50 transition-colors">
                                <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${ds.status === 'ready' ? 'bg-success-400/10' : 'bg-danger-400/10'
                                    }`}>
                                    {ds.status === 'ready' ? (
                                        <FileText className="w-4 h-4 text-success-600" />
                                    ) : (
                                        <AlertCircle className="w-4 h-4 text-danger-500" />
                                    )}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <p className="text-sm font-medium text-surface-900 truncate">{ds.original_name}</p>
                                    <p className="text-xs text-surface-500">
                                        {formatSize(ds.file_size || 0)} • {(ds.token_count || 0).toLocaleString()} tokens • {ds.chunk_count || 0} chunks
                                    </p>
                                </div>
                                <div className="flex items-center gap-1">
                                    <button
                                        onClick={() => handlePreview(ds)}
                                        className="p-1.5 text-surface-400 hover:text-primary-500 transition-colors rounded-lg hover:bg-primary-50"
                                        title="Preview"
                                    >
                                        <Eye className="w-4 h-4" />
                                    </button>
                                    <button
                                        onClick={() => handleDelete(ds.id)}
                                        className="p-1.5 text-surface-400 hover:text-danger-500 transition-colors rounded-lg hover:bg-danger-400/10"
                                        title="Delete"
                                    >
                                        <Trash2 className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Preview Modal */}
            {(preview || previewLoading) && (
                <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={() => setPreview(null)}>
                    <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col animate-fade-in" onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center justify-between p-5 border-b border-surface-100">
                            <div>
                                <h3 className="font-semibold text-surface-900">{preview?.original_name || 'Loading...'}</h3>
                                {preview && (
                                    <p className="text-xs text-surface-500 mt-0.5">
                                        {preview.total_tokens.toLocaleString()} tokens • {preview.total_chunks} chunks
                                    </p>
                                )}
                            </div>
                            <button onClick={() => setPreview(null)} className="text-surface-400 hover:text-surface-600">
                                <X className="w-5 h-5" />
                            </button>
                        </div>
                        <div className="flex-1 overflow-auto p-5 space-y-3">
                            {previewLoading ? (
                                <div className="flex items-center justify-center py-10">
                                    <Loader className="w-6 h-6 text-primary-500 animate-spin" />
                                </div>
                            ) : preview?.chunks?.map((chunk, i) => (
                                <div key={i} className="bg-surface-50 rounded-xl p-4 border border-surface-100">
                                    <div className="flex items-center gap-2 mb-2">
                                        <span className="text-xs font-medium text-surface-500 bg-surface-200 px-2 py-0.5 rounded-full">
                                            Chunk {chunk.index + 1}
                                        </span>
                                        <span className="text-xs text-surface-400">{chunk.token_count} tokens</span>
                                    </div>
                                    <p className="text-sm text-surface-700 whitespace-pre-wrap font-mono leading-relaxed">{chunk.text}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
