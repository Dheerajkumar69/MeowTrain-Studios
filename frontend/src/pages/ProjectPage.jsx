import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { projectsAPI, datasetsAPI, modelsAPI, trainingAPI, inferenceAPI } from '../services/api';
import { ErrorBoundary } from '../components/ErrorBoundary';
import DatasetPanel from '../components/dataset/DatasetPanel';
import DatasetPreviewPanel from '../components/dataset/DatasetPreviewPanel';
import DatasetAugmentPanel from '../components/dataset/DatasetAugmentPanel';
import ModelSelector from '../components/models/ModelSelector';
import TrainingPanel from '../components/training/TrainingPanel';
import PlaygroundPanel from '../components/playground/PlaygroundPanel';
import {
    Cat, ArrowLeft, Database, Brain, Zap, MessageCircle,
    FolderOpen, Settings
} from 'lucide-react';

const TABS = [
    { id: 'data', label: 'Data', icon: Database, desc: 'Upload & manage training data' },
    { id: 'model', label: 'Model', icon: Brain, desc: 'Choose your base model' },
    { id: 'train', label: 'Train', icon: Zap, desc: 'Configure & run training' },
    { id: 'playground', label: 'Playground', icon: MessageCircle, desc: 'Test your model' },
];

export default function ProjectPage() {
    const { id } = useParams();
    const navigate = useNavigate();
    const { user } = useAuth();
    const [project, setProject] = useState(null);
    const [activeTab, setActiveTab] = useState('data');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    // Shared state across tabs
    const [datasets, setDatasets] = useState([]);
    const [selectedModel, setSelectedModel] = useState(null);
    const [trainingConfig, setTrainingConfig] = useState(null);

    const loadProject = useCallback(async () => {
        setError('');
        try {
            const res = await projectsAPI.get(id);
            setProject(res.data);
            try {
                const dsRes = await datasetsAPI.list(id);
                setDatasets(dsRes.data?.items ?? dsRes.data ?? []);
            } catch { /* dataset list failure is non-fatal */ }
        } catch (err) {
            if (err.response?.status === 404) {
                navigate('/');
            } else {
                setError(err.response?.data?.detail || 'Failed to load project');
            }
        } finally {
            setLoading(false);
        }
    }, [id, navigate]);

    useEffect(() => {
        loadProject();
    }, [loadProject]);

    if (loading) {
        return (
            <div className="min-h-screen bg-surface-50 flex flex-col">
                {/* Skeleton header */}
                <header className="bg-white border-b border-surface-100 px-6 py-3 flex items-center gap-4">
                    <div className="w-20 h-4 bg-surface-200 rounded animate-pulse" />
                    <div className="w-px h-6 bg-surface-200" />
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-surface-200 animate-pulse" />
                        <div>
                            <div className="w-32 h-4 bg-surface-200 rounded animate-pulse" />
                            <div className="w-48 h-3 bg-surface-100 rounded animate-pulse mt-1" />
                        </div>
                    </div>
                </header>
                {/* Skeleton tabs */}
                <div className="bg-white border-b border-surface-100 px-6">
                    <nav className="flex gap-4 max-w-5xl mx-auto py-3">
                        {[1, 2, 3, 4].map((i) => (
                            <div key={i} className="w-20 h-4 bg-surface-200 rounded animate-pulse" />
                        ))}
                    </nav>
                </div>
                {/* Skeleton content */}
                <main className="flex-1 p-6">
                    <div className="max-w-5xl mx-auto space-y-6">
                        <div className="h-6 w-40 bg-surface-200 rounded animate-pulse" />
                        <div className="h-4 w-72 bg-surface-100 rounded animate-pulse" />
                        <div className="border-2 border-dashed border-surface-200 rounded-2xl p-10 flex flex-col items-center gap-3">
                            <div className="w-14 h-14 rounded-2xl bg-surface-200 animate-pulse" />
                            <div className="w-48 h-4 bg-surface-200 rounded animate-pulse" />
                        </div>
                        <div className="grid grid-cols-3 gap-4">
                            {[1, 2, 3].map((i) => (
                                <div key={i} className="bg-white rounded-xl border border-surface-100 p-4">
                                    <div className="w-12 h-6 bg-surface-200 rounded animate-pulse mx-auto" />
                                    <div className="w-16 h-3 bg-surface-100 rounded animate-pulse mx-auto mt-2" />
                                </div>
                            ))}
                        </div>
                    </div>
                </main>
            </div>
        );
    }

    if (!project) {
        return (
            <div className="min-h-screen bg-surface-50 flex flex-col items-center justify-center gap-4">
                <div className="bg-danger-400/10 border border-danger-400/30 text-danger-600 text-sm rounded-xl p-4 max-w-md text-center">
                    {error || 'Project not found'}
                </div>
                <button onClick={() => navigate('/')} className="text-sm text-primary-600 hover:underline">← Back to Dashboard</button>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-surface-50 flex flex-col">
            {/* Top Bar */}
            <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:top-2 focus:left-2 focus:px-4 focus:py-2 focus:bg-primary-600 focus:text-white focus:rounded-lg focus:text-sm">
                Skip to main content
            </a>
            <header className="bg-white border-b border-surface-100 px-4 sm:px-6 py-3 flex items-center justify-between" role="banner">
                <div className="flex items-center gap-3 sm:gap-4 min-w-0">
                    <button
                        onClick={() => navigate('/')}
                        className="flex items-center gap-1.5 sm:gap-2 text-surface-500 hover:text-surface-700 transition-colors text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/50 rounded-lg px-1 py-0.5"
                        aria-label="Back to Dashboard"
                    >
                        <ArrowLeft className="w-4 h-4" aria-hidden="true" /> <span className="hidden sm:inline">Dashboard</span>
                    </button>
                    <div className="w-px h-6 bg-surface-200 hidden sm:block" aria-hidden="true" />
                    <div className="flex items-center gap-2 min-w-0">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shrink-0">
                            <Cat className="w-4 h-4 text-white" aria-hidden="true" />
                        </div>
                        <div className="min-w-0">
                            <h1 className="font-semibold text-surface-900 text-sm truncate">{project.name}</h1>
                            <p className="text-xs text-surface-500 truncate">{project.intended_use} • {project.status}</p>
                        </div>
                    </div>
                </div>
            </header>

            {/* Tab Navigation */}
            <div className="bg-white border-b border-surface-100 px-4 sm:px-6">
                <nav className="flex gap-1 max-w-5xl mx-auto overflow-x-auto scrollbar-hide" role="tablist" aria-label="Project sections">
                    {TABS.map((tab) => {
                        const Icon = tab.icon;
                        const isActive = activeTab === tab.id;
                        return (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                role="tab"
                                id={`tab-${tab.id}`}
                                aria-selected={isActive}
                                aria-controls={`tabpanel-${tab.id}`}
                                tabIndex={isActive ? 0 : -1}
                                onKeyDown={(e) => {
                                    const tabIds = TABS.map(t => t.id);
                                    const idx = tabIds.indexOf(tab.id);
                                    if (e.key === 'ArrowRight') {
                                        e.preventDefault();
                                        setActiveTab(tabIds[(idx + 1) % tabIds.length]);
                                    } else if (e.key === 'ArrowLeft') {
                                        e.preventDefault();
                                        setActiveTab(tabIds[(idx - 1 + tabIds.length) % tabIds.length]);
                                    } else if (e.key === 'Home') {
                                        e.preventDefault();
                                        setActiveTab(tabIds[0]);
                                    } else if (e.key === 'End') {
                                        e.preventDefault();
                                        setActiveTab(tabIds[tabIds.length - 1]);
                                    }
                                }}
                                className={`flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-3 text-xs sm:text-sm font-medium border-b-2 transition-all whitespace-nowrap focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:ring-inset ${isActive
                                    ? 'border-primary-500 text-primary-700'
                                    : 'border-transparent text-surface-500 hover:text-surface-700 hover:border-surface-300'
                                    }`}
                            >
                                <Icon className="w-4 h-4" aria-hidden="true" />
                                {tab.label}
                            </button>
                        );
                    })}
                </nav>
            </div>

            {/* Tab Content */}
            <main id="main-content" className="flex-1 p-4 sm:p-6 overflow-auto" role="main">
                <div className="max-w-5xl mx-auto w-full animate-fade-in">
                    {activeTab === 'data' && (
                        <div
                            role="tabpanel"
                            id="tabpanel-data"
                            aria-labelledby="tab-data"
                            className="space-y-6"
                        >
                            <ErrorBoundary compact label="Dataset Panel">
                                <DatasetPanel
                                    projectId={id}
                                    datasets={datasets}
                                    setDatasets={setDatasets}
                                />
                            </ErrorBoundary>
                            {datasets.filter(d => d.status === 'ready').length > 0 && (
                                <ErrorBoundary compact label="Dataset Preview">
                                    <DatasetPreviewPanel
                                        projectId={id}
                                        selectedModel={selectedModel}
                                    />
                                </ErrorBoundary>
                            )}
                            {datasets.filter(d => d.status === 'ready').length > 0 && (
                                <ErrorBoundary compact label="Dataset Augmentation">
                                    <DatasetAugmentPanel
                                        projectId={id}
                                        onAugmented={() => loadProject()}
                                    />
                                </ErrorBoundary>
                            )}
                        </div>
                    )}
                    {activeTab === 'model' && (
                        <div
                            role="tabpanel"
                            id="tabpanel-model"
                            aria-labelledby="tab-model"
                        >
                            <ErrorBoundary compact label="Model Selector">
                                <ModelSelector
                                    selectedModel={selectedModel}
                                    onSelectModel={setSelectedModel}
                                />
                            </ErrorBoundary>
                        </div>
                    )}
                    {activeTab === 'train' && (
                        <div
                            role="tabpanel"
                            id="tabpanel-train"
                            aria-labelledby="tab-train"
                        >
                            <ErrorBoundary compact label="Training Panel">
                                <TrainingPanel
                                    projectId={id}
                                    selectedModel={selectedModel}
                                    datasets={datasets}
                                    project={project}
                                    onProjectUpdate={loadProject}
                                />
                            </ErrorBoundary>
                        </div>
                    )}
                    {activeTab === 'playground' && (
                        <div
                            role="tabpanel"
                            id="tabpanel-playground"
                            aria-labelledby="tab-playground"
                        >
                            <ErrorBoundary compact label="Playground">
                                <PlaygroundPanel
                                    projectId={id}
                                    project={project}
                                    datasets={datasets}
                                />
                            </ErrorBoundary>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}
