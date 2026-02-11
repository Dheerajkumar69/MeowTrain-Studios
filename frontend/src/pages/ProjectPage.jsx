import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { projectsAPI, datasetsAPI, modelsAPI, trainingAPI, inferenceAPI } from '../services/api';
import DatasetPanel from '../components/dataset/DatasetPanel';
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
                setDatasets(dsRes.data || []);
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
            <div className="min-h-screen bg-surface-50 flex items-center justify-center">
                <div className="w-8 h-8 border-3 border-primary-200 border-t-primary-600 rounded-full animate-spin" />
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
            <header className="bg-white border-b border-surface-100 px-6 py-3 flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <button
                        onClick={() => navigate('/')}
                        className="flex items-center gap-2 text-surface-500 hover:text-surface-700 transition-colors text-sm"
                    >
                        <ArrowLeft className="w-4 h-4" /> Dashboard
                    </button>
                    <div className="w-px h-6 bg-surface-200" />
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
                            <Cat className="w-4 h-4 text-white" />
                        </div>
                        <div>
                            <h1 className="font-semibold text-surface-900 text-sm">{project.name}</h1>
                            <p className="text-xs text-surface-500">{project.intended_use} • {project.status}</p>
                        </div>
                    </div>
                </div>
            </header>

            {/* Tab Navigation */}
            <div className="bg-white border-b border-surface-100 px-6">
                <nav className="flex gap-1 max-w-5xl mx-auto">
                    {TABS.map((tab) => {
                        const Icon = tab.icon;
                        const isActive = activeTab === tab.id;
                        return (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-all ${isActive
                                    ? 'border-primary-500 text-primary-700'
                                    : 'border-transparent text-surface-500 hover:text-surface-700 hover:border-surface-300'
                                    }`}
                            >
                                <Icon className="w-4 h-4" />
                                {tab.label}
                            </button>
                        );
                    })}
                </nav>
            </div>

            {/* Tab Content */}
            <main className="flex-1 p-6 overflow-auto">
                <div className="max-w-5xl mx-auto animate-fade-in">
                    {activeTab === 'data' && (
                        <DatasetPanel
                            projectId={id}
                            datasets={datasets}
                            setDatasets={setDatasets}
                        />
                    )}
                    {activeTab === 'model' && (
                        <ModelSelector
                            selectedModel={selectedModel}
                            onSelectModel={setSelectedModel}
                        />
                    )}
                    {activeTab === 'train' && (
                        <TrainingPanel
                            projectId={id}
                            selectedModel={selectedModel}
                            datasets={datasets}
                            project={project}
                            onProjectUpdate={loadProject}
                        />
                    )}
                    {activeTab === 'playground' && (
                        <PlaygroundPanel
                            projectId={id}
                            project={project}
                            datasets={datasets}
                        />
                    )}
                </div>
            </main>
        </div>
    );
}
