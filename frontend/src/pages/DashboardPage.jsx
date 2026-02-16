import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { projectsAPI, hardwareAPI } from '../services/api';
import { useToast } from '../components/Toast';
import ConfirmDialog from '../components/ConfirmDialog';
import { DashboardSkeleton, EmptyProjects } from '../components/Skeletons';
import {
    Cat, Plus, FolderOpen, Cpu, MemoryStick, HardDrive,
    LogOut, Search, Bot, Code, MessageCircle, Wand2, Trash2,
    Clock, Database, ChevronRight, Sparkles, Settings,
    Thermometer, Flame, Activity, Gauge
} from 'lucide-react';

const USE_ICONS = { chatbot: <Bot className="w-5 h-5" />, code: <Code className="w-5 h-5" />, qa: <MessageCircle className="w-5 h-5" />, custom: <Wand2 className="w-5 h-5" /> };
const USE_COLORS = { chatbot: 'from-blue-500 to-cyan-400', code: 'from-violet-500 to-purple-400', qa: 'from-amber-500 to-orange-400', custom: 'from-emerald-500 to-teal-400' };

function ProgressBar({ value, max, color = 'bg-primary-500', label }) {
    const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
    return (
        <div className="space-y-1">
            <div className="flex justify-between text-xs text-surface-500">
                <span>{label}</span>
                <span>{pct.toFixed(0)}%</span>
            </div>
            <div className="h-2 bg-surface-100 rounded-full overflow-hidden">
                <div className={`h-full rounded-full transition-all duration-500 ease-out ${color}`} style={{ width: `${pct}%` }} />
            </div>
        </div>
    );
}

export default function DashboardPage() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const toast = useToast();
    const [projects, setProjects] = useState([]);
    const [hardware, setHardware] = useState(null);
    const [showModal, setShowModal] = useState(false);
    const [search, setSearch] = useState('');
    const [loading, setLoading] = useState(true);
    const [hwError, setHwError] = useState(false);
    const hwPollRef = useRef(null);
    const [deleteTarget, setDeleteTarget] = useState(null);

    // New project form
    const [newName, setNewName] = useState('');
    const [newDesc, setNewDesc] = useState('');
    const [newUse, setNewUse] = useState('custom');
    const [creating, setCreating] = useState(false);

    // Load projects once
    useEffect(() => {
        const loadProjects = async () => {
            try {
                const res = await projectsAPI.list();
                setProjects(res.data?.items ?? res.data ?? []);
            } catch (err) {
                // silently handle error — UI shows no projects
            } finally {
                setLoading(false);
            }
        };
        loadProjects();
    }, []);

    // Poll hardware every 0.5s
    useEffect(() => {
        const pollHardware = async () => {
            try {
                const res = await hardwareAPI.status();
                setHardware(res.data);
                setHwError(false);
            } catch {
                setHwError(true);
            }
        };
        pollHardware();
        hwPollRef.current = setInterval(pollHardware, 3000);
        return () => clearInterval(hwPollRef.current);
    }, []);

    const createProject = async (e) => {
        e.preventDefault();
        if (!newName.trim()) return;
        setCreating(true);
        try {
            const res = await projectsAPI.create({ name: newName.trim(), description: newDesc.trim(), intended_use: newUse });
            toast.success('Project created successfully!');
            navigate(`/project/${res.data.id}`);
        } catch (err) {
            toast.error(err.response?.data?.detail || 'Failed to create project');
        } finally {
            setCreating(false);
        }
    };

    const deleteProject = async (id, e) => {
        e.stopPropagation();
        setDeleteTarget(id);
    };

    const confirmDelete = async () => {
        const id = deleteTarget;
        setDeleteTarget(null);
        try {
            await projectsAPI.delete(id);
            setProjects((p) => p.filter((x) => x.id !== id));
            toast.success('Project deleted');
        } catch (err) {
            toast.error(err.response?.data?.detail || 'Failed to delete project');
        }
    };

    const filtered = projects.filter((p) => p.name?.toLowerCase().includes(search.toLowerCase()));

    return (
        <div className="min-h-screen bg-surface-50 flex flex-col md:flex-row">
            {/* Skip link */}
            <a href="#dashboard-main" className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:top-2 focus:left-2 focus:px-4 focus:py-2 focus:bg-primary-600 focus:text-white focus:rounded-lg focus:text-sm">
                Skip to main content
            </a>
            {/* Sidebar — hidden on mobile, shown on md+ */}
            <aside className="hidden md:flex w-64 bg-white border-r border-surface-100 flex-col animate-slide-in-left" role="complementary" aria-label="Sidebar navigation">
                <div className="p-5 border-b border-surface-100">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
                            <Cat className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h1 className="font-bold text-surface-900 text-sm">MeowLLM Studio</h1>
                            <p className="text-xs text-surface-500">Local AI Training</p>
                        </div>
                    </div>
                </div>

                <nav className="flex-1 p-3 space-y-1">
                    <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-primary-50 text-primary-700 text-sm font-medium">
                        <FolderOpen className="w-4 h-4" /> Dashboard
                    </button>
                </nav>

                {/* Hardware Mini Widget — Live */}
                {hardware && (
                    <div className="p-4 border-t border-surface-100 space-y-3">
                        <p className="text-xs font-semibold text-surface-500 uppercase tracking-wider flex items-center gap-1">
                            <Activity className="w-3 h-3" /> System
                            <span className="ml-auto text-emerald-500 animate-pulse text-[10px]">● Live</span>
                        </p>
                        <ProgressBar value={hardware.ram_used_gb || (hardware.ram_total_gb - hardware.ram_available_gb)} max={hardware.ram_total_gb} color="bg-blue-500" label="RAM" />
                        <ProgressBar value={hardware.cpu_usage_percent} max={100} color="bg-emerald-500" label="CPU" />
                        {hardware.gpu_available && (
                            <>
                                <ProgressBar value={hardware.gpu_vram_used_gb || (hardware.gpu_vram_total_gb - hardware.gpu_vram_available_gb)} max={hardware.gpu_vram_total_gb} color="bg-violet-500" label="VRAM" />
                                <div className="flex items-center justify-between text-xs text-surface-500">
                                    <span className="flex items-center gap-1">
                                        <Thermometer className="w-3 h-3 text-amber-500" />
                                        {hardware.gpu_temp_celsius != null ? `${hardware.gpu_temp_celsius}°C` : '—'}
                                    </span>
                                    <span className="flex items-center gap-1">
                                        <Flame className="w-3 h-3 text-orange-500" />
                                        {hardware.gpu_power_watts != null ? `${hardware.gpu_power_watts}W` : '—'}
                                    </span>
                                </div>
                            </>
                        )}
                        <div className="flex items-center gap-2 text-xs text-surface-500">
                            <Database className="w-3 h-3" />
                            <span>Cache: {hardware.model_cache_size_gb} GB</span>
                        </div>
                    </div>
                )}

                {/* User */}
                <div className="p-4 border-t border-surface-100 flex items-center justify-between">
                    <div className="flex items-center gap-2 min-w-0">
                        <div className="w-8 h-8 rounded-lg bg-primary-100 flex items-center justify-center shrink-0">
                            <span className="text-sm font-bold text-primary-700">{user?.display_name?.[0]?.toUpperCase() || '?'}</span>
                        </div>
                        <span className="text-sm text-surface-700 truncate">{user?.display_name || 'User'}</span>
                    </div>
                    <button onClick={logout} className="text-surface-400 hover:text-danger-500 transition-colors focus:outline-none focus:ring-2 focus:ring-danger-500/50 rounded" aria-label="Sign out">
                        <LogOut className="w-4 h-4" aria-hidden="true" />
                    </button>
                </div>
            </aside>

            {/* Mobile top bar — shown on mobile only */}
            <header className="flex md:hidden items-center justify-between bg-white border-b border-surface-100 px-4 py-3" role="banner">
                <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
                        <Cat className="w-5 h-5 text-white" aria-hidden="true" />
                    </div>
                    <div>
                        <h1 className="font-bold text-surface-900 text-sm">MeowLLM Studio</h1>
                        <p className="text-xs text-surface-500">Local AI Training</p>
                    </div>
                </div>
                <button onClick={logout} className="text-surface-400 hover:text-danger-500 transition-colors" aria-label="Sign out">
                    <LogOut className="w-4 h-4" />
                </button>
            </header>

            {/* Main Content */}
            <main id="dashboard-main" className="flex-1 p-4 sm:p-6 md:p-8 overflow-auto" role="main">
                <div className="max-w-6xl mx-auto animate-fade-in">
                    {/* Header */}
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
                        <div>
                            <h2 className="text-2xl font-bold text-surface-900">Welcome back, {user?.display_name || 'User'} 👋</h2>
                            <p className="text-surface-500 mt-1">Manage your AI training projects</p>
                        </div>
                        <button
                            id="new-project-btn"
                            onClick={() => setShowModal(true)}
                            className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-primary-600 to-primary-500 text-white rounded-xl font-medium text-sm hover:from-primary-700 hover:to-primary-600 transition-all shadow-md hover:shadow-lg"
                        >
                            <Plus className="w-4 h-4" /> New Project
                        </button>
                    </div>

                    {/* Search */}
                    <div className="relative mb-6">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-surface-400" />
                        <input
                            type="text"
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            placeholder="Search projects..."
                            className="w-full max-w-md pl-10 pr-4 py-2.5 border border-surface-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-500 bg-white transition-all"
                        />
                    </div>

                    {/* Hardware Banner */}
                    {hardware && (
                        <div className="glass rounded-2xl p-6 mb-8 border border-surface-200">
                            <h3 className="text-sm font-semibold text-surface-700 mb-4 flex items-center gap-2">
                                <Settings className="w-4 h-4" /> Hardware Status
                                <span className="ml-auto text-xs text-emerald-500 font-normal animate-pulse">● Live (3s)</span>
                            </h3>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {/* CPU */}
                                <div className="bg-white rounded-xl p-4 border border-surface-100">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Cpu className="w-4 h-4 text-emerald-500" />
                                        <span className="text-xs font-medium text-surface-600">CPU</span>
                                    </div>
                                    <p className="text-sm font-semibold text-surface-900 truncate">{hardware.cpu_name || 'Unknown'}</p>
                                    <div className="mt-2">
                                        <div className="h-1.5 bg-surface-100 rounded-full overflow-hidden">
                                            <div className="h-full bg-emerald-500 rounded-full transition-all duration-500" style={{ width: `${hardware.cpu_usage_percent || 0}%` }} />
                                        </div>
                                        <p className="text-xs text-surface-500 mt-1">{hardware.cpu_cores || '—'} cores • {hardware.cpu_usage_percent?.toFixed(0) ?? '—'}%</p>
                                    </div>
                                </div>

                                {/* RAM */}
                                <div className="bg-white rounded-xl p-4 border border-surface-100">
                                    <div className="flex items-center gap-2 mb-2">
                                        <MemoryStick className="w-4 h-4 text-blue-500" />
                                        <span className="text-xs font-medium text-surface-600">RAM</span>
                                    </div>
                                    <p className="text-sm font-semibold text-surface-900">{hardware.ram_total_gb || '—'} GB total</p>
                                    <div className="mt-2">
                                        <div className="h-1.5 bg-surface-100 rounded-full overflow-hidden">
                                            <div className="h-full bg-blue-500 rounded-full transition-all duration-500" style={{ width: `${hardware.ram_usage_percent || 0}%` }} />
                                        </div>
                                        <p className="text-xs text-surface-500 mt-1">{hardware.ram_available_gb?.toFixed(1) ?? '—'} GB free • {hardware.ram_usage_percent?.toFixed(0) ?? '—'}%</p>
                                    </div>
                                </div>

                                {/* GPU */}
                                <div className="bg-white rounded-xl p-4 border border-surface-100">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Gauge className="w-4 h-4 text-violet-500" />
                                        <span className="text-xs font-medium text-surface-600">GPU</span>
                                    </div>
                                    {hardware.gpu_available ? (
                                        <>
                                            <p className="text-sm font-semibold text-surface-900 truncate">{hardware.gpu_name?.replace('NVIDIA ', '') || 'GPU'}</p>
                                            <div className="mt-2">
                                                <div className="h-1.5 bg-surface-100 rounded-full overflow-hidden">
                                                    <div className="h-full bg-violet-500 rounded-full transition-all duration-500" style={{ width: `${hardware.gpu_usage_percent || 0}%` }} />
                                                </div>
                                                <p className="text-xs text-surface-500 mt-1">
                                                    {hardware.gpu_usage_percent ?? '—'}% •{' '}
                                                    <span className={hardware.gpu_temp_celsius > 80 ? 'text-red-500 font-medium' : ''}>
                                                        {hardware.gpu_temp_celsius ?? '—'}°C
                                                    </span>
                                                    {' '} • {hardware.gpu_power_watts ?? '—'}W
                                                </p>
                                            </div>
                                        </>
                                    ) : (
                                        <>
                                            <p className="text-sm font-semibold text-surface-900">No GPU detected</p>
                                            <p className="text-xs text-surface-500 mt-1">CPU mode will be used</p>
                                        </>
                                    )}
                                </div>

                                {/* VRAM / Storage */}
                                <div className="bg-white rounded-xl p-4 border border-surface-100">
                                    {hardware.gpu_available ? (
                                        <>
                                            <div className="flex items-center gap-2 mb-2">
                                                <HardDrive className="w-4 h-4 text-purple-500" />
                                                <span className="text-xs font-medium text-surface-600">VRAM</span>
                                            </div>
                                            <p className="text-sm font-semibold text-surface-900">{hardware.gpu_vram_total_gb ?? '—'} GB</p>
                                            <div className="mt-2">
                                                <div className="h-1.5 bg-surface-100 rounded-full overflow-hidden">
                                                    <div className="h-full bg-purple-500 rounded-full transition-all duration-500" style={{ width: `${hardware.gpu_vram_total_gb > 0 ? (((hardware.gpu_vram_used_gb || 0) / hardware.gpu_vram_total_gb) * 100) : 0}%` }} />
                                                </div>
                                                <p className="text-xs text-surface-500 mt-1">{hardware.gpu_vram_available_gb?.toFixed(1) ?? '—'} GB free</p>
                                            </div>
                                        </>
                                    ) : (
                                        <>
                                            <div className="flex items-center gap-2 mb-2">
                                                <Database className="w-4 h-4 text-amber-500" />
                                                <span className="text-xs font-medium text-surface-600">Storage</span>
                                            </div>
                                            <p className="text-sm font-semibold text-surface-900">{hardware.disk_free_gb ?? '—'} GB free</p>
                                            <p className="text-xs text-surface-500 mt-1">Cache: {hardware.model_cache_size_gb ?? 0} GB</p>
                                        </>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Projects Grid */}
                    {loading ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {[1, 2, 3].map((i) => (
                                <div key={i} className="skeleton h-40 rounded-2xl" />
                            ))}
                        </div>
                    ) : filtered.length === 0 ? (
                        <div className="text-center py-20">
                            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-surface-100 mb-4">
                                <FolderOpen className="w-8 h-8 text-surface-400" />
                            </div>
                            <h3 className="text-lg font-semibold text-surface-700 mb-2">
                                {projects.length === 0 ? 'No projects yet' : 'No matching projects'}
                            </h3>
                            <p className="text-surface-500 text-sm mb-6">
                                {projects.length === 0 ? 'Create your first project to start training AI models' : 'Try a different search term'}
                            </p>
                            {projects.length === 0 && (
                                <button
                                    onClick={() => setShowModal(true)}
                                    className="px-6 py-2.5 bg-gradient-to-r from-primary-600 to-primary-500 text-white rounded-xl text-sm font-medium shadow-md"
                                >
                                    <Plus className="w-4 h-4 inline mr-2" /> Create Project
                                </button>
                            )}
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {filtered.map((project, i) => (
                                <div
                                    key={project.id}
                                    onClick={() => navigate(`/project/${project.id}`)}
                                    className="bg-white rounded-2xl border border-surface-100 p-5 hover:shadow-lg hover:border-primary-200 transition-all cursor-pointer group animate-fade-in"
                                    style={{ animationDelay: `${i * 50}ms` }}
                                >
                                    <div className="flex items-start justify-between mb-3">
                                        <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${USE_COLORS[project.intended_use] || USE_COLORS.custom} flex items-center justify-center text-white`}>
                                            {USE_ICONS[project.intended_use] || USE_ICONS.custom}
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${project.status === 'trained' ? 'bg-emerald-50 text-emerald-600' :
                                                project.status === 'training' ? 'bg-amber-50 text-amber-600' :
                                                    'bg-surface-100 text-surface-600'
                                                }`}>
                                                {project.status || 'created'}
                                            </span>
                                            <button
                                                onClick={(e) => deleteProject(project.id, e)}
                                                className="opacity-0 group-hover:opacity-100 p-1 text-surface-400 hover:text-danger-500 transition-all"
                                            >
                                                <Trash2 className="w-3.5 h-3.5" />
                                            </button>
                                        </div>
                                    </div>
                                    <h3 className="font-semibold text-surface-900 mb-1">{project.name}</h3>
                                    <p className="text-sm text-surface-500 line-clamp-2 mb-3">{project.description || 'No description'}</p>
                                    <div className="flex items-center justify-between text-xs text-surface-400">
                                        <div className="flex items-center gap-3">
                                            <span className="flex items-center gap-1"><Database className="w-3 h-3" /> {project.dataset_count ?? 0} files</span>
                                            <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {project.updated_at ? new Date(project.updated_at).toLocaleDateString() : '—'}</span>
                                        </div>
                                        <ChevronRight className="w-4 h-4 group-hover:text-primary-500 transition-colors" />
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </main>

            {/* New Project Modal */}
            {showModal && (
                <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={() => setShowModal(false)}>
                    <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg p-6 animate-fade-in" onClick={(e) => e.stopPropagation()}>
                        <h2 className="text-xl font-bold text-surface-900 mb-6 flex items-center gap-2">
                            <Sparkles className="w-5 h-5 text-primary-500" /> New Project
                        </h2>
                        <form onSubmit={createProject} className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-surface-700 mb-1.5">Project Name</label>
                                <input
                                    id="project-name"
                                    type="text"
                                    value={newName}
                                    onChange={(e) => setNewName(e.target.value)}
                                    className="w-full px-4 py-2.5 border border-surface-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-500"
                                    placeholder="My AI Assistant"
                                    required
                                    maxLength={100}
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-surface-700 mb-1.5">Description</label>
                                <textarea
                                    id="project-desc"
                                    value={newDesc}
                                    onChange={(e) => setNewDesc(e.target.value)}
                                    className="w-full px-4 py-2.5 border border-surface-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-500 resize-none"
                                    rows={3}
                                    placeholder="What will this model do?"
                                    maxLength={500}
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-surface-700 mb-1.5">Intended Use</label>
                                <div className="grid grid-cols-2 gap-2">
                                    {[
                                        { value: 'chatbot', label: 'Chatbot', icon: '🤖' },
                                        { value: 'code', label: 'Code Generation', icon: '💻' },
                                        { value: 'qa', label: 'Q&A', icon: '❓' },
                                        { value: 'custom', label: 'Custom', icon: '✨' },
                                    ].map((use) => (
                                        <button
                                            key={use.value}
                                            type="button"
                                            onClick={() => setNewUse(use.value)}
                                            className={`flex items-center gap-2 px-3 py-2.5 rounded-xl text-sm border transition-all ${newUse === use.value
                                                ? 'border-primary-500 bg-primary-50 text-primary-700 font-medium'
                                                : 'border-surface-200 text-surface-600 hover:border-surface-300'
                                                }`}
                                        >
                                            <span>{use.icon}</span> {use.label}
                                        </button>
                                    ))}
                                </div>
                            </div>
                            <div className="flex gap-3 pt-2">
                                <button
                                    type="button"
                                    onClick={() => { setShowModal(false); setNewName(''); setNewDesc(''); setNewUse('custom'); }}
                                    className="flex-1 py-2.5 border border-surface-200 rounded-xl text-sm text-surface-600 hover:bg-surface-50 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    id="create-project-submit"
                                    type="submit"
                                    disabled={creating || !newName.trim()}
                                    className="flex-1 py-2.5 bg-gradient-to-r from-primary-600 to-primary-500 text-white rounded-xl text-sm font-medium shadow-md disabled:opacity-50 transition-all"
                                >
                                    {creating ? 'Creating...' : 'Create Project'}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}

            {/* Confirm Dialog */}
            <ConfirmDialog
                open={deleteTarget !== null}
                title="Delete Project"
                message="This will permanently delete the project and all its datasets, training runs, and models. This action cannot be undone."
                confirmLabel="Delete Project"
                variant="danger"
                onConfirm={confirmDelete}
                onCancel={() => setDeleteTarget(null)}
            />
        </div>
    );
}
