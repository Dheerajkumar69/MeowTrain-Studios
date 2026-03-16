import { Database, FolderOpen, Cpu, Cat } from 'lucide-react';

// ===== Skeleton primitives =====
function SkeletonLine({ className = '' }) {
    return <div className={`h-4 bg-surface-200 rounded-md animate-pulse ${className}`} />;
}

function SkeletonBlock({ className = '' }) {
    return <div className={`bg-surface-200 rounded-xl animate-pulse ${className}`} />;
}

// ===== Card Skeleton (for project cards, dataset cards, etc.) =====
export function CardSkeleton({ count = 3 }) {
    return (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: count }).map((_, i) => (
                <div key={i} className="bg-white rounded-2xl border border-surface-200 p-5 space-y-4">
                    <div className="flex items-center gap-3">
                        <SkeletonBlock className="w-10 h-10 rounded-xl" />
                        <div className="flex-1 space-y-2">
                            <SkeletonLine className="w-3/4" />
                            <SkeletonLine className="w-1/2 h-3" />
                        </div>
                    </div>
                    <SkeletonLine className="w-full" />
                    <SkeletonLine className="w-2/3" />
                </div>
            ))}
        </div>
    );
}

// ===== Table Skeleton (for datasets, training runs) =====
export function TableSkeleton({ rows = 4, cols = 4 }) {
    return (
        <div className="bg-white rounded-2xl border border-surface-200 overflow-hidden">
            <div className="grid gap-0 divide-y divide-surface-100">
                {Array.from({ length: rows }).map((_, i) => (
                    <div key={i} className="flex items-center gap-4 px-5 py-4">
                        {Array.from({ length: cols }).map((_, j) => (
                            <SkeletonLine key={j} className={`flex-1 ${j === 0 ? 'w-1/3' : 'w-full'}`} />
                        ))}
                    </div>
                ))}
            </div>
        </div>
    );
}

// ===== Dashboard Skeleton =====
export function DashboardSkeleton() {
    return (
        <div className="space-y-6 p-6 animate-pulse">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div className="space-y-2">
                    <SkeletonLine className="w-48 h-6" />
                    <SkeletonLine className="w-32 h-3" />
                </div>
                <SkeletonBlock className="w-36 h-10 rounded-xl" />
            </div>
            {/* Hardware bars */}
            <div className="grid grid-cols-4 gap-4">
                {[1, 2, 3, 4].map(i => (
                    <SkeletonBlock key={i} className="h-20 rounded-2xl" />
                ))}
            </div>
            {/* Project cards */}
            <CardSkeleton count={3} />
        </div>
    );
}

// ===== Empty states =====
// eslint-disable-next-line no-unused-vars
export function EmptyState({ icon: Icon = Cat, title, description, action }) {
    return (
        <div className="flex flex-col items-center justify-center py-16 text-center">
            <div className="w-16 h-16 bg-surface-100 rounded-2xl flex items-center justify-center mb-4">
                <Icon className="w-8 h-8 text-surface-400" />
            </div>
            <h3 className="text-lg font-semibold text-surface-700 mb-1">{title}</h3>
            <p className="text-sm text-surface-500 max-w-xs mb-4">{description}</p>
            {action}
        </div>
    );
}

export function EmptyProjects({ onCreate }) {
    return (
        <EmptyState
            icon={FolderOpen}
            title="No projects yet"
            description="Create your first project to start fine-tuning a language model."
            action={
                <button onClick={onCreate} className="px-4 py-2.5 bg-primary-600 text-white rounded-xl text-sm font-medium hover:bg-primary-700 transition-colors">
                    Create Project
                </button>
            }
        />
    );
}

export function EmptyDatasets({ onUpload }) {
    return (
        <EmptyState
            icon={Database}
            title="No datasets uploaded"
            description="Upload training data in TXT, JSON, CSV, PDF, or DOCX format."
            action={onUpload && (
                <button onClick={onUpload} className="px-4 py-2.5 bg-primary-600 text-white rounded-xl text-sm font-medium hover:bg-primary-700 transition-colors">
                    Upload Dataset
                </button>
            )}
        />
    );
}

export function EmptyRuns() {
    return (
        <EmptyState
            icon={Cpu}
            title="No training runs"
            description="Configure and start training to see run history here."
        />
    );
}

export { SkeletonLine, SkeletonBlock };
