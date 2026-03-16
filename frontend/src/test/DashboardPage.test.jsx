/**
 * Tests for the Dashboard page component.
 * Verifies rendering, project listing, navigation, and empty states.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import React, { Suspense } from 'react';

// Mock the API module
vi.mock('../services/api', () => ({
    default: {
        get: vi.fn(),
        post: vi.fn(),
        delete: vi.fn(),
    },
    projectsAPI: {
        list: vi.fn(),
        create: vi.fn(),
        delete: vi.fn(),
    },
    hardwareAPI: {
        status: vi.fn().mockResolvedValue({ data: {} }),
        deviceInfo: vi.fn().mockResolvedValue({ data: { device: 'cpu', gpu_name: null } }),
        refreshDevice: vi.fn().mockResolvedValue({ data: {} }),
    },
    trainingAPI: {
        status: vi.fn().mockResolvedValue({ data: { status: 'idle' } }),
    },
    modelsAPI: {
        list: vi.fn().mockResolvedValue({ data: [] }),
    },
}));

// Mock the auth context
vi.mock('../contexts/AuthContext', () => ({
    useAuth: () => ({
        user: { id: 1, email: 'test@test.com', display_name: 'Test User', is_guest: false },
        isAuthenticated: true,
        logout: vi.fn(),
    }),
}));

// Mock Toast context so DashboardPage's useToast() doesn't throw
vi.mock('../components/Toast', () => ({
    useToast: () => ({ addToast: vi.fn() }),
    ToastProvider: ({ children }) => children,
}));

// Lazy import the page (same as App.jsx does)
const DashboardPage = React.lazy(() => import('../pages/DashboardPage'));

function renderWithRouter(ui) {
    return render(
        <BrowserRouter>
            <Suspense fallback={<div>Loading...</div>}>
                {ui}
            </Suspense>
        </BrowserRouter>
    );
}

describe('DashboardPage', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders without crashing', async () => {
        const { projectsAPI } = await import('../services/api');
        projectsAPI.list.mockResolvedValue({ data: { items: [], total: 0 } });

        renderWithRouter(<DashboardPage />);

        // The page should render some content (loading or dashboard)
        await waitFor(() => {
            expect(document.body.textContent.length).toBeGreaterThan(0);
        });
    });

    it('shows empty state when no projects exist', async () => {
        const { projectsAPI } = await import('../services/api');
        projectsAPI.list.mockResolvedValue({ data: { items: [], total: 0 } });

        renderWithRouter(<DashboardPage />);

        await waitFor(() => {
            // Should indicate no projects or show a create button
            const text = document.body.textContent.toLowerCase();
            const hasEmptyIndicator = text.includes('create') || text.includes('no project') || text.includes('get started');
            expect(hasEmptyIndicator).toBe(true);
        }, { timeout: 5000 });
    });

    it('displays projects when they exist', async () => {
        const { projectsAPI } = await import('../services/api');
        projectsAPI.list.mockResolvedValue({
            data: {
                items: [
                    { id: 1, name: 'Test Project', description: 'A test', status: 'created', created_at: new Date().toISOString(), updated_at: new Date().toISOString(), user_id: 1, intended_use: 'custom', dataset_count: 0, model_config_count: 0 },
                ],
                total: 1,
            },
        });

        renderWithRouter(<DashboardPage />);

        await waitFor(() => {
            expect(document.body.textContent).toContain('Test Project');
        }, { timeout: 5000 });
    });
});
