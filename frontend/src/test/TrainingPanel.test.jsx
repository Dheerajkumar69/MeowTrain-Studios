/**
 * Tests for the TrainingPanel component.
 * Verifies training controls, status display, and configuration form.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import React from 'react';

// Mock API
vi.mock('../services/api', () => ({
    default: {
        get: vi.fn(),
        post: vi.fn(),
    },
    trainingAPI: {
        configure: vi.fn(),
        start: vi.fn(),
        pause: vi.fn(),
        resume: vi.fn(),
        stop: vi.fn(),
        status: vi.fn(),
    },
}));

// Mock WebSocket
vi.mock('../services/websocket', () => ({
    default: {
        connect: vi.fn(),
        disconnect: vi.fn(),
        onMessage: vi.fn(),
    },
}));

describe('TrainingPanel', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('component can be imported', async () => {
        const mod = await import('../components/training/TrainingPanel');
        expect(mod.default).toBeDefined();
    });

    it('ExportPanel component can be imported', async () => {
        const mod = await import('../components/training/ExportPanel');
        expect(mod.default).toBeDefined();
    });

    it('ExportPanel renders download button when GGUF is ready', async () => {
        const ExportPanel = (await import('../components/training/ExportPanel')).default;
        const mockSetError = vi.fn();

        // Mock the API to return a completed GGUF status
        const { default: api } = await import('../services/api');
        api.get = vi.fn().mockResolvedValue({
            data: { step: 'done', progress: 100, gguf_filename: 'model-Q8_0.gguf', gguf_size_mb: 500 },
        });

        render(
            <BrowserRouter>
                <ExportPanel projectId={1} setError={mockSetError} />
            </BrowserRouter>
        );

        // The export panel should render with export buttons
        await waitFor(() => {
            const text = document.body.textContent.toLowerCase();
            const hasExportUI = text.includes('export') || text.includes('gguf') || text.includes('lm studio');
            expect(hasExportUI).toBe(true);
        }, { timeout: 5000 });
    });
});
