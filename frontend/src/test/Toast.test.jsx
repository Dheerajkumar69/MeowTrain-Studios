/**
 * Toast system tests — rendering, auto-dismiss, toast types, dismiss button
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ToastProvider, useToast } from '../components/Toast';

// Helper component that triggers toasts via buttons
function ToastTrigger() {
    const toast = useToast();
    return (
        <div>
            <button onClick={() => toast.success('Success msg')}>success</button>
            <button onClick={() => toast.error('Error msg')}>error</button>
            <button onClick={() => toast.warning('Warning msg')}>warning</button>
            <button onClick={() => toast.info('Info msg')}>info</button>
        </div>
    );
}

function renderWithToast() {
    return render(
        <ToastProvider>
            <ToastTrigger />
        </ToastProvider>,
    );
}

describe('Toast', () => {
    beforeEach(() => {
        vi.useFakeTimers({ shouldAdvanceTime: true });
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('renders a success toast', async () => {
        const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
        renderWithToast();
        await user.click(screen.getByText('success'));
        expect(screen.getByText('Success msg')).toBeInTheDocument();
        expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    it('renders an error toast', async () => {
        const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
        renderWithToast();
        await user.click(screen.getByText('error'));
        expect(screen.getByText('Error msg')).toBeInTheDocument();
    });

    it('shows all four toast types simultaneously', async () => {
        const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
        renderWithToast();
        await user.click(screen.getByText('success'));
        await user.click(screen.getByText('error'));
        await user.click(screen.getByText('warning'));
        await user.click(screen.getByText('info'));
        expect(screen.getAllByRole('alert')).toHaveLength(4);
    });

    it('auto-dismisses after default duration', async () => {
        const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
        renderWithToast();
        await user.click(screen.getByText('success'));
        expect(screen.getByText('Success msg')).toBeInTheDocument();

        // Default duration is 4000ms
        act(() => { vi.advanceTimersByTime(4100); });

        expect(screen.queryByText('Success msg')).not.toBeInTheDocument();
    });

    it('error toasts have longer default (6000ms)', async () => {
        const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
        renderWithToast();
        await user.click(screen.getByText('error'));

        // Still visible after 4.1s
        act(() => { vi.advanceTimersByTime(4100); });
        expect(screen.getByText('Error msg')).toBeInTheDocument();

        // Gone after 6.1s total
        act(() => { vi.advanceTimersByTime(2000); });
        expect(screen.queryByText('Error msg')).not.toBeInTheDocument();
    });

    it('useToast() throws when used outside ToastProvider', () => {
        const spy = vi.spyOn(console, 'error').mockImplementation(() => { });
        function Bad() { useToast(); return null; }
        expect(() => render(<Bad />)).toThrow('useToast must be used within ToastProvider');
        spy.mockRestore();
    });
});
