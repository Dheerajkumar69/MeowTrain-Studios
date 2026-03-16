/**
 * ConfirmDialog tests — rendering, ARIA, focus trap, keyboard interactions
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ConfirmDialog from '../components/ConfirmDialog';

describe('ConfirmDialog', () => {
    const defaultProps = {
        open: true,
        title: 'Delete item?',
        message: 'This cannot be undone.',
        confirmLabel: 'Delete',
        cancelLabel: 'Cancel',
        variant: 'danger',
        onConfirm: vi.fn(),
        onCancel: vi.fn(),
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders nothing when open=false', () => {
        const { container } = render(<ConfirmDialog {...defaultProps} open={false} />);
        expect(container.innerHTML).toBe('');
    });

    it('renders title, message, and buttons when open', () => {
        render(<ConfirmDialog {...defaultProps} />);
        expect(screen.getByText('Delete item?')).toBeInTheDocument();
        expect(screen.getByText('This cannot be undone.')).toBeInTheDocument();
        expect(screen.getByText('Delete')).toBeInTheDocument();
        expect(screen.getByText('Cancel')).toBeInTheDocument();
    });

    it('has proper ARIA attributes', () => {
        render(<ConfirmDialog {...defaultProps} />);
        const dialog = screen.getByRole('dialog');
        expect(dialog).toHaveAttribute('aria-modal', 'true');
        expect(dialog).toHaveAttribute('aria-labelledby', 'confirm-dialog-title');
        expect(dialog).toHaveAttribute('aria-describedby', 'confirm-dialog-message');
    });

    it('calls onConfirm when confirm button clicked', async () => {
        const user = userEvent.setup();
        render(<ConfirmDialog {...defaultProps} />);
        await user.click(screen.getByText('Delete'));
        expect(defaultProps.onConfirm).toHaveBeenCalledOnce();
    });

    it('calls onCancel when cancel button clicked', async () => {
        const user = userEvent.setup();
        render(<ConfirmDialog {...defaultProps} />);
        await user.click(screen.getByText('Cancel'));
        expect(defaultProps.onCancel).toHaveBeenCalledOnce();
    });

    it('calls onCancel when close (X) button clicked', async () => {
        const user = userEvent.setup();
        render(<ConfirmDialog {...defaultProps} />);
        await user.click(screen.getByLabelText('Close dialog'));
        expect(defaultProps.onCancel).toHaveBeenCalledOnce();
    });

    it('calls onCancel on backdrop click', async () => {
        const user = userEvent.setup();
        render(<ConfirmDialog {...defaultProps} />);
        // Click on the backdrop (the dialog role div itself is the overlay)
        await user.click(screen.getByRole('dialog'));
        expect(defaultProps.onCancel).toHaveBeenCalledOnce();
    });

    it('calls onCancel on Escape key', async () => {
        const user = userEvent.setup();
        render(<ConfirmDialog {...defaultProps} />);
        // Focus an element inside the dialog first so the keydown bubbles up to the overlay
        screen.getByText('Delete').focus();
        await user.keyboard('{Escape}');
        expect(defaultProps.onCancel).toHaveBeenCalledOnce();
    });

    it('auto-focuses the confirm button via data-autofocus', async () => {
        render(<ConfirmDialog {...defaultProps} />);
        await waitFor(() => {
            const confirmBtn = screen.getByText('Delete');
            expect(confirmBtn).toHaveAttribute('data-autofocus');
        });
    });

    it('traps focus: Tab from last element wraps to first', async () => {
        const user = userEvent.setup();
        render(<ConfirmDialog {...defaultProps} />);

        // Wait for autofocus
        await waitFor(() => {
            expect(document.activeElement).not.toBe(document.body);
        });

        // Get all buttons in the dialog
        const closeBtn = screen.getByLabelText('Close dialog');
        screen.getByText('Cancel');  // Assert it exists
        const confirmBtn = screen.getByText('Delete');

        // Focus the last focusable element (confirm button)
        confirmBtn.focus();
        expect(document.activeElement).toBe(confirmBtn);

        // Tab should wrap to first focusable (close button)
        await user.tab();
        expect(document.activeElement).toBe(closeBtn);
    });

    it('traps focus: Shift+Tab from first element wraps to last', async () => {
        const user = userEvent.setup();
        render(<ConfirmDialog {...defaultProps} />);

        await waitFor(() => {
            expect(document.activeElement).not.toBe(document.body);
        });

        const closeBtn = screen.getByLabelText('Close dialog');
        const confirmBtn = screen.getByText('Delete');

        // Focus the first focusable element (close button)
        closeBtn.focus();
        expect(document.activeElement).toBe(closeBtn);

        // Shift+Tab should wrap to last focusable (confirm button)
        await user.tab({ shift: true });
        expect(document.activeElement).toBe(confirmBtn);
    });
});
