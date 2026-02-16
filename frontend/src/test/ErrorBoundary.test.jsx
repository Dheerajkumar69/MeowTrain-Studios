/**
 * ErrorBoundary tests — catches render errors and shows fallback UI
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ErrorBoundary from '../components/ErrorBoundary';

function ThrowingChild({ shouldThrow = true }) {
    if (shouldThrow) throw new Error('kaboom');
    return <span data-testid="child">OK</span>;
}

describe('ErrorBoundary', () => {
    afterEach(() => vi.restoreAllMocks());

    it('renders children when no error', () => {
        render(
            <ErrorBoundary>
                <ThrowingChild shouldThrow={false} />
            </ErrorBoundary>,
        );
        expect(screen.getByTestId('child')).toBeInTheDocument();
    });

    it('renders full fallback UI on error', () => {
        vi.spyOn(console, 'error').mockImplementation(() => {});
        render(
            <ErrorBoundary>
                <ThrowingChild />
            </ErrorBoundary>,
        );
        expect(screen.getByRole('alert')).toBeInTheDocument();
        expect(screen.getByText('Something went wrong')).toBeInTheDocument();
        expect(screen.getByText('kaboom')).toBeInTheDocument();
        expect(screen.getByText('Try Again')).toBeInTheDocument();
    });

    it('renders compact fallback when compact prop is true', () => {
        vi.spyOn(console, 'error').mockImplementation(() => {});
        render(
            <ErrorBoundary compact label="Widget">
                <ThrowingChild />
            </ErrorBoundary>,
        );
        expect(screen.getByRole('alert')).toBeInTheDocument();
        expect(screen.getByText('Widget crashed')).toBeInTheDocument();
        expect(screen.getByText('Retry')).toBeInTheDocument();
    });

    it('recovers when Try Again is clicked and child no longer throws', async () => {
        vi.spyOn(console, 'error').mockImplementation(() => {});
        const user = userEvent.setup();

        // Use a ref-like pattern that persists across renders
        const state = { shouldThrow: true };
        function Child() {
            if (state.shouldThrow) throw new Error('kaboom');
            return <span data-testid="child">OK</span>;
        }

        render(
            <ErrorBoundary>
                <Child />
            </ErrorBoundary>,
        );
        expect(screen.getByText('Something went wrong')).toBeInTheDocument();

        // Stop throwing so retry succeeds
        state.shouldThrow = false;
        await user.click(screen.getByText('Try Again'));
        expect(screen.getByTestId('child')).toBeInTheDocument();
    });
});
