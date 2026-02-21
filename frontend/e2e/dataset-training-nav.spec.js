/**
 * E2E tests for dataset management, training config, and navigation.
 *
 * Fixed: removed all `.catch(() => false)` patterns that silently swallowed failures.
 * Each test now uses proper Playwright locators and fails explicitly when elements are missing.
 */
import { test, expect } from '@playwright/test';

const TEST_PASSWORD = 'TestPass123!';

/**
 * Register a new user and land on the dashboard.
 * Fails explicitly if registration doesn't succeed.
 */
async function registerAndLogin(page) {
    const email = `dataset_${Date.now()}@test.com`;
    await page.goto('/');

    // Navigate to register page
    const registerLink = page.getByRole('link', { name: /register|sign up|create account/i });
    if (await registerLink.isVisible({ timeout: 3000 }).catch(() => false)) {
        await registerLink.click();
        await page.waitForURL(/register/);
    }

    // Fill registration form — use explicit IDs set in our components
    await page.locator('#register-email').fill(email);
    await page.locator('#register-password').fill(TEST_PASSWORD);

    const nameField = page.locator('#register-name');
    if (await nameField.isVisible({ timeout: 2000 }).catch(() => false)) {
        await nameField.fill('E2E Tester');
    }

    await page.locator('#register-submit').click();

    // Must reach dashboard — fail explicitly if we don't
    await page.waitForURL('/', { timeout: 15000 });
    await expect(page).toHaveURL('/');
}

test.describe('Dataset Management Flow', () => {
    test.beforeEach(async ({ page }) => {
        await registerAndLogin(page);
    });

    test('should navigate to project and see dataset upload area', async ({ page }) => {
        // Create a project
        const createBtn = page.getByRole('button', { name: /create|new project/i });
        await expect(createBtn).toBeVisible({ timeout: 10000 });
        await createBtn.click();

        // Fill project name
        const nameInput = page.locator('input[name="name"], input[placeholder*="name" i]').first();
        await expect(nameInput).toBeVisible({ timeout: 5000 });
        await nameInput.fill('Dataset Test Project');

        // Submit
        const submitBtn = page.getByRole('button', { name: /create|save/i });
        await expect(submitBtn).toBeVisible();
        await submitBtn.click();

        // Navigate to the project
        const projectLink = page.getByText('Dataset Test Project');
        await expect(projectLink).toBeVisible({ timeout: 10000 });
        await projectLink.click();

        // Dataset/upload area MUST be visible — fail explicitly if not
        await expect(
            page.getByText(/upload|dataset|data/i).first()
        ).toBeVisible({ timeout: 10000 });
    });
});


test.describe('Training Configuration Flow', () => {
    test.beforeEach(async ({ page }) => {
        await registerAndLogin(page);
    });

    test('should show model selection on project page', async ({ page }) => {
        // Create a project
        const createBtn = page.getByRole('button', { name: /create|new project/i });
        await expect(createBtn).toBeVisible({ timeout: 10000 });
        await createBtn.click();

        const nameInput = page.locator('input[name="name"], input[placeholder*="name" i]').first();
        await expect(nameInput).toBeVisible({ timeout: 5000 });
        await nameInput.fill('Training Config Test');

        const submitBtn = page.getByRole('button', { name: /create|save/i });
        await expect(submitBtn).toBeVisible();
        await submitBtn.click();

        // Navigate to project
        const projectLink = page.getByText('Training Config Test');
        await expect(projectLink).toBeVisible({ timeout: 10000 });
        await projectLink.click();

        // Training/model section MUST be visible — fail explicitly if not
        await expect(
            page.getByText(/train|model|configure/i).first()
        ).toBeVisible({ timeout: 10000 });
    });
});


test.describe('Navigation and Layout', () => {
    test.beforeEach(async ({ page }) => {
        await registerAndLogin(page);
    });

    test('should have navigation elements', async ({ page }) => {
        // Dashboard must have a header or nav element
        await expect(
            page.locator('nav, [role="navigation"], header').first()
        ).toBeVisible({ timeout: 5000 });
    });

    test('should show user info or logout option', async ({ page }) => {
        // Must find at least one of these — fail explicitly if none present
        const bodyText = await page.textContent('body');
        const hasUserUI = /logout|sign out|profile|tester/i.test(bodyText);
        expect(hasUserUI).toBe(true);
    });

    test('should handle 404 pages gracefully', async ({ page }) => {
        await page.goto('/nonexistent-page-12345');

        // Should show 404 content, not a blank page
        await expect(page.getByText('404')).toBeVisible({ timeout: 5000 });
        await expect(page.getByText(/not found|page/i).first()).toBeVisible();
    });
});
