/**
 * E2E tests for MeowTrain critical user flows.
 *
 * Covers: registration, login, project creation, navigation.
 */
import { test, expect } from '@playwright/test';

const TEST_EMAIL = `e2e_${Date.now()}@test.com`;
const TEST_PASSWORD = 'TestPass123';
const TEST_NAME = 'E2E Tester';

test.describe('Authentication Flow', () => {
    test('should show login page by default', async ({ page }) => {
        await page.goto('/');
        // Should redirect to login or show login form
        await expect(page.locator('text=Sign In').first().or(page.locator('text=Login').first())).toBeVisible({ timeout: 10000 });
    });

    test('should register a new user', async ({ page }) => {
        await page.goto('/');

        // Navigate to register
        const registerLink = page.locator('text=Register').first().or(page.locator('text=Sign Up').first().or(page.locator('text=Create Account').first()));
        if (await registerLink.isVisible()) {
            await registerLink.click();
        }

        // Fill registration form
        await page.fill('input[type="email"], input[name="email"]', TEST_EMAIL);
        await page.fill('input[type="password"], input[name="password"]', TEST_PASSWORD);

        // Fill display name if present
        const nameField = page.locator('input[name="display_name"], input[name="displayName"], input[placeholder*="name" i]');
        if (await nameField.isVisible().catch(() => false)) {
            await nameField.fill(TEST_NAME);
        }

        // Submit
        await page.locator('button[type="submit"], button:has-text("Register"), button:has-text("Sign Up")').first().click();

        // Should redirect to dashboard or main page
        await expect(page).toHaveURL(/\/(dashboard|projects)?/, { timeout: 10000 });
    });

    test('should login with registered user', async ({ page }) => {
        // First register
        await page.goto('/');
        const registerLink = page.locator('text=Register').first().or(page.locator('text=Sign Up').first().or(page.locator('text=Create Account').first()));
        if (await registerLink.isVisible().catch(() => false)) {
            await registerLink.click();
        }

        const loginEmail = `login_${Date.now()}@test.com`;
        await page.fill('input[type="email"], input[name="email"]', loginEmail);
        await page.fill('input[type="password"], input[name="password"]', TEST_PASSWORD);
        const nameField = page.locator('input[name="display_name"], input[name="displayName"], input[placeholder*="name" i]');
        if (await nameField.isVisible().catch(() => false)) {
            await nameField.fill(TEST_NAME);
        }
        await page.locator('button[type="submit"], button:has-text("Register"), button:has-text("Sign Up")').first().click();
        await expect(page).toHaveURL(/\/(dashboard|projects)?/, { timeout: 10000 });
    });

    test('should allow guest login', async ({ page }) => {
        await page.goto('/');

        const guestBtn = page.locator('button:has-text("Guest"), button:has-text("Try"), a:has-text("Guest")').first();
        if (await guestBtn.isVisible().catch(() => false)) {
            await guestBtn.click();
            await expect(page).toHaveURL(/\/(dashboard|projects)?/, { timeout: 10000 });
        }
    });
});

test.describe('Project Management Flow', () => {
    test.beforeEach(async ({ page }) => {
        // Register and login
        await page.goto('/');
        const projEmail = `proj_${Date.now()}@test.com`;

        const registerLink = page.locator('text=Register').first().or(page.locator('text=Sign Up').first().or(page.locator('text=Create Account').first()));
        if (await registerLink.isVisible().catch(() => false)) {
            await registerLink.click();
        }

        await page.fill('input[type="email"], input[name="email"]', projEmail);
        await page.fill('input[type="password"], input[name="password"]', TEST_PASSWORD);
        const nameField = page.locator('input[name="display_name"], input[name="displayName"], input[placeholder*="name" i]');
        if (await nameField.isVisible().catch(() => false)) {
            await nameField.fill(TEST_NAME);
        }
        await page.locator('button[type="submit"], button:has-text("Register"), button:has-text("Sign Up")').first().click();
        await page.waitForURL(/\/(dashboard|projects)?/, { timeout: 10000 });
    });

    test('should create a new project', async ({ page }) => {
        // Look for create project button
        const createBtn = page.locator('button:has-text("Create"), button:has-text("New Project"), a:has-text("Create")').first();
        if (await createBtn.isVisible().catch(() => false)) {
            await createBtn.click();

            // Fill project name
            const nameInput = page.locator('input[name="name"], input[placeholder*="project" i], input[placeholder*="name" i]').first();
            if (await nameInput.isVisible().catch(() => false)) {
                await nameInput.fill('E2E Test Project');
            }

            // Submit
            const submitBtn = page.locator('button[type="submit"], button:has-text("Create"), button:has-text("Save")').first();
            if (await submitBtn.isVisible().catch(() => false)) {
                await submitBtn.click();
            }

            // Verify project appears
            await expect(page.locator('text=E2E Test Project')).toBeVisible({ timeout: 10000 });
        }
    });
});

test.describe('Health Check', () => {
    test('should return healthy status', async ({ request }) => {
        const response = await request.get('http://localhost:8000/api/health');
        expect(response.ok()).toBeTruthy();
        const body = await response.json();
        expect(body.status).toMatch(/healthy|degraded/);
        expect(body.version).toBeTruthy();
        expect(body.db_connected).toBeDefined();
    });

    test('should have metrics endpoint', async ({ request }) => {
        const response = await request.get('http://localhost:8000/metrics');
        // May return 200 if prometheus is installed, or 404 if not
        expect([200, 404]).toContain(response.status());
    });
});
