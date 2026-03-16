/* global process */
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
    testDir: './e2e',
    fullyParallel: false,
    forbidOnly: !!process.env.CI,
    retries: process.env.CI ? 2 : 0,
    workers: 1,
    reporter: process.env.CI ? 'html' : 'list',
    timeout: 30000,

    use: {
        baseURL: 'http://localhost:5173',
        trace: 'on-first-retry',
        screenshot: 'only-on-failure',
    },

    projects: [
        {
            name: 'chromium',
            use: { ...devices['Desktop Chrome'] },
        },
    ],

    webServer: [
        {
            command: 'cd ../backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000',
            port: 8000,
            reuseExistingServer: !process.env.CI,
            timeout: 30000,
            env: {
                MEOWLLM_JWT_SECRET: 'e2e-test-secret-key-not-for-prod',
            },
        },
        {
            command: 'npm run dev -- --host 0.0.0.0 --port 5173',
            port: 5173,
            reuseExistingServer: !process.env.CI,
            timeout: 30000,
        },
    ],
});
