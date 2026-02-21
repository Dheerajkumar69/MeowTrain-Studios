"""
Tests for health, hardware, inference, augmentation, LM Studio,
model-export, and the request-ID correlation middleware.

Fills the coverage gaps identified in issue #19.
"""

import io
import json
from unittest.mock import patch, MagicMock

import pytest

# Heavy ML packages are optional in test environments
try:
    import datasets as _hf_datasets  # noqa: F401
    _has_ml_deps = True
except ImportError:
    _has_ml_deps = False

_skip_no_ml = pytest.mark.skipif(not _has_ml_deps, reason="ML deps (datasets) not installed")


# ──────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_has_request_id_header(self, client):
        resp = client.get("/api/health")
        assert "x-request-id" in resp.headers


# ──────────────────────────────────────────────────────────────
# Hardware
# ──────────────────────────────────────────────────────────────

class TestHardwareEndpoint:
    def test_hardware_status(self, client, auth_headers):
        resp = client.get("/api/hardware/", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "cpu_name" in data
        assert "ram_total_gb" in data
        assert isinstance(data["gpu_available"], bool)

    def test_hardware_returns_numeric_values(self, client, auth_headers):
        data = client.get("/api/hardware/", headers=auth_headers).json()
        assert isinstance(data["cpu_cores"], int)
        assert isinstance(data["ram_total_gb"], (int, float))
        assert isinstance(data["disk_total_gb"], (int, float))


# ──────────────────────────────────────────────────────────────
# Inference — chat & context
# ──────────────────────────────────────────────────────────────

class TestInferenceChat:
    def test_chat_no_model_available(self, client, auth_headers, project_id):
        """Chat should error gracefully when no model is trained."""
        resp = client.post(f"/api/projects/{project_id}/chat", json={
            "prompt": "Hello, world!",
        }, headers=auth_headers)
        # Expect 400 (no trained model) or 500 — either way, not a crash
        assert resp.status_code in (400, 500)
        assert "detail" in resp.json()

    def test_chat_prompt_too_short(self, client, auth_headers, project_id):
        """Pydantic enforces min_length=1."""
        resp = client.post(f"/api/projects/{project_id}/chat", json={
            "prompt": "",
        }, headers=auth_headers)
        assert resp.status_code == 422

    def test_chat_unauthenticated(self, client, project_id):
        resp = client.post(f"/api/projects/{project_id}/chat", json={
            "prompt": "hi",
        })
        assert resp.status_code in (401, 422)


class TestInferenceContext:
    def test_get_context_empty(self, client, auth_headers, project_id):
        """Context on a fresh project should still return valid shape."""
        resp = client.get(f"/api/projects/{project_id}/chat/context", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "datasets" in data
        assert isinstance(data["datasets"], list)

    def test_get_context_unauthenticated(self, client, project_id):
        resp = client.get(f"/api/projects/{project_id}/chat/context")
        assert resp.status_code in (401, 422)


# ──────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────

class TestPromptTemplates:
    def test_create_prompt(self, client, auth_headers, project_id):
        resp = client.post(f"/api/projects/{project_id}/prompts", json={
            "name": "My Prompt",
            "system_prompt": "You are a cat.",
            "user_prompt": "Meow?",
            "temperature": 0.5,
        }, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "My Prompt"
        assert data["temperature"] == 0.5

    def test_list_prompts_empty(self, client, auth_headers, project_id):
        resp = client.get(f"/api/projects/{project_id}/prompts", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_list_prompts_pagination(self, client, auth_headers, project_id):
        # Create 3 prompts
        for i in range(3):
            client.post(f"/api/projects/{project_id}/prompts", json={
                "name": f"Prompt {i}",
            }, headers=auth_headers)
        # Fetch with limit
        resp = client.get(f"/api/projects/{project_id}/prompts?page=1&per_page=2", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["total"] == 3

    def test_create_prompt_name_too_long(self, client, auth_headers, project_id):
        resp = client.post(f"/api/projects/{project_id}/prompts", json={
            "name": "x" * 101,
        }, headers=auth_headers)
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────────────────────

class TestAugmentation:
    def test_augment_no_datasets(self, client, auth_headers, project_id):
        resp = client.post(f"/api/projects/{project_id}/datasets/augment", json={
            "preview_only": True,
        }, headers=auth_headers)
        assert resp.status_code == 400
        assert "no ready datasets" in resp.json()["detail"].lower()

    @_skip_no_ml
    def test_augment_preview(self, client, auth_headers, project_id):
        """Upload a dataset then preview augmentation."""
        data_content = json.dumps([
            {"instruction": "What is AI?", "output": "Artificial Intelligence"},
            {"instruction": "What is ML?", "output": "Machine Learning"},
            {"instruction": "What is DL?", "output": "Deep Learning"},
        ])
        client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("train.json", io.BytesIO(data_content.encode()), "application/json")},
            headers=auth_headers,
        )
        resp = client.post(f"/api/projects/{project_id}/datasets/augment", json={
            "preview_only": True,
            "enable_dedup": True,
            "enable_clean": True,
            "enable_filter": True,
            "min_length": 5,
        }, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["preview_only"] is True
        assert "stats" in data
        assert "datasets" in data

    def test_augment_unauthenticated(self, client, project_id):
        resp = client.post(f"/api/projects/{project_id}/datasets/augment", json={
            "preview_only": True,
        })
        assert resp.status_code in (401, 422)


# ──────────────────────────────────────────────────────────────
# LM Studio
# ──────────────────────────────────────────────────────────────

class TestLMStudio:
    def test_get_config(self, client, auth_headers):
        resp = client.get("/api/lmstudio/config", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "host" in data
        assert "port" in data
        assert "enabled" in data

    def test_set_config(self, client, auth_headers):
        resp = client.put("/api/lmstudio/config", json={
            "host": "127.0.0.1",
            "port": 1234,
            "enabled": False,
        }, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "127.0.0.1" in data["host"]
        assert data["port"] == 1234

    def test_set_config_invalid_port(self, client, auth_headers):
        resp = client.put("/api/lmstudio/config", json={
            "port": 99999,  # exceeds max 65535
        }, headers=auth_headers)
        assert resp.status_code == 422

    def test_get_models_disabled(self, client, auth_headers):
        # Disable LM Studio
        client.put("/api/lmstudio/config", json={"enabled": False}, headers=auth_headers)
        resp = client.get("/api/lmstudio/models", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False

    def test_get_config_unauthenticated(self, client):
        resp = client.get("/api/lmstudio/config")
        assert resp.status_code == 401

    def test_set_config_unauthenticated(self, client):
        resp = client.put("/api/lmstudio/config", json={"enabled": False})
        assert resp.status_code == 401

    def test_guest_cannot_set_config(self, client):
        guest_resp = client.post("/api/auth/guest")
        guest_headers = {"Authorization": f"Bearer {guest_resp.json()['token']}"}
        resp = client.put("/api/lmstudio/config", json={"enabled": False}, headers=guest_headers)
        assert resp.status_code == 403


# ──────────────────────────────────────────────────────────────
# Model export
# ──────────────────────────────────────────────────────────────

class TestModelExport:
    def test_export_no_trained_model(self, client, auth_headers, project_id):
        resp = client.get(f"/api/models/export/{project_id}", headers=auth_headers)
        assert resp.status_code == 404
        assert "no trained model" in resp.json()["detail"].lower()

    def test_gguf_invalid_quantization(self, client, auth_headers, project_id):
        resp = client.post(f"/api/models/export/{project_id}/gguf?quantization=INVALID", headers=auth_headers)
        assert resp.status_code == 400
        assert "invalid quantization" in resp.json()["detail"].lower()

    def test_gguf_status_not_started(self, client, auth_headers, project_id):
        resp = client.get(f"/api/models/export/{project_id}/gguf/status", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "step" in data
        assert data["step"] in ("not_started", "completed")

    def test_gguf_download_no_file(self, client, auth_headers, project_id):
        resp = client.get(f"/api/models/export/{project_id}/gguf/download", headers=auth_headers)
        assert resp.status_code == 404

    def test_export_unauthenticated(self, client, project_id):
        resp = client.get(f"/api/models/export/{project_id}")
        assert resp.status_code in (401, 422)


# ──────────────────────────────────────────────────────────────
# Request-ID Middleware
# ──────────────────────────────────────────────────────────────

class TestRequestIDMiddleware:
    def test_generates_request_id(self, client):
        resp = client.get("/api/health")
        rid = resp.headers.get("x-request-id")
        assert rid is not None
        assert len(rid) > 0

    def test_passes_through_caller_id(self, client):
        """If the caller supplies X-Request-ID, the server echoes it back."""
        resp = client.get("/api/health", headers={"X-Request-ID": "my-trace-42"})
        assert resp.headers["x-request-id"] == "my-trace-42"

    def test_ids_unique_per_request(self, client):
        rid1 = client.get("/api/health").headers["x-request-id"]
        rid2 = client.get("/api/health").headers["x-request-id"]
        assert rid1 != rid2


# ──────────────────────────────────────────────────────────────
# Dataset preview-training
# ──────────────────────────────────────────────────────────────

class TestDatasetPreviewTraining:
    def test_preview_training_no_datasets(self, client, auth_headers, project_id):
        resp = client.post(f"/api/projects/{project_id}/datasets/preview-training", headers=auth_headers)
        assert resp.status_code == 400

    @_skip_no_ml
    def test_preview_training_with_data(self, client, auth_headers, project_id):
        data_content = json.dumps([
            {"instruction": "What is AI?", "output": "Artificial Intelligence"},
        ])
        client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("train.json", io.BytesIO(data_content.encode()), "application/json")},
            headers=auth_headers,
        )
        resp = client.post(f"/api/projects/{project_id}/datasets/preview-training", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "datasets" in data
        assert "summary" in data
        assert data["summary"]["total_examples"] >= 1


# ──────────────────────────────────────────────────────────────
# Role guards
# ──────────────────────────────────────────────────────────────

class TestRoleGuards:
    def test_guest_can_create_project(self, client):
        """Guests should be able to create projects (up to the cap)."""
        guest_resp = client.post("/api/auth/guest")
        token = guest_resp.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        resp = client.post("/api/projects/", json={"name": "Guest Project"}, headers=headers)
        assert resp.status_code == 200

    def test_guest_cannot_see_other_users_project(self, client, auth_headers, project_id):
        """Guest should get 404 on another user's project."""
        guest_resp = client.post("/api/auth/guest")
        guest_headers = {"Authorization": f"Bearer {guest_resp.json()['token']}"}

        resp = client.get(f"/api/projects/{project_id}", headers=guest_headers)
        assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────
# Dataset preview
# ──────────────────────────────────────────────────────────────

class TestDatasetPreview:
    def test_preview_uploaded_dataset(self, client, auth_headers, project_id):
        upload_resp = client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("test.txt", io.BytesIO(b"Hello world " * 100), "text/plain")},
            headers=auth_headers,
        )
        assert upload_resp.status_code == 200
        ds_id = upload_resp.json()["id"]

        resp = client.get(f"/api/projects/{project_id}/datasets/{ds_id}/preview", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["original_name"] == "test.txt"
        assert "chunks" in data
        assert len(data["chunks"]) > 0

    def test_preview_nonexistent_dataset(self, client, auth_headers, project_id):
        resp = client.get(f"/api/projects/{project_id}/datasets/99999/preview", headers=auth_headers)
        assert resp.status_code == 404
