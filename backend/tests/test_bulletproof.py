"""Tests for bulletproofing edge cases: validation, sanitization, and security."""

import io


class TestEmailValidation:
    """Verify Pydantic EmailStr validation rejects malformed emails."""

    def test_register_invalid_email_no_at(self, client):
        resp = client.post("/auth/register", json={
            "email": "notanemail",
            "password": "SecurePass1",
        })
        assert resp.status_code == 422  # Pydantic validation error

    def test_register_invalid_email_no_domain(self, client):
        resp = client.post("/auth/register", json={
            "email": "user@",
            "password": "SecurePass1",
        })
        assert resp.status_code == 422

    def test_register_valid_email(self, client):
        resp = client.post("/auth/register", json={
            "email": "valid@example.com",
            "password": "SecurePass1",
        })
        assert resp.status_code == 200


class TestDisplayNameCap:
    def test_display_name_too_long(self, client):
        resp = client.post("/auth/register", json={
            "email": "cap@test.com",
            "password": "SecurePass1",
            "display_name": "x" * 101,  # exceeds 100 char cap
        })
        assert resp.status_code == 422

    def test_display_name_at_limit(self, client):
        resp = client.post("/auth/register", json={
            "email": "cap@test.com",
            "password": "SecurePass1",
            "display_name": "x" * 100,  # exactly at cap
        })
        assert resp.status_code == 200


class TestProjectStatusValidation:
    def test_update_invalid_status(self, client, auth_headers, project_id):
        resp = client.put(f"/projects/{project_id}", json={
            "status": "hacked",  # not in valid statuses
        }, headers=auth_headers)
        assert resp.status_code == 422

    def test_update_valid_status(self, client, auth_headers, project_id):
        resp = client.put(f"/projects/{project_id}", json={
            "status": "trained",
        }, headers=auth_headers)
        assert resp.status_code == 200


class TestTrainingMethodValidation:
    def test_configure_invalid_method(self, client, auth_headers, project_id):
        # Upload a dataset first
        client.post(
            f"/projects/{project_id}/datasets/upload",
            files={"file": ("data.txt", io.BytesIO(b"Some training data content here"), "text/plain")},
            headers=auth_headers,
        )
        resp = client.post(f"/projects/{project_id}/train/configure", json={
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "method": "invalid_method",
        }, headers=auth_headers)
        assert resp.status_code == 422  # Pydantic rejects invalid Literal


class TestPromptMaxLength:
    def test_chat_prompt_too_long(self, client, auth_headers, project_id):
        resp = client.post(f"/projects/{project_id}/chat", json={
            "prompt": "x" * 50001,  # exceeds 50000 char cap
        }, headers=auth_headers)
        assert resp.status_code == 422


class TestEmptyFileUpload:
    def test_upload_empty_file(self, client, auth_headers, project_id):
        resp = client.post(
            f"/projects/{project_id}/datasets/upload",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()


class TestCascadeDelete:
    def test_delete_project_with_datasets(self, client, auth_headers, project_id):
        # Upload a dataset
        client.post(
            f"/projects/{project_id}/datasets/upload",
            files={"file": ("data.txt", io.BytesIO(b"Training data"), "text/plain")},
            headers=auth_headers,
        )
        # Verify dataset exists
        resp = client.get(f"/projects/{project_id}/datasets/", headers=auth_headers)
        assert len(resp.json()) > 0

        # Delete project (should cascade)
        resp = client.delete(f"/projects/{project_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["detail"] == "Project deleted"

        # Verify project is gone
        resp = client.get(f"/projects/{project_id}", headers=auth_headers)
        assert resp.status_code == 404
