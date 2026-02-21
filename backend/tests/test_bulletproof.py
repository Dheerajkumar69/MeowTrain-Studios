"""Tests for bulletproofing edge cases: validation, sanitization, and security."""

import io


class TestEmailValidation:
    """Verify Pydantic EmailStr validation rejects malformed emails."""

    def test_register_invalid_email_no_at(self, client):
        resp = client.post("/api/auth/register", json={
            "email": "notanemail",
            "password": "SecurePass1",
        })
        assert resp.status_code == 422  # Pydantic validation error

    def test_register_invalid_email_no_domain(self, client):
        resp = client.post("/api/auth/register", json={
            "email": "user@",
            "password": "SecurePass1",
        })
        assert resp.status_code == 422

    def test_register_valid_email(self, client):
        resp = client.post("/api/auth/register", json={
            "email": "valid@example.com",
            "password": "SecurePass1",
        })
        assert resp.status_code == 200


class TestDisplayNameCap:
    def test_display_name_too_long(self, client):
        resp = client.post("/api/auth/register", json={
            "email": "cap@test.com",
            "password": "SecurePass1",
            "display_name": "x" * 101,  # exceeds 100 char cap
        })
        assert resp.status_code == 422

    def test_display_name_at_limit(self, client):
        resp = client.post("/api/auth/register", json={
            "email": "cap@test.com",
            "password": "SecurePass1",
            "display_name": "x" * 100,  # exactly at cap
        })
        assert resp.status_code == 200


class TestProjectStatusValidation:
    def test_update_status_field_ignored(self, client, auth_headers, project_id):
        """Status field is intentionally excluded from ProjectUpdate schema.
        Sending it should be silently ignored (Pydantic drops unknown fields),
        and the project status should remain 'created'."""
        resp = client.put(f"/api/projects/{project_id}", json={
            "status": "trained",  # this field is not in the schema — ignored
        }, headers=auth_headers)
        assert resp.status_code == 200
        # Verify status was NOT changed
        assert resp.json()["status"] == "created"

    def test_update_with_valid_fields(self, client, auth_headers, project_id):
        resp = client.put(f"/api/projects/{project_id}", json={
            "name": "Updated Name",
            "description": "Updated description",
        }, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated Name"


class TestTrainingMethodValidation:
    def test_configure_invalid_method(self, client, auth_headers, project_id):
        # Upload a dataset first
        client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("data.txt", io.BytesIO(b"Some training data content here"), "text/plain")},
            headers=auth_headers,
        )
        resp = client.post(f"/api/projects/{project_id}/train/configure", json={
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "method": "invalid_method",
        }, headers=auth_headers)
        assert resp.status_code == 422  # Pydantic rejects invalid Literal


class TestPromptMaxLength:
    def test_chat_prompt_too_long(self, client, auth_headers, project_id):
        resp = client.post(f"/api/projects/{project_id}/chat", json={
            "prompt": "x" * 50001,  # exceeds 50000 char cap
        }, headers=auth_headers)
        assert resp.status_code == 422


class TestEmptyFileUpload:
    def test_upload_empty_file(self, client, auth_headers, project_id):
        resp = client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()


class TestCascadeDelete:
    def test_delete_project_with_datasets(self, client, auth_headers, project_id):
        # Upload a dataset
        client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("data.txt", io.BytesIO(b"Training data"), "text/plain")},
            headers=auth_headers,
        )
        # Verify dataset exists
        resp = client.get(f"/api/projects/{project_id}/datasets/", headers=auth_headers)
        assert resp.json()["total"] > 0

        # Delete project (should cascade)
        resp = client.delete(f"/api/projects/{project_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["detail"] == "Project deleted"

        # Verify project is gone
        resp = client.get(f"/api/projects/{project_id}", headers=auth_headers)
        assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────
# Security-focused tests (Phase 2 bulletproofing)
# ──────────────────────────────────────────────────────────────

import json
import zipfile


class TestSQLLikeInjection:
    """Ensure LIKE wildcards in search params are escaped, not interpreted."""

    def test_search_with_percent_returns_no_false_matches(self, client, auth_headers):
        """A search for '%' should not match all projects."""
        # Create a project with a normal name
        client.post("/api/projects/", json={"name": "MyProject"}, headers=auth_headers)

        # Search with raw '%' — should be escaped so it only matches literal '%'
        resp = client.get("/api/projects/?search=%25", headers=auth_headers)  # %25 = URL-encoded '%'
        assert resp.status_code == 200
        data = resp.json()
        # The search term is literal "%" — should NOT match "MyProject"
        for item in data.get("items", []):
            assert "%" in item["name"], f"False match: {item['name']}"

    def test_search_with_underscore_returns_no_false_matches(self, client, auth_headers):
        """A search for '_' should not match single-character wildcards."""
        client.post("/api/projects/", json={"name": "ABC"}, headers=auth_headers)

        resp = client.get("/api/projects/?search=_BC", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        # "_BC" should NOT match "ABC" because '_' is escaped
        for item in data.get("items", []):
            assert "_BC" in item["name"] or "_bc" in item["name"].lower(), f"False match: {item['name']}"

    def test_search_normal_term_still_works(self, client, auth_headers):
        """Normal search (no LIKE special chars) should still work."""
        client.post("/api/projects/", json={"name": "UniqueSearchTerm"}, headers=auth_headers)

        resp = client.get("/api/projects/?search=UniqueSearch", headers=auth_headers)
        assert resp.status_code == 200
        assert any("UniqueSearchTerm" in item["name"] for item in resp.json().get("items", []))


class TestAccountDeletionSecurity:
    """Account deletion must require password confirmation for registered users."""

    def test_delete_requires_password_for_registered_user(self, client, auth_headers):
        """DELETE /api/auth/account without password should be rejected for registered users."""
        resp = client.delete("/api/auth/account", headers=auth_headers)
        assert resp.status_code == 400
        assert "password" in resp.json()["detail"].lower()

    def test_delete_guest_no_password_needed(self, client):
        """Guest users should be able to delete without password."""
        guest_resp = client.post("/api/auth/guest")
        assert guest_resp.status_code == 200
        token = guest_resp.json()["token"]

        resp = client.delete("/api/auth/account", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert "deleted" in resp.json()["detail"].lower()


class TestBackupImportValidation:
    """Validate that backup import rejects malformed archives."""

    def _make_zip(self, metadata_dict):
        """Helper: create an in-memory zip with a metadata.json."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("metadata.json", json.dumps(metadata_dict))
        buf.seek(0)
        return buf

    def test_import_missing_metadata(self, client, auth_headers):
        """Zip without metadata.json should fail."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("dummy.txt", "hello")
        buf.seek(0)

        resp = client.post(
            "/api/projects/import",
            files={"file": ("backup.zip", buf, "application/zip")},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "metadata" in resp.json()["detail"].lower()

    def test_import_missing_project_name(self, client, auth_headers):
        """Metadata without project name should fail."""
        buf = self._make_zip({"project": {}})
        resp = client.post(
            "/api/projects/import",
            files={"file": ("backup.zip", buf, "application/zip")},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "name" in resp.json()["detail"].lower()

    def test_import_invalid_datasets_format(self, client, auth_headers):
        """Datasets field as a string instead of list should fail."""
        buf = self._make_zip({"project": {"name": "Test"}, "datasets": "not a list"})
        resp = client.post(
            "/api/projects/import",
            files={"file": ("backup.zip", buf, "application/zip")},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "datasets" in resp.json()["detail"].lower()

    def test_import_valid_backup_succeeds(self, client, auth_headers):
        """A properly formatted backup should import successfully."""
        buf = self._make_zip({
            "project": {"name": "ValidProject", "description": "A test"},
            "datasets": [],
            "prompt_templates": [],
        })
        resp = client.post(
            "/api/projects/import",
            files={"file": ("backup.zip", buf, "application/zip")},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["project_id"] > 0

    def test_import_non_zip_rejected(self, client, auth_headers):
        """Uploading a non-zip file should fail."""
        resp = client.post(
            "/api/projects/import",
            files={"file": ("backup.txt", io.BytesIO(b"not a zip"), "text/plain")},
            headers=auth_headers,
        )
        assert resp.status_code == 400


class TestPaginationClamping:
    """Verify that pagination parameters are properly clamped."""

    def test_per_page_clamped_high(self, client, auth_headers):
        """per_page=9999 should be clamped to max (100 for projects, 200 for admin)."""
        resp = client.get("/api/projects/?per_page=9999", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["per_page"] <= 100

    def test_page_clamped_low(self, client, auth_headers):
        """page=0 or page=-1 should be clamped to 1."""
        resp = client.get("/api/projects/?page=0", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["page"] >= 1


class TestTrainingHyperparamValidation:
    """Cross-field validation of training hyperparameters."""

    def test_lora_alpha_less_than_rank_rejected(self, client, auth_headers, project_id):
        """lora_alpha < lora_rank should be rejected for LoRA/QLoRA methods."""
        # Upload a dataset first
        client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("data.txt", io.BytesIO(b"Some training data content"), "text/plain")},
            headers=auth_headers,
        )

        resp = client.post(f"/api/projects/{project_id}/train/configure", json={
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "method": "lora",
            "lora_rank": 64,
            "lora_alpha": 16,  # alpha < rank — should fail
        }, headers=auth_headers)
        assert resp.status_code == 422

    def test_valid_lora_config_accepted(self, client, auth_headers, project_id):
        """Valid LoRA config (alpha >= rank) should be accepted."""
        client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("data.txt", io.BytesIO(b"Some training data content"), "text/plain")},
            headers=auth_headers,
        )

        resp = client.post(f"/api/projects/{project_id}/train/configure", json={
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "method": "lora",
            "lora_rank": 16,
            "lora_alpha": 32,  # alpha >= rank — good
        }, headers=auth_headers)
        assert resp.status_code == 200

