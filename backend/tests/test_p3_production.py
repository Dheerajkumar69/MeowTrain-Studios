"""
Tests for P3 — Production Hardening features.

Covers:
  - Health check with DB connectivity
  - Structured logging configuration
  - Admin panel (users, stats, cache)
  - Backup / restore
  - Model versioning / lineage
  - Email verification
  - OAuth endpoints (redirect behavior)
"""

import json
import io
import zipfile

import pytest
from fastapi.testclient import TestClient


# ── Helpers ──

def _register(client, email="p3test@example.com", password="TestPass123!"):
    resp = client.post("/api/auth/register", json={
        "email": email,
        "password": password,
        "display_name": "P3 Tester",
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def _login(client, email="p3test@example.com", password="TestPass123!"):
    resp = client.post("/api/auth/login", json={
        "email": email, "password": password,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def _auth_headers(client, email="p3test@example.com"):
    data = _register(client, email)
    return {"Authorization": f"Bearer {data['token']}"}


def _make_admin(db, email="p3test@example.com"):
    from app.models.user import User
    user = db.query(User).filter(User.email == email).first()
    if user:
        user.role = "admin"
        db.commit()


def _create_project(client, headers):
    resp = client.post("/api/projects/", json={
        "name": "Lineage Test Project",
        "description": "For testing lineage",
    }, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


# ===================================================================
# Health Check
# ===================================================================

class TestHealthCheck:
    def test_health_returns_db_status(self, client: TestClient):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert data["version"] == "0.4.0"
        assert data["db_connected"] is True
        assert data["db_latency_ms"] is not None
        assert isinstance(data["db_latency_ms"], (int, float))

    def test_health_db_latency_reasonable(self, client: TestClient):
        resp = client.get("/api/health")
        data = resp.json()
        # In-memory SQLite should respond in < 100ms
        assert data["db_latency_ms"] < 100


# ===================================================================
# Structured Logging
# ===================================================================

class TestLogging:
    def test_json_formatter(self):
        from app.logging_config import JSONFormatter
        import logging

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="hello %s", args=("world",), exc_info=None,
        )
        record.request_id = "abc123"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello world"
        assert parsed["request_id"] == "abc123"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed

    def test_json_formatter_with_exception(self):
        from app.logging_config import JSONFormatter
        import logging
        import sys

        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="test.py",
            lineno=1, msg="oops", args=(), exc_info=exc_info,
        )
        record.request_id = "-"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["exc_type"] == "ValueError"
        assert "test error" in parsed["exc_message"]

    def test_configure_logging_text(self):
        from app.logging_config import configure_logging
        configure_logging(level="DEBUG", log_format="text")
        import logging
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_configure_logging_json(self):
        from app.logging_config import configure_logging
        configure_logging(level="WARNING", log_format="json")
        import logging
        root = logging.getLogger()
        assert root.level == logging.WARNING


# ===================================================================
# Email Verification
# ===================================================================

class TestEmailVerification:
    def test_register_creates_unverified_user(self, client: TestClient):
        data = _register(client, "verify1@example.com")
        assert data["user"]["email_verified"] is False

    def test_verify_email_valid_token(self, client: TestClient, db):
        data = _register(client, "verify2@example.com")
        # Get the verification token from DB
        from app.models.user import User
        user = db.query(User).filter(User.email == "verify2@example.com").first()
        token = user.email_verification_token
        assert token is not None

        resp = client.post("/api/auth/verify-email", json={"token": token})
        assert resp.status_code == 200
        assert "verified" in resp.json()["detail"].lower()

        # Confirm in DB
        db.refresh(user)
        assert user.email_verified is True
        assert user.email_verification_token is None

    def test_verify_email_invalid_token(self, client: TestClient):
        resp = client.post("/api/auth/verify-email", json={"token": "bogus-token"})
        assert resp.status_code == 400

    def test_verify_email_already_verified(self, client: TestClient, db):
        data = _register(client, "verify3@example.com")
        from app.models.user import User
        user = db.query(User).filter(User.email == "verify3@example.com").first()
        token = user.email_verification_token

        # Verify once
        client.post("/api/auth/verify-email", json={"token": token})

        # Verify again — should say already verified
        resp = client.post("/api/auth/verify-email", json={"token": token})
        # Token was cleared, so it should 400
        assert resp.status_code == 400

    def test_resend_verification(self, client: TestClient, db):
        _register(client, "verify4@example.com")
        resp = client.post("/api/auth/resend-verification", json={"email": "verify4@example.com"})
        assert resp.status_code == 200

    def test_resend_verification_unknown_email(self, client: TestClient):
        resp = client.post("/api/auth/resend-verification", json={"email": "nobody@example.com"})
        # Should still return 200 to prevent enumeration
        assert resp.status_code == 200


# ===================================================================
# Admin Panel
# ===================================================================

class TestAdminPanel:
    def test_admin_stats_requires_admin(self, client: TestClient):
        headers = _auth_headers(client, "nonadmin@example.com")
        resp = client.get("/api/admin/stats", headers=headers)
        assert resp.status_code == 403

    def test_admin_stats(self, client: TestClient, db):
        headers = _auth_headers(client, "admin1@example.com")
        _make_admin(db, "admin1@example.com")
        resp = client.get("/api/admin/stats", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "users" in data
        assert "projects" in data
        assert "training" in data
        assert "disk" in data
        assert data["users"]["total"] >= 1

    def test_admin_list_users(self, client: TestClient, db):
        headers = _auth_headers(client, "admin2@example.com")
        _make_admin(db, "admin2@example.com")
        resp = client.get("/api/admin/users", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        assert len(data["items"]) >= 1

    def test_admin_list_users_search(self, client: TestClient, db):
        headers = _auth_headers(client, "admin3@example.com")
        _make_admin(db, "admin3@example.com")
        # Create another user
        _register(client, "findme@example.com")
        resp = client.get("/api/admin/users", headers=headers, params={"search": "findme"})
        assert resp.status_code == 200
        data = resp.json()
        assert any("findme" in u["email"] for u in data["items"])

    def test_admin_update_user_role(self, client: TestClient, db):
        headers = _auth_headers(client, "admin4@example.com")
        _make_admin(db, "admin4@example.com")
        target_data = _register(client, "target@example.com")
        target_id = target_data["user"]["id"]

        resp = client.patch(f"/api/admin/users/{target_id}", headers=headers, params={"role": "admin"})
        assert resp.status_code == 200

    def test_admin_cannot_change_own_role(self, client: TestClient, db):
        data = _register(client, "admin5@example.com")
        _make_admin(db, "admin5@example.com")
        headers = {"Authorization": f"Bearer {data['token']}"}
        own_id = data["user"]["id"]

        resp = client.patch(f"/api/admin/users/{own_id}", headers=headers, params={"role": "member"})
        assert resp.status_code == 400

    def test_admin_delete_user(self, client: TestClient, db):
        headers = _auth_headers(client, "admin6@example.com")
        _make_admin(db, "admin6@example.com")
        target_data = _register(client, "delete_target@example.com")
        target_id = target_data["user"]["id"]

        resp = client.delete(f"/api/admin/users/{target_id}", headers=headers)
        assert resp.status_code == 200

    def test_admin_cannot_delete_self(self, client: TestClient, db):
        data = _register(client, "admin7@example.com")
        _make_admin(db, "admin7@example.com")
        headers = {"Authorization": f"Bearer {data['token']}"}
        own_id = data["user"]["id"]

        resp = client.delete(f"/api/admin/users/{own_id}", headers=headers)
        assert resp.status_code == 400

    def test_admin_cache_list(self, client: TestClient, db):
        headers = _auth_headers(client, "admin8@example.com")
        _make_admin(db, "admin8@example.com")
        resp = client.get("/api/admin/cache", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert "total_size_gb" in data

    def test_admin_cache_evict_not_found(self, client: TestClient, db):
        headers = _auth_headers(client, "admin9@example.com")
        _make_admin(db, "admin9@example.com")
        resp = client.delete("/api/admin/cache/nonexistent-model", headers=headers)
        assert resp.status_code == 404

    def test_non_admin_cannot_access_admin_routes(self, client: TestClient):
        headers = _auth_headers(client, "regular@example.com")
        assert client.get("/api/admin/users", headers=headers).status_code == 403
        assert client.get("/api/admin/stats", headers=headers).status_code == 403
        assert client.get("/api/admin/cache", headers=headers).status_code == 403


# ===================================================================
# Backup / Restore
# ===================================================================

class TestBackupRestore:
    def test_backup_project(self, client: TestClient):
        headers = _auth_headers(client, "backup1@example.com")
        project_id = _create_project(client, headers)

        resp = client.get(f"/api/projects/{project_id}/backup", headers=headers)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

        # Verify it's valid zip
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        assert "metadata.json" in zf.namelist()
        metadata = json.loads(zf.read("metadata.json"))
        assert metadata["project"]["name"] == "Lineage Test Project"
        assert metadata["version"] == "1.0"

    def test_backup_not_found(self, client: TestClient):
        headers = _auth_headers(client, "backup2@example.com")
        resp = client.get("/api/projects/99999/backup", headers=headers)
        assert resp.status_code == 404

    def test_import_project(self, client: TestClient):
        headers = _auth_headers(client, "import1@example.com")

        # First create and backup a project
        project_id = _create_project(client, headers)
        backup_resp = client.get(f"/api/projects/{project_id}/backup", headers=headers)
        assert backup_resp.status_code == 200

        # Now import it
        import_resp = client.post(
            "/api/projects/import",
            files={"file": ("backup.zip", backup_resp.content, "application/zip")},
            headers=headers,
        )
        assert import_resp.status_code == 200
        data = import_resp.json()
        assert data["project_id"] != project_id  # New project created
        assert "imported" in data["detail"].lower()

    def test_import_invalid_file(self, client: TestClient):
        headers = _auth_headers(client, "import2@example.com")
        resp = client.post(
            "/api/projects/import",
            files={"file": ("bad.zip", b"not a zip", "application/zip")},
            headers=headers,
        )
        assert resp.status_code == 400

    def test_import_not_zip_extension(self, client: TestClient):
        headers = _auth_headers(client, "import3@example.com")
        resp = client.post(
            "/api/projects/import",
            files={"file": ("bad.txt", b"hello", "text/plain")},
            headers=headers,
        )
        assert resp.status_code == 400


# ===================================================================
# Model Versioning / Lineage
# ===================================================================

class TestLineage:
    def test_lineage_empty_project(self, client: TestClient):
        headers = _auth_headers(client, "lineage1@example.com")
        project_id = _create_project(client, headers)

        resp = client.get(f"/api/projects/{project_id}/lineage", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_id"] == project_id
        assert data["total_runs"] == 0
        assert data["lineage"] == []

    def test_lineage_not_found(self, client: TestClient):
        headers = _auth_headers(client, "lineage2@example.com")
        resp = client.get("/api/projects/99999/lineage", headers=headers)
        assert resp.status_code == 404

    def test_run_lineage_not_found(self, client: TestClient):
        headers = _auth_headers(client, "lineage3@example.com")
        project_id = _create_project(client, headers)

        resp = client.get(f"/api/projects/{project_id}/lineage/runs/99999", headers=headers)
        assert resp.status_code == 404


# ===================================================================
# OAuth (redirect behavior — no actual OAuth server)
# ===================================================================

class TestOAuth:
    def test_google_redirect_not_configured(self, client: TestClient):
        """Google OAuth should return 501 when not configured."""
        resp = client.get("/api/auth/oauth/google", follow_redirects=False)
        # If OAuth is not configured, it should 501
        # If it IS configured (unlikely in test), it redirects
        assert resp.status_code in (302, 307, 501)

    def test_github_redirect_not_configured(self, client: TestClient):
        """GitHub OAuth should return 501 when not configured."""
        resp = client.get("/api/auth/oauth/github", follow_redirects=False)
        assert resp.status_code in (302, 307, 501)

    def test_google_callback_not_configured(self, client: TestClient):
        resp = client.get("/api/auth/oauth/google/callback", params={"code": "fake"})
        assert resp.status_code in (400, 501)

    def test_github_callback_not_configured(self, client: TestClient):
        resp = client.get("/api/auth/oauth/github/callback", params={"code": "fake"})
        assert resp.status_code in (400, 501)
