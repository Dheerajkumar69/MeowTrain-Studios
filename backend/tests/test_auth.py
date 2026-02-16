"""Tests for authentication endpoints."""


class TestRegister:
    def test_register_success(self, client):
        resp = client.post("/api/auth/register", json={
            "email": "user@test.com",
            "password": "SecurePass1",
            "display_name": "New User",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["user"]["email"] == "user@test.com"
        assert data["user"]["display_name"] == "New User"
        assert data["user"]["is_guest"] is False

    def test_register_duplicate_email(self, client):
        client.post("/api/auth/register", json={
            "email": "dup@test.com",
            "password": "SecurePass1",
        })
        resp = client.post("/api/auth/register", json={
            "email": "dup@test.com",
            "password": "SecurePass1",
        })
        assert resp.status_code == 400
        assert "already registered" in resp.json()["detail"].lower()

    def test_register_weak_password_too_short(self, client):
        # Password is 7 chars, Pydantic min_length=8 catches it → 422
        resp = client.post("/api/auth/register", json={
            "email": "weak@test.com",
            "password": "Short1x",  # 7 chars, fails Pydantic min_length=8
        })
        assert resp.status_code == 422

    def test_register_weak_password_no_digit(self, client):
        resp = client.post("/api/auth/register", json={
            "email": "weak@test.com",
            "password": "NoDigitsHereX",  # Long enough but no digit
        })
        assert resp.status_code == 400
        assert "digit" in resp.json()["detail"]

    def test_register_weak_password_no_letter(self, client):
        resp = client.post("/api/auth/register", json={
            "email": "weak@test.com",
            "password": "1234567890",  # Long enough but no letter
        })
        assert resp.status_code == 400
        assert "letter" in resp.json()["detail"]


class TestLogin:
    def test_login_success(self, client):
        # Register first
        client.post("/api/auth/register", json={
            "email": "login@test.com",
            "password": "LoginPass1",
        })
        resp = client.post("/api/auth/login", json={
            "email": "login@test.com",
            "password": "LoginPass1",
        })
        assert resp.status_code == 200
        assert "token" in resp.json()

    def test_login_wrong_password(self, client):
        client.post("/api/auth/register", json={
            "email": "login@test.com",
            "password": "CorrectPass1",
        })
        resp = client.post("/api/auth/login", json={
            "email": "login@test.com",
            "password": "WrongPass1",
        })
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, client):
        resp = client.post("/api/auth/login", json={
            "email": "nobody@test.com",
            "password": "Whatever1",
        })
        assert resp.status_code == 401


class TestGuestLogin:
    def test_guest_login(self, client):
        resp = client.post("/api/auth/guest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"]["is_guest"] is True
        assert "token" in data


class TestMe:
    def test_me_authenticated(self, client, auth_headers):
        resp = client.get("/api/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["email"] == "test@example.com"

    def test_me_no_token(self, client):
        resp = client.get("/api/auth/me")
        # Should fail without auth (401, 422, or 500 depending on error handling)
        assert resp.status_code != 200
