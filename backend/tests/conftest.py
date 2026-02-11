"""
Shared test fixtures for MeowLLM backend tests.

Overrides the entire database engine to an in-memory SQLite DB.
Each test function gets a clean database.
"""

import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Set test env vars BEFORE importing the app
os.environ["MEOWLLM_JWT_SECRET"] = "test-secret-key-for-testing-only"

# Import the app and DB pieces
import app.database as db_module
from app.database import Base, get_db

# Import all models so Base.metadata knows about them
import app.models  # noqa: F401

# Create test engine ONCE (in-memory with StaticPool so all connections share it)
TEST_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)

# Monkey-patch the database module so everything uses our test engine
db_module.engine = TEST_ENGINE
db_module.SessionLocal = TestingSessionLocal

# NOW import the FastAPI app (which registers routes that may use SessionLocal directly)
from app.main import app as fastapi_app, limiter


@pytest.fixture(autouse=True)
def _setup_test_db():
    """Create all tables fresh for each test, then drop them after."""
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)


@pytest.fixture(autouse=True)
def _disable_rate_limit():
    """Disable rate limiting in tests."""
    limiter.enabled = False
    # Also disable the route-level limiter in auth
    from app.routes.auth import _limiter as auth_limiter
    auth_limiter.enabled = False
    yield
    limiter.enabled = True
    auth_limiter.enabled = True


@pytest.fixture(scope="function")
def db():
    """Get a test database session."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def client(db: Session):
    """FastAPI test client with the test database injected."""
    def override_get_db():
        try:
            yield db
        finally:
            pass

    fastapi_app.dependency_overrides[get_db] = override_get_db
    with TestClient(fastapi_app) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(client: TestClient) -> dict:
    """Register a test user and return auth headers."""
    resp = client.post("/auth/register", json={
        "email": "test@example.com",
        "password": "TestPass123",
        "display_name": "Test User",
    })
    assert resp.status_code == 200, f"Registration failed: {resp.text}"
    token = resp.json()["token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def project_id(client: TestClient, auth_headers: dict) -> int:
    """Create a test project and return its ID."""
    resp = client.post("/projects/", json={
        "name": "Test Project",
        "description": "A test project for unit testing",
    }, headers=auth_headers)
    assert resp.status_code == 200, f"Project creation failed: {resp.text}"
    return resp.json()["id"]
