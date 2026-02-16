"""Tests for model registry endpoints."""


class TestListModels:
    def test_list_models(self, client):
        resp = client.get("/api/models/")
        assert resp.status_code == 200
        models = resp.json()
        assert len(models) > 0
        # Check model structure
        first = models[0]
        assert "model_id" in first
        assert "name" in first
        assert "size_gb" in first
        assert "is_cached" in first
        assert "compatibility" in first

    def test_model_has_required_fields(self, client):
        resp = client.get("/api/models/")
        for model in resp.json():
            assert isinstance(model["model_id"], str)
            assert isinstance(model["size_gb"], (int, float))
            assert model["compatibility"] in ("compatible", "may_be_slow", "too_large")

    def test_list_models_returns_catalog(self, client):
        """Catalog should include TinyLlama at minimum."""
        models = client.get("/api/models/").json()
        model_ids = [m["model_id"] for m in models]
        assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0" in model_ids

    def test_model_has_numeric_specs(self, client):
        """Each model should have numeric hardware spec fields."""
        models = client.get("/api/models/").json()
        for m in models:
            assert isinstance(m["ram_required_gb"], (int, float))
            assert isinstance(m["vram_required_gb"], (int, float))
            assert isinstance(m["estimated_train_minutes"], (int, float))


class TestModelStatus:
    def test_status_known_model(self, client, auth_headers):
        resp = client.get("/api/models/TinyLlama%2FTinyLlama-1.1B-Chat-v1.0/status", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert "is_cached" in data

    def test_status_unknown_model(self, client, auth_headers):
        resp = client.get("/api/models/nonexistent%2Fmodel/status", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "nonexistent/model"
        assert data["is_cached"] is False
        assert data["size_gb"] is None  # not in catalog

    def test_status_unauthenticated(self, client):
        """Model status requires auth."""
        resp = client.get("/api/models/TinyLlama%2FTinyLlama-1.1B-Chat-v1.0/status")
        assert resp.status_code == 401


class TestModelDownload:
    def test_download_returns_status(self, client, auth_headers):
        """Test that download endpoint returns proper status (not actually downloading)."""
        resp = client.post("/api/models/TinyLlama%2FTinyLlama-1.1B-Chat-v1.0/download", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("downloading", "cached")

    def test_download_unknown_model(self, client, auth_headers):
        """Unknown models are accepted (custom HF models) — download starts or fails gracefully."""
        resp = client.post("/api/models/nonexistent%2Fmodel/download", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("downloading", "cached")

    def test_download_unauthenticated(self, client):
        """Download requires auth."""
        resp = client.post("/api/models/TinyLlama%2FTinyLlama-1.1B-Chat-v1.0/download")
        assert resp.status_code == 401

    def test_download_invalid_model_id(self, client, auth_headers):
        """Invalid model ID format should be rejected."""
        resp = client.post("/api/models/invalid-no-slash/download", headers=auth_headers)
        assert resp.status_code == 400

    def test_download_progress_not_started(self, client, auth_headers):
        """Progress for a model that hasn't been downloaded yet."""
        resp = client.get("/api/models/some-org%2Fsome-model/download/progress", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("not_started", "cached")

    def test_cancel_download_no_active(self, client, auth_headers):
        """Cancel should fail if no active download."""
        resp = client.delete("/api/models/some-org%2Fsome-model/download", headers=auth_headers)
        assert resp.status_code == 400
