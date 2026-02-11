"""Tests for model registry endpoints."""


class TestListModels:
    def test_list_models(self, client):
        resp = client.get("/models/")
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
        resp = client.get("/models/")
        for model in resp.json():
            assert isinstance(model["model_id"], str)
            assert isinstance(model["size_gb"], (int, float))
            assert model["compatibility"] in ("compatible", "may_be_slow", "too_large")


class TestModelStatus:
    def test_status_known_model(self, client):
        resp = client.get("/models/TinyLlama%2FTinyLlama-1.1B-Chat-v1.0/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert "is_cached" in data

    def test_status_unknown_model(self, client):
        resp = client.get("/models/nonexistent%2Fmodel/status")
        assert resp.status_code == 404


class TestModelDownload:
    def test_download_returns_status(self, client):
        """Test that download endpoint returns proper status (not actually downloading)."""
        resp = client.post("/models/TinyLlama%2FTinyLlama-1.1B-Chat-v1.0/download")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("downloading", "cached")

    def test_download_unknown_model(self, client):
        resp = client.post("/models/nonexistent%2Fmodel/download")
        assert resp.status_code == 404
