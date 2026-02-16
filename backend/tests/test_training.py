"""Tests for training configuration and status endpoints."""

import io
import json


class TestTrainingConfigure:
    def _upload_dataset(self, client, auth_headers, project_id):
        """Helper: upload a dataset so training can be configured."""
        data = json.dumps([
            {"instruction": "What is AI?", "output": "AI is artificial intelligence."},
            {"instruction": "What is ML?", "output": "ML is machine learning."},
        ])
        resp = client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("train.json", io.BytesIO(data.encode()), "application/json")},
            headers=auth_headers,
        )
        assert resp.status_code in (200, 201)

    def test_configure_training(self, client, auth_headers, project_id):
        self._upload_dataset(client, auth_headers, project_id)
        resp = client.post(f"/api/projects/{project_id}/train/configure", json={
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "method": "lora",
            "epochs": 1,
            "batch_size": 2,
        }, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "configured"
        assert data["total_epochs"] == 1

    def test_configure_without_datasets(self, client, auth_headers, project_id):
        resp = client.post(f"/api/projects/{project_id}/train/configure", json={
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        }, headers=auth_headers)
        assert resp.status_code == 400
        assert "dataset" in resp.json()["detail"].lower()


class TestTrainingStatus:
    def test_status_no_runs(self, client, auth_headers, project_id):
        resp = client.get(f"/api/projects/{project_id}/train/status", headers=auth_headers)
        assert resp.status_code == 404

    def test_status_after_configure(self, client, auth_headers, project_id):
        # Upload data and configure
        data = json.dumps([{"instruction": "test", "output": "out"}])
        client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("train.json", io.BytesIO(data.encode()), "application/json")},
            headers=auth_headers,
        )
        client.post(f"/api/projects/{project_id}/train/configure", json={
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        }, headers=auth_headers)

        resp = client.get(f"/api/projects/{project_id}/train/status", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "configured"


class TestTrainingHistory:
    def test_empty_history(self, client, auth_headers, project_id):
        resp = client.get(f"/api/projects/{project_id}/train/history", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["runs"] == []
        assert data["total"] == 0


class TestTrainingControls:
    def test_start_without_config(self, client, auth_headers, project_id):
        resp = client.post(f"/api/projects/{project_id}/train/start", headers=auth_headers)
        assert resp.status_code == 400

    def test_pause_without_active(self, client, auth_headers, project_id):
        resp = client.post(f"/api/projects/{project_id}/train/pause", headers=auth_headers)
        assert resp.status_code == 409  # No active training to pause

    def test_stop_without_active(self, client, auth_headers, project_id):
        resp = client.post(f"/api/projects/{project_id}/train/stop", headers=auth_headers)
        assert resp.status_code == 409  # No active training to stop
