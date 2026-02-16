"""Tests for dataset upload and management endpoints."""

import io
import json


class TestDatasetUpload:
    def test_upload_text_file(self, client, auth_headers, project_id):
        file_content = b"This is a test document for training."
        resp = client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
            headers=auth_headers,
        )
        # Accept 200 or 201
        assert resp.status_code in (200, 201), f"Upload failed: {resp.text}"
        data = resp.json()
        assert data["original_name"] == "test.txt"
        assert data["status"] in ("ready", "processing")

    def test_upload_json_file(self, client, auth_headers, project_id):
        data_content = json.dumps([
            {"instruction": "What is AI?", "output": "AI is..."},
            {"instruction": "Explain ML", "output": "ML is..."},
        ])
        resp = client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("train.json", io.BytesIO(data_content.encode()), "application/json")},
            headers=auth_headers,
        )
        assert resp.status_code in (200, 201)

    def test_upload_invalid_extension(self, client, auth_headers, project_id):
        resp = client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("malware.exe", io.BytesIO(b"bad"), "application/octet-stream")},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_upload_to_nonexistent_project(self, client, auth_headers):
        resp = client.post(
            "/api/projects/99999/datasets/upload",
            files={"file": ("test.txt", io.BytesIO(b"data"), "text/plain")},
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestListDatasets:
    def test_list_empty(self, client, auth_headers, project_id):
        resp = client.get(f"/api/projects/{project_id}/datasets/", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_list_after_upload(self, client, auth_headers, project_id):
        client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("test.txt", io.BytesIO(b"Some text data"), "text/plain")},
            headers=auth_headers,
        )
        resp = client.get(f"/api/projects/{project_id}/datasets/", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1


class TestDeleteDataset:
    def test_delete_dataset(self, client, auth_headers, project_id):
        # Upload first
        upload_resp = client.post(
            f"/api/projects/{project_id}/datasets/upload",
            files={"file": ("test.txt", io.BytesIO(b"Data"), "text/plain")},
            headers=auth_headers,
        )
        if upload_resp.status_code in (200, 201):
            dataset_id = upload_resp.json()["id"]
            resp = client.delete(f"/api/projects/{project_id}/datasets/{dataset_id}", headers=auth_headers)
            assert resp.status_code == 200
