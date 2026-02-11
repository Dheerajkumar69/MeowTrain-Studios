"""Tests for project CRUD endpoints."""


class TestCreateProject:
    def test_create_project(self, client, auth_headers):
        resp = client.post("/projects/", json={
            "name": "My ML Project",
            "description": "Testing project creation",
            "intended_use": "chatbot",
        }, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "My ML Project"
        assert data["description"] == "Testing project creation"
        assert data["intended_use"] == "chatbot"
        assert data["status"] == "created"

    def test_create_project_minimal(self, client, auth_headers):
        resp = client.post("/projects/", json={
            "name": "Minimal",
        }, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["description"] == ""

    def test_create_project_unauthenticated(self, client):
        resp = client.post("/projects/", json={"name": "Fail"})
        assert resp.status_code in (401, 422)


class TestListProjects:
    def test_list_projects(self, client, auth_headers):
        # Create two projects
        client.post("/projects/", json={"name": "P1"}, headers=auth_headers)
        client.post("/projects/", json={"name": "P2"}, headers=auth_headers)

        resp = client.get("/projects/", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_list_projects_isolated(self, client, auth_headers):
        # Create project as user 1
        client.post("/projects/", json={"name": "User1 Project"}, headers=auth_headers)

        # Guest should not see user1's projects
        guest_resp = client.post("/auth/guest")
        guest_token = guest_resp.json()["token"]
        guest_headers = {"Authorization": f"Bearer {guest_token}"}

        resp = client.get("/projects/", headers=guest_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 0


class TestGetProject:
    def test_get_project(self, client, auth_headers, project_id):
        resp = client.get(f"/projects/{project_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == project_id

    def test_get_nonexistent(self, client, auth_headers):
        resp = client.get("/projects/99999", headers=auth_headers)
        assert resp.status_code == 404


class TestUpdateProject:
    def test_update_project(self, client, auth_headers, project_id):
        resp = client.put(f"/projects/{project_id}", json={
            "name": "Updated Name",
            "description": "Updated desc",
        }, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated Name"


class TestDeleteProject:
    def test_delete_project(self, client, auth_headers, project_id):
        resp = client.delete(f"/projects/{project_id}", headers=auth_headers)
        assert resp.status_code == 200

        # Verify it's gone
        resp = client.get(f"/projects/{project_id}", headers=auth_headers)
        assert resp.status_code == 404
