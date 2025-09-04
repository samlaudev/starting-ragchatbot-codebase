import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


class TestAPIEndpoints:
    """Test suite for FastAPI endpoints"""

    def test_query_endpoint_success(self, client, sample_query_request):
        """Test successful query endpoint"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "Test answer"
        assert data["sources"] == ["source1", "source2"]
        assert data["session_id"] == "test_session_id"

    def test_query_endpoint_with_session_id(self, client):
        """Test query endpoint with existing session ID"""
        request_data = {"query": "What is AI?", "session_id": "existing_session"}
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing_session"

    def test_query_endpoint_empty_query(self, client):
        """Test query endpoint with empty query"""
        request_data = {"query": ""}
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_query_endpoint_exception_handling(self, client):
        """Test query endpoint exception handling"""
        with patch.object(client.app.state.rag_system, 'query', side_effect=Exception("Test error")):
            request_data = {"query": "test query"}
            response = client.post("/api/query", json=request_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert data["detail"] == "Test error"

    def test_clear_session_endpoint_success(self, client, sample_clear_session_request):
        """Test successful clear session endpoint"""
        response = client.post("/api/clear-session", json=sample_clear_session_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Session test_session_id cleared successfully"

    def test_clear_session_endpoint_exception_handling(self, client):
        """Test clear session endpoint exception handling"""
        with patch.object(client.app.state.rag_system.session_manager, 'clear_session', side_effect=Exception("Clear error")):
            request_data = {"session_id": "test_session"}
            response = client.post("/api/clear-session", json=request_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert data["detail"] == "Clear error"

    def test_get_course_stats_endpoint_success(self, client):
        """Test successful get course stats endpoint"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 1
        assert data["course_titles"] == ["Test Course"]

    def test_get_course_stats_endpoint_exception_handling(self, client):
        """Test get course stats endpoint exception handling"""
        with patch.object(client.app.state.rag_system, 'get_course_analytics', side_effect=Exception("Stats error")):
            response = client.get("/api/courses")
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert data["detail"] == "Stats error"

    def test_root_endpoint_redirect(self, client):
        """Test root endpoint redirects appropriately"""
        response = client.get("/")
        assert response.status_code == 404

    def test_invalid_endpoint(self, client):
        """Test invalid endpoint returns 404"""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404

    def test_query_endpoint_invalid_method(self, client):
        """Test query endpoint with invalid HTTP method"""
        response = client.get("/api/query")
        assert response.status_code == 405

    def test_clear_session_endpoint_invalid_method(self, client):
        """Test clear session endpoint with invalid HTTP method"""
        response = client.get("/api/clear-session")
        assert response.status_code == 405

    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/api/query")
        assert response.status_code == 405
        # CORS headers may not be present on OPTIONS requests in test client

    def test_query_endpoint_request_validation(self, client):
        """Test query endpoint request validation"""
        # Missing required query field
        response = client.post("/api/query", json={})
        assert response.status_code == 422

    def test_clear_session_endpoint_request_validation(self, client):
        """Test clear session endpoint request validation"""
        # Missing required session_id field
        response = client.post("/api/clear-session", json={})
        assert response.status_code == 422

    @pytest.mark.integration
    def test_full_query_flow(self, client):
        """Test complete flow from query to session management"""
        # Initial query
        query_data = {"query": "What is machine learning?"}
        response = client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        session_id = data["session_id"]
        
        # Clear the session
        clear_data = {"session_id": session_id}
        response = client.post("/api/clear-session", json=clear_data)
        
        assert response.status_code == 200
        assert response.json()["success"] is True

    @pytest.mark.api
    def test_api_response_structure(self, client):
        """Test API response structures match expected format"""
        # Test query response
        query_data = {"query": "test query"}
        response = client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Test courses response
        response = client.get("/api/courses")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Test clear session response
        clear_data = {"session_id": "test"}
        response = client.post("/api/clear-session", json=clear_data)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["success"], bool)
        assert isinstance(data["message"], str)