import os
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from config import config
from document_processor import DocumentProcessor
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from session_manager import SessionManager
from vector_store import VectorStore


@pytest.fixture
def sample_course_data():
    """Sample course data for testing"""
    return {
        "title": "Test Course",
        "course_link": "https://example.com/course",
        "instructor": "Test Instructor",
        "lessons": [
            {
                "lesson_number": 1,
                "title": "Introduction",
                "lesson_link": "https://example.com/lesson1",
                "content": "This is the introduction to the test course. It covers basic concepts and prerequisites.",
            },
            {
                "lesson_number": 2,
                "title": "Advanced Topics",
                "lesson_link": "https://example.com/lesson2",
                "content": "This lesson covers advanced topics and complex concepts for deeper understanding.",
            },
        ],
    }


@pytest.fixture
def sample_course_document():
    """Sample course document text for testing"""
    return """Course Title: Test Course
Course Link: https://example.com/course
Course Instructor: Test Instructor

Lesson 1: Introduction
Lesson Link: https://example.com/lesson1
This is the introduction to the test course. It covers basic concepts and prerequisites. Students will learn fundamental principles and prepare for advanced topics.

Lesson 2: Advanced Topics
Lesson Link: https://example.com/lesson2
This lesson covers advanced topics and complex concepts for deeper understanding. Building on the foundation from lesson 1, students explore sophisticated techniques."""


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    # Use in-memory ChromaDB for testing
    import shutil
    import tempfile

    temp_dir = tempfile.mkdtemp()
    store = VectorStore(temp_dir, "all-MiniLM-L6-v2", max_results=5)

    yield store

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_document_processor():
    """Create a document processor for testing"""
    return DocumentProcessor(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def mock_ai_generator():
    """Create a mock AI generator for testing"""
    # Use a dummy API key for testing
    return AIGenerator("test_api_key", "glm-4.5")


@pytest.fixture
def mock_session_manager():
    """Create a session manager for testing"""
    return SessionManager(max_history=2)


@pytest.fixture
def mock_tool_manager(mock_vector_store):
    """Create a tool manager with search tools for testing"""
    tool_manager = ToolManager()
    search_tool = CourseSearchTool(mock_vector_store)
    outline_tool = CourseOutlineTool(mock_vector_store)
    tool_manager.register_tool(search_tool)
    tool_manager.register_tool(outline_tool)
    return tool_manager


@pytest.fixture
def mock_rag_system(
    mock_vector_store,
    mock_document_processor,
    mock_ai_generator,
    mock_session_manager,
    mock_tool_manager,
):
    """Create a complete RAG system for testing"""
    from dataclasses import dataclass

    from config import Config

    # Create a test config
    @dataclass
    class TestConfig:
        ZHIPUAI_API_KEY: str = "test_key"
        ZHIPUAI_MODEL: str = "glm-4.5"
        EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
        CHUNK_SIZE: int = 100
        CHUNK_OVERLAP: int = 20
        MAX_RESULTS: int = 5
        MAX_HISTORY: int = 2
        CHROMA_PATH: str = "./test_chroma_db"

    test_config = TestConfig()

    # Create RAG system
    rag_system = RAGSystem(test_config)

    # Replace components with mocks
    rag_system.vector_store = mock_vector_store
    rag_system.document_processor = mock_document_processor
    rag_system.ai_generator = mock_ai_generator
    rag_system.session_manager = mock_session_manager
    rag_system.tool_manager = mock_tool_manager

    return rag_system


@pytest.fixture
def populated_vector_store(mock_vector_store, sample_course_data):
    """Vector store populated with sample course data"""
    # Create course object
    course = Course(
        title=sample_course_data["title"],
        course_link=sample_course_data["course_link"],
        instructor=sample_course_data["instructor"],
    )

    # Add lessons to course
    for lesson_data in sample_course_data["lessons"]:
        lesson = Lesson(
            lesson_number=lesson_data["lesson_number"],
            title=lesson_data["title"],
            lesson_link=lesson_data["lesson_link"],
        )
        course.lessons.append(lesson)

    # Create course chunks
    chunks = []
    for i, lesson_data in enumerate(sample_course_data["lessons"]):
        chunk = CourseChunk(
            content=f"Course {sample_course_data['title']} Lesson {lesson_data['lesson_number']} content: {lesson_data['content']}",
            course_title=sample_course_data["title"],
            lesson_number=lesson_data["lesson_number"],
            chunk_index=i,
        )
        chunks.append(chunk)

    # Add to vector store
    mock_vector_store.add_course_metadata(course)
    mock_vector_store.add_course_content(chunks)
return mock_vector_store

@pytest.fixture
def test_app():
    """Create a test FastAPI app without static files"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create test app without static files
    app = FastAPI(title="Test Course Materials RAG System", root_path="")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models for request/response
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    class ClearSessionRequest(BaseModel):
        session_id: str

    class ClearSessionResponse(BaseModel):
        success: bool
        message: str
    
    # Mock RAG system
    mock_rag = MagicMock()
    mock_rag.query.return_value = ("Test answer", ["source1", "source2"])
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Test Course"]
    }
    mock_rag.session_manager = MagicMock()
    mock_rag.session_manager.create_session.return_value = "test_session_id"
    mock_rag.session_manager.clear_session.return_value = None
    
    # Store mock in app state
    app.state.rag_system = mock_rag
    
    # API endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = app.state.rag_system.session_manager.create_session()
            
            answer, sources = app.state.rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/clear-session", response_model=ClearSessionResponse)
    async def clear_session(request: ClearSessionRequest):
        try:
            app.state.rag_system.session_manager.clear_session(request.session_id)
            
            return ClearSessionResponse(
                success=True,
                message=f"Session {request.session_id} cleared successfully"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = app.state.rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

@pytest.fixture
def client(test_app):
    """Create test client for the app"""
    return TestClient(test_app)

@pytest.fixture
def sample_query_request():
    """Sample query request for testing"""
    return {"query": "What is machine learning?"}

@pytest.fixture
def sample_clear_session_request():
    """Sample clear session request for testing"""
    return {"session_id": "test_session_id"}
