import pytest
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from rag_system import RAGSystem
from vector_store import VectorStore
from document_processor import DocumentProcessor
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
from models import Course, Lesson, CourseChunk
from session_manager import SessionManager

@pytest.fixture
def sample_course_data():
    """Sample course data for testing"""
    return {
        'title': 'Test Course',
        'course_link': 'https://example.com/course',
        'instructor': 'Test Instructor',
        'lessons': [
            {
                'lesson_number': 1,
                'title': 'Introduction',
                'lesson_link': 'https://example.com/lesson1',
                'content': 'This is the introduction to the test course. It covers basic concepts and prerequisites.'
            },
            {
                'lesson_number': 2,
                'title': 'Advanced Topics',
                'lesson_link': 'https://example.com/lesson2',
                'content': 'This lesson covers advanced topics and complex concepts for deeper understanding.'
            }
        ]
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
    import tempfile
    import shutil
    
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
def mock_rag_system(mock_vector_store, mock_document_processor, mock_ai_generator, mock_session_manager, mock_tool_manager):
    """Create a complete RAG system for testing"""
    from config import Config
    from dataclasses import dataclass
    
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
        title=sample_course_data['title'],
        course_link=sample_course_data['course_link'],
        instructor=sample_course_data['instructor']
    )
    
    # Add lessons to course
    for lesson_data in sample_course_data['lessons']:
        lesson = Lesson(
            lesson_number=lesson_data['lesson_number'],
            title=lesson_data['title'],
            lesson_link=lesson_data['lesson_link']
        )
        course.lessons.append(lesson)
    
    # Create course chunks
    chunks = []
    for i, lesson_data in enumerate(sample_course_data['lessons']):
        chunk = CourseChunk(
            content=f"Course {sample_course_data['title']} Lesson {lesson_data['lesson_number']} content: {lesson_data['content']}",
            course_title=sample_course_data['title'],
            lesson_number=lesson_data['lesson_number'],
            chunk_index=i
        )
        chunks.append(chunk)
    
    # Add to vector store
    mock_vector_store.add_course_metadata(course)
    mock_vector_store.add_course_content(chunks)
    
    return mock_vector_store