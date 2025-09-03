import pytest
from unittest.mock import Mock, patch
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestVectorStore:
    """Test suite for VectorStore functionality"""

    def test_search_results_empty(self):
        """Test SearchResults.empty method"""
        results = SearchResults.empty("No results found")
        assert results.is_empty() is True
        assert results.error == "No results found"
        assert len(results.documents) == 0

    def test_search_results_from_chroma(self):
        """Test SearchResults.from_chroma method"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course': 'test1'}, {'course': 'test2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        assert results.is_empty() is False
        assert len(results.documents) == 2
        assert results.documents[0] == 'doc1'

    def test_vector_store_initialization(self, mock_vector_store):
        """Test VectorStore initialization"""
        assert mock_vector_store.client is not None
        assert mock_vector_store.course_catalog is not None
        assert mock_vector_store.course_content is not None
        assert mock_vector_store.max_results == 5

    def test_search_empty_store(self, mock_vector_store):
        """Test search on empty vector store"""
        results = mock_vector_store.search("test query")
        
        assert results.is_empty() is True
        assert len(results.documents) == 0

    def test_search_with_course_name(self, populated_vector_store):
        """Test search with course name filter"""
        results = populated_vector_store.search("introduction", course_name="Test Course")
        
        assert results.is_empty() is False
        assert len(results.documents) > 0

    def test_search_with_lesson_number(self, populated_vector_store):
        """Test search with lesson number filter"""
        results = populated_vector_store.search("introduction", lesson_number=1)
        
        assert results.is_empty() is False
        assert len(results.documents) > 0

    def test_search_invalid_course_name(self, mock_vector_store):
        """Test search with invalid course name"""
        results = mock_vector_store.search("test query", course_name="Non-existent Course")
        
        assert results.is_empty() is True
        assert "No course found matching" in results.error

    def test_resolve_course_name(self, populated_vector_store):
        """Test course name resolution"""
        resolved = populated_vector_store._resolve_course_name("Test Course")
        
        assert resolved == "Test Course"

    def test_resolve_course_name_not_found(self, mock_vector_store):
        """Test course name resolution for non-existent course"""
        resolved = mock_vector_store._resolve_course_name("Non-existent Course")
        
        assert resolved is None

    def test_build_filter(self, mock_vector_store):
        """Test filter building logic"""
        # Test no filter
        filter_dict = mock_vector_store._build_filter(None, None)
        assert filter_dict is None
        
        # Test course only
        filter_dict = mock_vector_store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}
        
        # Test lesson only
        filter_dict = mock_vector_store._build_filter(None, 1)
        assert filter_dict == {"lesson_number": 1}
        
        # Test both
        filter_dict = mock_vector_store._build_filter("Test Course", 1)
        assert filter_dict == {"$and": [{"course_title": "Test Course"}, {"lesson_number": 1}]}

    def test_add_course_metadata(self, mock_vector_store):
        """Test adding course metadata"""
        course = Course(
            title="Test Course",
            course_link="https://example.com",
            instructor="Test Instructor"
        )
        lesson = Lesson(
            lesson_number=1,
            title="Introduction",
            lesson_link="https://example.com/lesson1"
        )
        course.lessons.append(lesson)
        
        with patch.object(mock_vector_store.course_catalog, 'add'):
            mock_vector_store.add_course_metadata(course)
            # Verify that add was called
            mock_vector_store.course_catalog.add.assert_called_once()

    def test_add_course_content(self, mock_vector_store):
        """Test adding course content"""
        chunks = [
            CourseChunk(
                content="Test content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            )
        ]
        
        with patch.object(mock_vector_store.course_content, 'add'):
            mock_vector_store.add_course_content(chunks)
            # Verify that add was called
            mock_vector_store.course_content.add.assert_called_once()

    def test_get_existing_course_titles(self, mock_vector_store):
        """Test getting existing course titles"""
        with patch.object(mock_vector_store.course_catalog, 'get', return_value={
            'ids': ['Course 1', 'Course 2']
        }):
            titles = mock_vector_store.get_existing_course_titles()
            
            assert len(titles) == 2
            assert 'Course 1' in titles
            assert 'Course 2' in titles

    def test_get_course_count(self, mock_vector_store):
        """Test getting course count"""
        with patch.object(mock_vector_store.course_catalog, 'get', return_value={
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }):
            count = mock_vector_store.get_course_count()
            
            assert count == 3

    def test_clear_all_data(self, mock_vector_store):
        """Test clearing all data"""
        with patch.object(mock_vector_store.client, 'delete_collection'):
            with patch.object(mock_vector_store, '_create_collection'):
                mock_vector_store.clear_all_data()
                # Verify that collections were deleted and recreated
                mock_vector_store.client.delete_collection.assert_any_call("course_catalog")
                mock_vector_store.client.delete_collection.assert_any_call("course_content")

    def test_get_course_link(self, mock_vector_store):
        """Test getting course link"""
        with patch.object(mock_vector_store.course_catalog, 'get', return_value={
            'metadatas': [{'course_link': 'https://example.com'}]
        }):
            link = mock_vector_store.get_course_link("Test Course")
            
            assert link == "https://example.com"

    def test_get_lesson_link(self, mock_vector_store):
        """Test getting lesson link"""
        with patch.object(mock_vector_store.course_catalog, 'get', return_value={
            'metadatas': [{
                'lessons_json': '[{"lesson_number": 1, "lesson_link": "https://example.com/lesson1"}]'
            }]
        }):
            link = mock_vector_store.get_lesson_link("Test Course", 1)
            
            assert link == "https://example.com/lesson1"

    def test_search_error_handling(self, mock_vector_store):
        """Test search error handling"""
        # Mock query to raise exception
        with patch.object(mock_vector_store.course_content, 'query', side_effect=Exception("Search error")):
            results = mock_vector_store.search("test query")
            
            assert results.is_empty() is True
            assert "Search error" in results.error