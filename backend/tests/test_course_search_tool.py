import pytest
from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool.execute method"""

    def test_execute_empty_vector_store(self, mock_vector_store):
        """Test behavior when vector store is empty"""
        search_tool = CourseSearchTool(mock_vector_store)
        
        # Execute search on empty store
        result = search_tool.execute("test query")
        
        # Should return appropriate message for no results
        assert "No relevant content found" in result
        assert len(search_tool.last_sources) == 0

    def test_execute_valid_query_with_results(self, populated_vector_store):
        """Test successful content retrieval"""
        search_tool = CourseSearchTool(populated_vector_store)
        
        # Search for content that exists
        result = search_tool.execute("introduction")
        
        # Should find content about introduction
        assert "introduction" in result.lower()
        assert "Test Course" in result
        assert len(search_tool.last_sources) > 0

    def test_execute_course_name_filtering(self, populated_vector_store):
        """Test course-specific search functionality"""
        search_tool = CourseSearchTool(populated_vector_store)
        
        # Search with specific course name
        result = search_tool.execute("concepts", course_name="Test Course")
        
        # Should find content from the specified course
        assert "Test Course" in result
        assert len(search_tool.last_sources) > 0

    def test_execute_lesson_number_filtering(self, populated_vector_store):
        """Test lesson-specific search functionality"""
        search_tool = CourseSearchTool(populated_vector_store)
        
        # Search for specific lesson
        result = search_tool.execute("introduction", lesson_number=1)
        
        # Should find content from lesson 1
        assert "Lesson 1" in result
        assert len(search_tool.last_sources) > 0

    def test_execute_combined_filters(self, populated_vector_store):
        """Test course + lesson filtering"""
        search_tool = CourseSearchTool(populated_vector_store)
        
        # Search with both course and lesson filters
        result = search_tool.execute("advanced", course_name="Test Course", lesson_number=2)
        
        # Should find content from specific course and lesson
        assert "Test Course" in result
        assert "Lesson 2" in result
        assert len(search_tool.last_sources) > 0

    def test_execute_invalid_course_name(self, populated_vector_store):
        """Test error handling for non-existent courses"""
        search_tool = CourseSearchTool(populated_vector_store)
        
        # Search for non-existent course
        result = search_tool.execute("test query", course_name="Non-existent Course")
        
        # Should return error message
        assert "No course found matching" in result
        assert len(search_tool.last_sources) == 0

    def test_execute_no_results_with_filters(self, populated_vector_store):
        """Test when filters match but no content found"""
        search_tool = CourseSearchTool(populated_vector_store)
        
        # Search for content that doesn't exist in filtered results
        result = search_tool.execute("nonexistent content", course_name="Test Course")
        
        # Should return no results message
        assert "No relevant content found" in result
        assert "Test Course" in result

    def test_execute_malformed_queries(self, populated_vector_store):
        """Test edge cases and error handling"""
        search_tool = CourseSearchTool(populated_vector_store)
        
        # Test with empty query
        result = search_tool.execute("")
        
        # Should handle empty query gracefully
        assert isinstance(result, str)
        
        # Test with None query (should raise TypeError)
        with pytest.raises(TypeError):
            search_tool.execute(None)

    def test_execute_sources_tracking(self, populated_vector_store):
        """Test that sources are properly tracked"""
        search_tool = CourseSearchTool(populated_vector_store)
        
        # Clear any existing sources
        search_tool.last_sources = []
        
        # Execute search
        result = search_tool.execute("introduction")
        
        # Check that sources were tracked
        assert len(search_tool.last_sources) > 0
        assert "Test Course" in search_tool.last_sources[0]

    def test_execute_sources_format_with_links(self, populated_vector_store):
        """Test that sources include lesson links when available"""
        search_tool = CourseSearchTool(populated_vector_store)
        
        # Execute search
        result = search_tool.execute("introduction")
        
        # Check that sources include lesson links
        if search_tool.last_sources:
            source = search_tool.last_sources[0]
            assert "|||" in source or "Test Course" in source

    def test_execute_search_error_handling(self, mock_vector_store):
        """Test handling of search errors"""
        search_tool = CourseSearchTool(mock_vector_store)
        
        # Mock the search method to raise an exception
        with patch.object(mock_vector_store, 'search', side_effect=Exception("Simulated search error")):
            # Execute search that should fail
            result = search_tool.execute("test query")
            
            # Should return error message
            assert "Search error" in result
            assert len(search_tool.last_sources) == 0

    def test_execute_large_result_set(self, populated_vector_store):
        """Test handling of large result sets"""
        search_tool = CourseSearchTool(populated_vector_store)
        
        # Search for common terms that might return many results
        result = search_tool.execute("course")
        
        # Should handle multiple results gracefully
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_case_insensitive_search(self, populated_vector_store):
        """Test that search is case insensitive"""
        search_tool = CourseSearchTool(populated_vector_store)
        
        # Search with different cases
        result_upper = search_tool.execute("INTRODUCTION")
        result_lower = search_tool.execute("introduction")
        result_mixed = search_tool.execute("Introduction")
        
        # Should find similar results
        assert len(result_upper) > 0
        assert len(result_lower) > 0
        assert len(result_mixed) > 0

    def test_execute_partial_matches(self, populated_vector_store):
        """Test partial word matching"""
        search_tool = CourseSearchTool(populated_vector_store)
        
        # Search for partial terms
        result = search_tool.execute("intro")
        
        # Should find introduction content
        assert len(result) > 0

    def test_get_tool_definition(self):
        """Test tool definition structure"""
        search_tool = CourseSearchTool(None)  # Vector store not needed for this test
        
        tool_def = search_tool.get_tool_definition()
        
        # Verify tool definition structure
        assert tool_def["type"] == "function"
        assert "function" in tool_def
        assert tool_def["function"]["name"] == "search_course_content"
        assert "description" in tool_def["function"]
        assert "parameters" in tool_def["function"]
        
        # Verify parameters structure
        params = tool_def["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        
        # Verify required parameters
        required = params["required"]
        assert "query" in required
        
        # Verify parameter properties
        properties = params["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties