import pytest
from unittest.mock import Mock, patch
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool


class TestToolManager:
    """Test suite for ToolManager functionality"""

    def test_tool_manager_initialization(self):
        """Test ToolManager initialization"""
        tool_manager = ToolManager()
        assert tool_manager.tools == {}

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        
        tool_manager.register_tool(search_tool)
        
        assert "search_course_content" in tool_manager.tools
        assert tool_manager.tools["search_course_content"] == search_tool

    def test_register_tool_with_old_format(self):
        """Test tool registration with old format (for compatibility)"""
        tool_manager = ToolManager()
        
        # Create a mock tool with old format
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {
            "name": "old_tool",
            "description": "Old format tool"
        }
        
        tool_manager.register_tool(mock_tool)
        
        assert "old_tool" in tool_manager.tools

    def test_register_tool_without_name(self):
        """Test tool registration without name raises error"""
        tool_manager = ToolManager()
        
        # Create a mock tool without name
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {
            "description": "Tool without name"
        }
        
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            tool_manager.register_tool(mock_tool)

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting tool definitions"""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        tool_manager.register_tool(search_tool)
        tool_manager.register_tool(outline_tool)
        
        definitions = tool_manager.get_tool_definitions()
        
        assert len(definitions) == 2
        assert any(d['function']['name'] == 'search_course_content' for d in definitions)
        assert any(d['function']['name'] == 'get_course_outline' for d in definitions)

    def test_execute_tool_success(self, mock_vector_store):
        """Test successful tool execution"""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        
        # Mock the execute method
        with patch.object(search_tool, 'execute', return_value="Search results"):
            tool_manager.register_tool(search_tool)
            
            result = tool_manager.execute_tool("search_course_content", query="test")
            
            assert result == "Search results"
            search_tool.execute.assert_called_once_with(query="test")

    def test_execute_tool_not_found(self):
        """Test executing non-existent tool"""
        tool_manager = ToolManager()
        
        result = tool_manager.execute_tool("nonexistent_tool")
        
        assert "not found" in result

    def test_get_last_sources(self, mock_vector_store):
        """Test getting last sources from tools"""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        # Set up sources in search tool
        search_tool.last_sources = ["Source 1", "Source 2"]
        outline_tool.last_sources = []
        
        tool_manager.register_tool(search_tool)
        tool_manager.register_tool(outline_tool)
        
        sources = tool_manager.get_last_sources()
        
        assert sources == ["Source 1", "Source 2"]

    def test_get_last_sources_empty(self, mock_vector_store):
        """Test getting last sources when none exist"""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        
        # No sources set up
        search_tool.last_sources = []
        
        tool_manager.register_tool(search_tool)
        
        sources = tool_manager.get_last_sources()
        
        assert sources == []

    def test_reset_sources(self, mock_vector_store):
        """Test resetting sources from all tools"""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        # Set up sources
        search_tool.last_sources = ["Source 1"]
        outline_tool.last_sources = ["Source 2"]
        
        tool_manager.register_tool(search_tool)
        tool_manager.register_tool(outline_tool)
        
        tool_manager.reset_sources()
        
        assert search_tool.last_sources == []
        assert outline_tool.last_sources == []

    def test_get_last_sources_tools_without_sources(self, mock_vector_store):
        """Test getting last sources when tools don't have sources attribute"""
        tool_manager = ToolManager()
        
        # Create a mock tool without last_sources attribute
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "Mock tool"
            }
        }
        # Ensure the tool doesn't have last_sources attribute
        if hasattr(mock_tool, 'last_sources'):
            delattr(mock_tool, 'last_sources')
        
        tool_manager.register_tool(mock_tool)
        
        sources = tool_manager.get_last_sources()
        
        assert sources == []

    def test_reset_sources_tools_without_sources(self, mock_vector_store):
        """Test resetting sources when tools don't have sources attribute"""
        tool_manager = ToolManager()
        
        # Create a mock tool without last_sources attribute
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "Mock tool"
            }
        }
        # Ensure the tool doesn't have last_sources attribute
        if hasattr(mock_tool, 'last_sources'):
            delattr(mock_tool, 'last_sources')
        
        tool_manager.register_tool(mock_tool)
        
        # Should not raise an error
        tool_manager.reset_sources()


class TestCourseOutlineTool:
    """Test suite for CourseOutlineTool functionality"""

    def test_course_outline_tool_initialization(self, mock_vector_store):
        """Test CourseOutlineTool initialization"""
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        assert outline_tool.store == mock_vector_store
        assert outline_tool.last_sources == []

    def test_get_tool_definition(self, mock_vector_store):
        """Test CourseOutlineTool tool definition"""
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        tool_def = outline_tool.get_tool_definition()
        
        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "get_course_outline"
        assert "description" in tool_def["function"]
        assert "parameters" in tool_def["function"]
        
        # Verify parameters
        params = tool_def["function"]["parameters"]
        assert params["type"] == "object"
        assert "course_title" in params["properties"]
        assert "course_title" in params["required"]

    def test_execute_success(self, mock_vector_store):
        """Test successful course outline execution"""
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        # Mock vector store responses
        with patch.object(mock_vector_store, '_resolve_course_name', return_value="Test Course"):
            with patch.object(mock_vector_store.course_catalog, 'get', return_value={
                'metadatas': [{
                    'title': 'Test Course',
                    'course_link': 'https://example.com',
                    'instructor': 'Test Instructor',
                    'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "https://example.com/lesson1"}]'
                }]
            }):
                result = outline_tool.execute("Test Course")
                
                assert "Test Course" in result
                assert "Test Instructor" in result
                assert "https://example.com" in result
                assert "Lesson 1" in result
                assert "Introduction" in result
                assert len(outline_tool.last_sources) > 0

    def test_execute_course_not_found(self, mock_vector_store):
        """Test course outline execution when course not found"""
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        # Mock vector store to return None for course resolution
        with patch.object(mock_vector_store, '_resolve_course_name', return_value=None):
            result = outline_tool.execute("Non-existent Course")
            
            assert "No course found matching" in result
            assert len(outline_tool.last_sources) == 0

    def test_execute_metadata_not_found(self, mock_vector_store):
        """Test course outline execution when metadata not found"""
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        # Mock vector store responses
        with patch.object(mock_vector_store, '_resolve_course_name', return_value="Test Course"):
            with patch.object(mock_vector_store.course_catalog, 'get', return_value={
                'metadatas': []  # Empty metadata
            }):
                result = outline_tool.execute("Test Course")
                
                assert "Course metadata not found" in result
                assert len(outline_tool.last_sources) == 0

    def test_execute_error_handling(self, mock_vector_store):
        """Test course outline execution error handling"""
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        # Mock vector store to raise exception
        with patch.object(mock_vector_store, '_resolve_course_name', side_effect=Exception("Test error")):
            result = outline_tool.execute("Test Course")
            
            assert "Error retrieving course outline" in result
            assert len(outline_tool.last_sources) == 0