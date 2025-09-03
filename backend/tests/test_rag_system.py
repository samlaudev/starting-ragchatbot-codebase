import pytest
from unittest.mock import Mock, patch
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class TestRAGSystem:
    """Test suite for RAG system content query handling"""

    def test_rag_system_initialization(self, mock_rag_system):
        """Test RAG system initialization"""
        assert mock_rag_system is not None
        assert mock_rag_system.document_processor is not None
        assert mock_rag_system.vector_store is not None
        assert mock_rag_system.ai_generator is not None
        assert mock_rag_system.session_manager is not None
        assert mock_rag_system.tool_manager is not None

    def test_query_without_session_id(self, mock_rag_system):
        """Test query processing without session ID"""
        # Mock AI generator response
        mock_response = "Test response"
        with patch.object(mock_rag_system.ai_generator, 'generate_response', return_value=(mock_response, [])):
            result, sources = mock_rag_system.query("test query")
            
            assert result == mock_response
            assert sources == []
            # Verify that AI generator was called
            mock_rag_system.ai_generator.generate_response.assert_called_once()

    def test_query_with_session_id(self, mock_rag_system):
        """Test query processing with session ID"""
        # Create a session first
        session_id = mock_rag_system.session_manager.create_session()
        
        # Mock AI generator response
        mock_response = "Test response"
        with patch.object(mock_rag_system.ai_generator, 'generate_response', return_value=(mock_response, [])):
            result, sources = mock_rag_system.query("test query", session_id)
            
            assert result == mock_response
            assert sources == []
            # Verify that conversation history was passed to AI generator
            mock_rag_system.ai_generator.generate_response.assert_called_once()
            call_args = mock_rag_system.ai_generator.generate_response.call_args
            assert call_args[1]['conversation_history'] is not None

    def test_query_conversation_history_tracking(self, mock_rag_system):
        """Test that conversation history is properly tracked"""
        # Create a session
        session_id = mock_rag_system.session_manager.create_session()
        
        # Mock AI generator response
        mock_response = "Test response"
        with patch.object(mock_rag_system.ai_generator, 'generate_response', return_value=(mock_response, [])):
            # Send a query
            mock_rag_system.query("test query", session_id)
            
            # Check that the exchange was added to conversation history
            history = mock_rag_system.session_manager.get_conversation_history(session_id)
            assert history is not None
            assert "test query" in history
            assert "Test response" in history

    def test_query_tool_integration(self, mock_rag_system):
        """Test that tools are properly integrated into query processing"""
        # Mock AI generator response
        mock_response = "Tool-based response"
        with patch.object(mock_rag_system.ai_generator, 'generate_response', return_value=(mock_response, ["Test Course"])):
            result, sources = mock_rag_system.query("test query")
            
            assert result == mock_response
            assert sources == ["Test Course"]
            
            # Verify that tools were passed to AI generator
            call_args = mock_rag_system.ai_generator.generate_response.call_args
            assert 'tools' in call_args[1]
            assert 'tool_manager' in call_args[1]

    def test_add_course_document_success(self, mock_rag_system, sample_course_document):
        """Test successful course document addition"""
        # Mock document processor
        mock_course = Course(
            title="Test Course",
            course_link="https://example.com/course",
            instructor="Test Instructor"
        )
        mock_chunk = CourseChunk(
            content="Test content",
            course_title="Test Course",
            chunk_index=0
        )
        mock_rag_system.document_processor.process_course_document.return_value = (mock_course, [mock_chunk])
        
        # Create temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_course_document)
            temp_file = f.name
        
        try:
            course, chunk_count = mock_rag_system.add_course_document(temp_file)
            
            assert course is not None
            assert chunk_count == 1
            assert course.title == "Test Course"
            
            # Verify that vector store methods were called
            mock_rag_system.vector_store.add_course_metadata.assert_called_once_with(mock_course)
            mock_rag_system.vector_store.add_course_content.assert_called_once_with([mock_chunk])
        finally:
            os.unlink(temp_file)

    def test_add_course_document_error_handling(self, mock_rag_system):
        """Test error handling in course document addition"""
        # Mock document processor to raise exception
        mock_rag_system.document_processor.process_course_document.side_effect = Exception("Processing error")
        
        # Create temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Invalid content")
            temp_file = f.name
        
        try:
            course, chunk_count = mock_rag_system.add_course_document(temp_file)
            
            assert course is None
            assert chunk_count == 0
        finally:
            os.unlink(temp_file)

    def test_add_course_folder(self, mock_rag_system):
        """Test adding course folder"""
        # Mock folder operations
        mock_rag_system.vector_store.get_existing_course_titles.return_value = []
        
        # Mock document processor
        mock_course = Course(title="Test Course")
        mock_chunk = CourseChunk(content="Test content", course_title="Test Course", chunk_index=0)
        mock_rag_system.document_processor.process_course_document.return_value = (mock_course, [mock_chunk])
        
        # Create temporary folder with test file
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("Course Title: Test Course\nCourse Link: https://example.com\nCourse Instructor: Test\n\nLesson 1: Test\nTest content")
            
            courses, chunks = mock_rag_system.add_course_folder(temp_dir, clear_existing=False)
            
            assert courses == 1
            assert chunks == 1
            mock_rag_system.vector_store.add_course_metadata.assert_called_once()
            mock_rag_system.vector_store.add_course_content.assert_called_once()

    def test_add_course_folder_clear_existing(self, mock_rag_system):
        """Test adding course folder with clear existing flag"""
        # Create temporary folder
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            courses, chunks = mock_rag_system.add_course_folder(temp_dir, clear_existing=True)
            
            assert courses == 0
            assert chunks == 0
            mock_rag_system.vector_store.clear_all_data.assert_called_once()

    def test_add_course_folder_nonexistent_folder(self, mock_rag_system):
        """Test adding course folder that doesn't exist"""
        courses, chunks = mock_rag_system.add_course_folder("/nonexistent/folder")
        
        assert courses == 0
        assert chunks == 0

    def test_get_course_analytics(self, mock_rag_system):
        """Test getting course analytics"""
        # Mock vector store analytics
        mock_rag_system.vector_store.get_course_count.return_value = 5
        mock_rag_system.vector_store.get_existing_course_titles.return_value = ["Course 1", "Course 2", "Course 3", "Course 4", "Course 5"]
        
        analytics = mock_rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]

    def test_query_sources_reset(self, mock_rag_system):
        """Test that sources are reset after query"""
        # Mock AI generator response
        mock_response = "Test response"
        with patch.object(mock_rag_system.ai_generator, 'generate_response', return_value=(mock_response, ["Test Course"])):
            with patch.object(mock_rag_system.tool_manager, 'get_last_sources', return_value=["Test Course"]):
                result, sources = mock_rag_system.query("test query")
                
                assert sources == ["Test Course"]
                # Verify that sources were reset
                mock_rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_prompt_formatting(self, mock_rag_system):
        """Test that query prompt is properly formatted"""
        # Mock AI generator response
        mock_response = "Test response"
        with patch.object(mock_rag_system.ai_generator, 'generate_response', return_value=(mock_response, [])):
            mock_rag_system.query("What is machine learning?")
            
            # Verify that the prompt was formatted correctly
            call_args = mock_rag_system.ai_generator.generate_response.call_args
            query = call_args[1]['query']
            assert "course materials" in query
            assert "machine learning" in query

    def test_query_empty_response_handling(self, mock_rag_system):
        """Test handling of empty responses"""
        # Mock AI generator to return empty response
        with patch.object(mock_rag_system.ai_generator, 'generate_response', return_value=("", [])):
            result, sources = mock_rag_system.query("test query")
            
            assert result == ""
            assert sources == []

    def test_conversation_history_limit(self, mock_rag_system):
        """Test that conversation history respects limits"""
        # Create a session with limited history
        session_id = mock_rag_system.session_manager.create_session()
        
        # Mock AI generator response
        mock_response = "Response"
        mock_rag_system.ai_generator.generate_response.return_value = (mock_response, [])
        
        # Send multiple queries to test history limit
        for i in range(5):
            mock_rag_system.query(f"Query {i}", session_id)
        
        # Check that history is limited
        history = mock_rag_system.session_manager.get_conversation_history(session_id)
        assert history is not None
        # Should only have last 4 exchanges (2 max_history * 2 messages per exchange)
        lines = history.split('\n')
        assert len(lines) <= 8  # 4 exchanges * 2 lines each

    def test_tool_manager_initialization(self, mock_rag_system):
        """Test that tool manager is properly initialized"""
        # Check that tools are registered
        tool_defs = mock_rag_system.tool_manager.get_tool_definitions()
        assert len(tool_defs) >= 2  # Should have search and outline tools
        
        # Check that search tool is registered
        search_tool_def = next((t for t in tool_defs if t['function']['name'] == 'search_course_content'), None)
        assert search_tool_def is not None
        
        # Check that outline tool is registered
        outline_tool_def = next((t for t in tool_defs if t['function']['name'] == 'get_course_outline'), None)
        assert outline_tool_def is not None

    def test_duplicate_course_handling(self, mock_rag_system):
        """Test handling of duplicate courses"""
        # Mock existing courses
        mock_rag_system.vector_store.get_existing_course_titles.return_value = ["Existing Course"]
        
        # Mock document processor
        mock_course = Course(title="Existing Course")
        mock_chunk = CourseChunk(content="Test content", course_title="Existing Course", chunk_index=0)
        mock_rag_system.document_processor.process_course_document.return_value = (mock_course, [mock_chunk])
        
        # Create temporary folder with test file
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("Course Title: Existing Course\nCourse Link: https://example.com\nCourse Instructor: Test\n\nLesson 1: Test\nTest content")
            
            courses, chunks = mock_rag_system.add_course_folder(temp_dir, clear_existing=False)
            
            assert courses == 0  # Should not add duplicate
            assert chunks == 0
            # Vector store should not be called for duplicates
            mock_rag_system.vector_store.add_course_metadata.assert_not_called()
            mock_rag_system.vector_store.add_course_content.assert_not_called()