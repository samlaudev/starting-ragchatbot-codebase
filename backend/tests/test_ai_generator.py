import pytest
from unittest.mock import Mock, patch
from ai_generator import AIGenerator
from search_tools import ToolManager


class TestAIGenerator:
    """Test suite for AI Generator tool calling functionality"""

    def test_init(self):
        """Test AI Generator initialization"""
        generator = AIGenerator("test_api_key", "glm-4.5")
        
        assert generator.client is not None
        assert generator.model == "glm-4.5"
        assert generator.base_params["model"] == "glm-4.5"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
        assert len(generator.SYSTEM_PROMPT) > 0

    def test_system_prompt_content(self):
        """Test that system prompt contains required elements"""
        generator = AIGenerator("test_api_key", "glm-4.5")
        
        # Check that system prompt mentions search tools
        assert "search_course_content" in generator.SYSTEM_PROMPT
        assert "get_course_outline" in generator.SYSTEM_PROMPT
        
        # Check that system prompt contains tool usage guidelines
        assert "Tool Usage Guidelines" in generator.SYSTEM_PROMPT
        assert "One tool use per query maximum" in generator.SYSTEM_PROMPT
        
        # Check that system prompt contains response protocol
        assert "Response Protocol" in generator.SYSTEM_PROMPT
        assert "General knowledge questions" in generator.SYSTEM_PROMPT
        assert "Course content questions" in generator.SYSTEM_PROMPT

    def test_generate_response_without_tools(self, mock_ai_generator):
        """Test direct response generation without tools"""
        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.content = "Direct response to general knowledge question"
        
        with patch.object(mock_ai_generator.client.chat.completions, 'create', return_value=mock_response):
            result = mock_ai_generator.generate_response("What is AI?")
            
            assert result == "Direct response to general knowledge question"
            assert isinstance(result, str)

    def test_generate_response_with_tools(self, mock_ai_generator, mock_tool_manager):
        """Test response generation with tools available"""
        # Get tool definitions
        tools = mock_tool_manager.get_tool_definitions()
        
        # Mock the API response for tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "test_tool_call_id"
        mock_tool_call.function.name = "search_course_content"
        mock_tool_call.function.arguments = "{'query': 'test query', 'course_name': 'Test Course'}"
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.choices[0].message.content = "I'll search for that information."
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.choices = [Mock()]
        mock_final_response.choices[0].finish_reason = "stop"
        mock_final_response.choices[0].message.content = "Found the course content you requested."
        
        with patch.object(mock_ai_generator.client.chat.completions, 'create', side_effect=[mock_response, mock_final_response]):
            result = mock_ai_generator.generate_response(
                "What does the test course cover?",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            assert result == "Found the course content you requested"
            assert isinstance(result, str)

    def test_generate_response_with_conversation_history(self, mock_ai_generator):
        """Test response generation with conversation history"""
        history = "User: What is AI?\nAssistant: AI is artificial intelligence."
        
        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.content = "AI is a fascinating field of study."
        
        with patch.object(mock_ai_generator.client.chat.completions, 'create', return_value=mock_response):
            result = mock_ai_generator.generate_response(
                "Tell me more about AI",
                conversation_history=history
            )
            
            assert result == "AI is a fascinating field of study."
            assert isinstance(result, str)

    def test_handle_tool_execution(self, mock_ai_generator, mock_tool_manager):
        """Test tool execution handling"""
        # Mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "test_tool_call_id"
        mock_tool_call.function.name = "search_course_content"
        mock_tool_call.function.arguments = "{'query': 'test query'}"
        
        # Mock initial response with tool calls
        initial_response = Mock()
        initial_response.choices = [Mock()]
        initial_response.choices[0].finish_reason = "tool_calls"
        initial_response.choices[0].message.content = "I'll search for that."
        initial_response.choices[0].message.tool_calls = [mock_tool_call]
        
        # Mock final response
        final_response = Mock()
        final_response.choices = [Mock()]
        final_response.choices[0].finish_reason = "stop"
        final_response.choices[0].message.content = "Search results: Found content"
        
        # Mock tool manager execution
        mock_tool_manager.execute_tool.return_value = "Found course content"
        
        with patch.object(mock_ai_generator.client.chat.completions, 'create', return_value=final_response):
            result = mock_ai_generator._handle_tool_execution(
                initial_response,
                {"messages": [{"role": "system", "content": "test"}]},
                mock_tool_manager
            )
            
            assert result == "Search results: Found content"
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="test query"
            )

    def test_handle_tool_execution_multiple_tools(self, mock_ai_generator, mock_tool_manager):
        """Test handling multiple tool calls"""
        # Mock multiple tool calls
        mock_tool_call1 = Mock()
        mock_tool_call1.id = "tool_call_1"
        mock_tool_call1.function.name = "search_course_content"
        mock_tool_call1.function.arguments = "{'query': 'test query 1'}"
        
        mock_tool_call2 = Mock()
        mock_tool_call2.id = "tool_call_2"
        mock_tool_call2.function.name = "get_course_outline"
        mock_tool_call2.function.arguments = "{'course_title': 'Test Course'}"
        
        # Mock initial response
        initial_response = Mock()
        initial_response.choices = [Mock()]
        initial_response.choices[0].finish_reason = "tool_calls"
        initial_response.choices[0].message.content = "I'll search and get the outline."
        initial_response.choices[0].message.tool_calls = [mock_tool_call1, mock_tool_call2]
        
        # Mock final response
        final_response = Mock()
        final_response.choices = [Mock()]
        final_response.choices[0].finish_reason = "stop"
        final_response.choices[0].message.content = "Combined results"
        
        # Mock tool manager execution
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        with patch.object(mock_ai_generator.client.chat.completions, 'create', return_value=final_response):
            result = mock_ai_generator._handle_tool_execution(
                initial_response,
                {"messages": [{"role": "system", "content": "test"}]},
                mock_tool_manager
            )
            
            assert result == "Combined results"
            assert mock_tool_manager.execute_tool.call_count == 2

    def test_tool_execution_error_handling(self, mock_ai_generator, mock_tool_manager):
        """Test error handling when tool execution fails"""
        # Mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "test_tool_call_id"
        mock_tool_call.function.name = "search_course_content"
        mock_tool_call.function.arguments = "{'query': 'test query'}"
        
        # Mock initial response
        initial_response = Mock()
        initial_response.choices = [Mock()]
        initial_response.choices[0].finish_reason = "tool_calls"
        initial_response.choices[0].message.content = "I'll search for that."
        initial_response.choices[0].message.tool_calls = [mock_tool_call]
        
        # Mock final response
        final_response = Mock()
        final_response.choices = [Mock()]
        final_response.choices[0].finish_reason = "stop"
        final_response.choices[0].message.content = "Error occurred"
        
        # Mock tool manager to return error
        mock_tool_manager.execute_tool.return_value = "Error: Tool not found"
        
        with patch.object(mock_ai_generator.client.chat.completions, 'create', return_value=final_response):
            result = mock_ai_generator._handle_tool_execution(
                initial_response,
                {"messages": [{"role": "system", "content": "test"}]},
                mock_tool_manager
            )
            
            assert result == "Error occurred"

    def test_api_error_handling(self, mock_ai_generator):
        """Test handling of API errors"""
        # Mock API to raise exception
        with patch.object(mock_ai_generator.client.chat.completions, 'create', side_effect=Exception("API Error")):
            with pytest.raises(Exception) as exc_info:
                mock_ai_generator.generate_response("test query")
            
            assert "API Error" in str(exc_info.value)

    def test_conversation_history_formatting(self, mock_ai_generator):
        """Test that conversation history is properly formatted"""
        history = "User: What is AI?\nAssistant: AI is artificial intelligence."
        
        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.content = "Response"
        
        with patch.object(mock_ai_generator.client.chat.completions, 'create') as mock_create:
            mock_create.return_value = mock_response
            
            mock_ai_generator.generate_response("Tell me more", conversation_history=history)
            
            # Check that the API was called with formatted history
            call_args = mock_create.call_args
            messages = call_args[1]['messages']
            
            # System message should contain history
            system_content = messages[0]['content']
            assert "Previous conversation:" in system_content
            assert history in system_content

    def test_no_conversation_history(self, mock_ai_generator):
        """Test behavior when no conversation history is provided"""
        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.content = "Response"
        
        with patch.object(mock_ai_generator.client.chat.completions, 'create') as mock_create:
            mock_create.return_value = mock_response
            
            mock_ai_generator.generate_response("test query")
            
            # Check that the API was called without history
            call_args = mock_create.call_args
            messages = call_args[1]['messages']
            
            # System message should not contain history
            system_content = messages[0]['content']
            assert "Previous conversation:" not in system_content

    def test_tool_parameter_parsing(self, mock_ai_generator, mock_tool_manager):
        """Test that tool arguments are properly parsed"""
        # Mock tool call with complex arguments
        mock_tool_call = Mock()
        mock_tool_call.id = "test_tool_call_id"
        mock_tool_call.function.name = "search_course_content"
        mock_tool_call.function.arguments = "{'query': 'complex query with spaces', 'course_name': 'Test Course', 'lesson_number': 1}"
        
        # Mock initial response
        initial_response = Mock()
        initial_response.choices = [Mock()]
        initial_response.choices[0].finish_reason = "tool_calls"
        initial_response.choices[0].message.content = "I'll search for that."
        initial_response.choices[0].message.tool_calls = [mock_tool_call]
        
        # Mock final response
        final_response = Mock()
        final_response.choices = [Mock()]
        final_response.choices[0].finish_reason = "stop"
        final_response.choices[0].message.content = "Search results"
        
        with patch.object(mock_ai_generator.client.chat.completions, 'create', return_value=final_response):
            mock_ai_generator._handle_tool_execution(
                initial_response,
                {"messages": [{"role": "system", "content": "test"}]},
                mock_tool_manager
            )
            
            # Check that tool was called with correct arguments
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="complex query with spaces",
                course_name="Test Course",
                lesson_number=1
            )