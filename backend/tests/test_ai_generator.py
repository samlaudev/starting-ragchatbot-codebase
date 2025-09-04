import pytest
from unittest.mock import Mock, patch
from ai_generator import AIGenerator


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
        
        # Check that system prompt supports multiple tool calls
        assert "Multiple tool use allowed" in generator.SYSTEM_PROMPT
        assert "up to 2 tool calls sequentially" in generator.SYSTEM_PROMPT

    def test_system_prompt_content(self):
        """Test that system prompt contains required elements"""
        generator = AIGenerator("test_api_key", "glm-4.5")
        
        # Check that system prompt mentions search tools
        assert "search_course_content" in generator.SYSTEM_PROMPT
        assert "get_course_outline" in generator.SYSTEM_PROMPT
        
        # Check that system prompt contains updated tool usage guidelines
        assert "Tool Usage Guidelines" in generator.SYSTEM_PROMPT
        assert "Multiple tool use allowed" in generator.SYSTEM_PROMPT
        assert "up to 2 tool calls sequentially" in generator.SYSTEM_PROMPT
        
        # Check that system prompt contains examples of sequential usage
        assert "Examples of Sequential Tool Usage" in generator.SYSTEM_PROMPT
        assert "When to Use Multiple Tools" in generator.SYSTEM_PROMPT
        assert "When to Stop After One Tool" in generator.SYSTEM_PROMPT
        
        # Check that system prompt contains response protocol
        assert "Response Protocol" in generator.SYSTEM_PROMPT
        assert "General knowledge questions" in generator.SYSTEM_PROMPT
        assert "Course content questions" in generator.SYSTEM_PROMPT
        assert "Complex multi-part questions" in generator.SYSTEM_PROMPT

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
    
    def test_generate_response_with_max_rounds_parameter(self, mock_ai_generator):
        """Test that max_rounds parameter is properly handled"""
        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.content = "Response with custom max_rounds"
        
        with patch.object(mock_ai_generator.client.chat.completions, 'create', return_value=mock_response):
            # Test with custom max_rounds
            result = mock_ai_generator.generate_response("What is AI?", max_rounds=1)
            
            assert result == "Response with custom max_rounds"
            assert isinstance(result, str)
            
            # Test with default max_rounds
            result = mock_ai_generator.generate_response("What is AI?")
            
            assert result == "Response with custom max_rounds"
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
            
            assert result == "Found the course content you requested."
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
        mock_tool_manager.execute_tool = Mock(return_value="Found course content")
        
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
        mock_tool_manager.execute_tool = Mock(side_effect=["Result 1", "Result 2"])
        
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
        mock_tool_manager.execute_tool = Mock(return_value="Error: Tool not found")
        
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
        
        # Mock tool manager execution for this test
        mock_tool_manager.execute_tool = Mock(return_value="Tool executed successfully")
        
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
    
    def test_sequential_tool_calls(self, mock_ai_generator, mock_tool_manager):
        """Test two rounds of sequential tool calls"""
        # First round: get course outline
        mock_tool_call_1 = Mock()
        mock_tool_call_1.id = "tool_call_1"
        mock_tool_call_1.function.name = "get_course_outline"
        mock_tool_call_1.function.arguments = "{'course_title': 'Course X'}"
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].finish_reason = "tool_calls"
        mock_response_1.choices[0].message.content = "Let me get the course outline."
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call_1]
        
        # Second round: search for related content
        mock_tool_call_2 = Mock()
        mock_tool_call_2.id = "tool_call_2"
        mock_tool_call_2.function.name = "search_course_content"
        mock_tool_call_2.function.arguments = "{'query': 'lesson 3 topic'}"
        
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].finish_reason = "tool_calls"
        mock_response_2.choices[0].message.content = "Now let me search for related content."
        mock_response_2.choices[0].message.tool_calls = [mock_tool_call_2]
        
        # Final response
        mock_final_response = Mock()
        mock_final_response.choices = [Mock()]
        mock_final_response.choices[0].finish_reason = "stop"
        mock_final_response.choices[0].message.content = "Based on both searches, I found..."
        
        # Mock tool execution
        mock_tool_manager.execute_tool = Mock(side_effect=[
            "Course outline: Lesson 3 is about Advanced Topics",
            "Found related courses covering Advanced Topics"
        ])
        
        with patch.object(mock_ai_generator.client.chat.completions, 'create', side_effect=[mock_response_1, mock_response_2, mock_final_response]):
            result = mock_ai_generator.generate_response(
                "Find courses similar to lesson 3 of Course X",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager
            )
            
            assert "Based on both searches" in result
            assert mock_tool_manager.execute_tool.call_count == 2
            mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_title="Course X")
            mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="lesson 3 topic")
    
    def test_max_rounds_limit(self, mock_ai_generator, mock_tool_manager):
        """Test that max_rounds limit is respected"""
        # Create a tool call that would continue indefinitely
        mock_tool_call = Mock()
        mock_tool_call.id = "tool_call_id"
        mock_tool_call.function.name = "search_course_content"
        mock_tool_call.function.arguments = "{'query': 'test'}"
        
        # Mock tool response (requests another tool call)
        mock_tool_response = Mock()
        mock_tool_response.choices = [Mock()]
        mock_tool_response.choices[0].finish_reason = "tool_calls"
        mock_tool_response.choices[0].message.content = "I need to search again."
        mock_tool_response.choices[0].message.tool_calls = [mock_tool_call]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.choices = [Mock()]
        mock_final_response.choices[0].finish_reason = "stop"
        mock_final_response.choices[0].message.content = "Final response after max rounds."
        
        # Mock tool execution
        mock_tool_manager.execute_tool = Mock(return_value="Some result")
        
        # API calls: 2 tool calls + 1 final response
        with patch.object(mock_ai_generator.client.chat.completions, 'create', side_effect=[mock_tool_response, mock_tool_response, mock_final_response]):
            result = mock_ai_generator.generate_response(
                "Complex query",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
                max_rounds=2  # Limit to 2 rounds
            )
            
            # Should stop after 2 rounds
            assert mock_tool_manager.execute_tool.call_count == 2
            assert result == "Final response after max rounds."
    
    def test_single_tool_call_backward_compatibility(self, mock_ai_generator, mock_tool_manager):
        """Test that single tool calls still work (backward compatibility)"""
        # Mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "test_tool_call_id"
        mock_tool_call.function.name = "search_course_content"
        mock_tool_call.function.arguments = "{'query': 'test query'}"
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.choices[0].message.content = "I'll search for that."
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.choices = [Mock()]
        mock_final_response.choices[0].finish_reason = "stop"
        mock_final_response.choices[0].message.content = "Found the course content."
        
        # Mock tool execution
        mock_tool_manager.execute_tool = Mock(return_value="Found course content")
        
        with patch.object(mock_ai_generator.client.chat.completions, 'create', side_effect=[mock_response, mock_final_response]):
            result = mock_ai_generator.generate_response(
                "What does the test course cover?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager
            )
            
            assert result == "Found the course content."
            assert mock_tool_manager.execute_tool.call_count == 1
            mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="test query")
    
    def test_api_error_handling(self, mock_ai_generator, mock_tool_manager):
        """Test API error handling in new architecture"""
        with patch.object(mock_ai_generator.client.chat.completions, 'create', side_effect=Exception("API Error")):
            result = mock_ai_generator.generate_response(
                "Test query",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager
            )
            
            assert "API调用错误" in result
    
    def test_tool_execution_error_handling(self, mock_ai_generator, mock_tool_manager):
        """Test tool execution error handling in new architecture"""
        # Mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "test_tool_call_id"
        mock_tool_call.function.name = "search_course_content"
        mock_tool_call.function.arguments = "{'query': 'test'}"
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.choices[0].message.content = "I'll search for that."
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        
        # Mock tool execution error
        mock_tool_manager.execute_tool = Mock(return_value="Error: Tool not found")
        
        mock_final_response = Mock()
        mock_final_response.choices = [Mock()]
        mock_final_response.choices[0].finish_reason = "stop"
        mock_final_response.choices[0].message.content = "Response after tool error."
        
        with patch.object(mock_ai_generator.client.chat.completions, 'create', side_effect=[mock_response, mock_final_response]):
            result = mock_ai_generator.generate_response(
                "Test query",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager
            )
            
            assert "Response after tool error" in result
            mock_tool_manager.execute_tool.assert_called_once()
    
    def test_build_initial_messages(self, mock_ai_generator):
        """Test _build_initial_messages method"""
        # Test without conversation history
        messages = mock_ai_generator._build_initial_messages("What is AI?")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is AI?"
        assert "Previous conversation:" not in messages[0]["content"]
        
        # Test with conversation history
        history = "User: Hello\nAssistant: Hi there!"
        messages = mock_ai_generator._build_initial_messages("What is AI?", history)
        assert len(messages) == 2
        assert "Previous conversation:" in messages[0]["content"]
        assert history in messages[0]["content"]
    
    def test_build_api_params(self, mock_ai_generator):
        """Test _build_api_params method"""
        messages = [{"role": "system", "content": "test"}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        
        # Test first round with tools
        params = mock_ai_generator._build_api_params(messages, tools, 0)
        assert "tools" in params
        assert "tool_choice" in params
        assert params["tools"] == tools
        
        # Test second round (no tools)
        params = mock_ai_generator._build_api_params(messages, tools, 1)
        assert "tools" not in params
        assert "tool_choice" not in params
        
        # Test without tools
        params = mock_ai_generator._build_api_params(messages, None, 0)
        assert "tools" not in params
        assert "tool_choice" not in params
    
    def test_should_terminate(self, mock_ai_generator, mock_tool_manager):
        """Test _should_terminate method"""
        # Create mock response with tool calls
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.choices[0].message.tool_calls = [Mock()]
        
        # Test termination conditions
        assert mock_ai_generator._should_terminate(mock_response, 3, 2, mock_tool_manager) == True  # Max rounds exceeded
        assert mock_ai_generator._should_terminate(mock_response, 1, 2, None) == True  # No tool manager
        assert mock_ai_generator._should_terminate(mock_response, 1, 2, mock_tool_manager) == False  # Should continue
        
        # Test response without tool calls
        mock_response_no_tools = Mock()
        mock_response_no_tools.choices = [Mock()]
        mock_response_no_tools.choices[0].finish_reason = "stop"
        mock_response_no_tools.choices[0].message.tool_calls = []
        assert mock_ai_generator._should_terminate(mock_response_no_tools, 0, 2, mock_tool_manager) == True