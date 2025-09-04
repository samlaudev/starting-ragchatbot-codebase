from typing import Any, Dict, List, Optional

from zai import ZhipuAiClient


class AIGenerator:
    """Handles interactions with Zhipu AI's GLM-4.5 API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **search_course_content** - Search for specific course content and materials
2. **get_course_outline** - Get course outline including title, link, and complete lesson list

Tool Usage Guidelines:
- **Use search_course_content** for questions about specific course content, concepts, or detailed educational materials
- **Use get_course_outline** for questions about course structure, lesson lists, or course overview information
- **Multiple tool use allowed** - Use up to 2 tool calls sequentially for complex queries
- **Each tool call is a separate interaction** - You can see tool results before deciding on additional calls
- **Use tools strategically** - Plan your tool usage to gather all necessary information efficiently
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Examples of Sequential Tool Usage:

Example 1: Comparing courses
- User: "Find a course that covers similar topics to lesson 3 of course X"
- Step 1: Use get_course_outline to find lesson 3 topic
- Step 2: Use search_course_content to find similar courses

Example 2: Comprehensive analysis
- User: "What are the prerequisites for course X and what courses build on it?"
- Step 1: Use get_course_outline for course X structure
- Step 2: Use search_course_content to find related/prerequisite courses

Example 3: Multi-lesson research
- User: "Compare the teaching approaches in lesson 5 of course A and lesson 8 of course B"
- Step 1: Use get_course_outline for course A
- Step 2: Use get_lesson_content for lesson 5
- Step 3: Use get_course_outline for course B
- Step 4: Use get_lesson_content for lesson 8

When to Use Multiple Tools:
- When you need information from multiple courses or lessons
- When you need to compare or contrast different materials
- When the first tool result suggests additional research is needed
- When answering complex questions that require comprehensive analysis

When to Stop After One Tool:
- When the single tool result fully answers the question
- When the question is simple and direct
- When additional tools would not provide more relevant information

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use search_course_content tool first, then use additional tools if needed
- **Course outline/structure questions**: Use get_course_outline tool first, then use additional tools if needed
- **Complex multi-part questions**: Use multiple tools sequentially to gather comprehensive information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the tool output"

For course outline queries, ensure your response includes:
- Course title
- Course link
- Complete lesson list with lesson numbers and titles
- Instructor information if available

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = ZhipuAiClient(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with up to max_rounds sequential tool calls.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of sequential tool rounds (default: 2)

        Returns:
            Generated response as string
        """
        # Build initial messages
        messages = self._build_initial_messages(query, conversation_history)

        # Start recursive execution
        return self._execute_round(messages, tools, tool_manager, 0, max_rounds)

    def _execute_round(
        self,
        messages: List[Dict],
        tools: Optional[List],
        tool_manager,
        round_num: int,
        max_rounds: int,
    ) -> str:
        """
        Execute one round of conversation recursively.

        Args:
            messages: Current message history
            tools: Tools available for this round
            tool_manager: Tool manager instance
            round_num: Current round number
            max_rounds: Maximum allowed rounds

        Returns:
            Final response string
        """
        # Build API parameters for this round
        api_params = self._build_api_params(messages, tools, round_num)

        # Make API call
        try:
            response = self._make_api_call(api_params)
        except Exception as e:
            return f"API调用错误: {str(e)}"

        # Check termination conditions
        if self._should_terminate(response, round_num, max_rounds, tool_manager):
            return response.choices[0].message.content

        # Execute tools and update messages
        try:
            updated_messages = self._execute_tools_and_update(
                response, messages, tool_manager
            )
        except Exception as e:
            return f"工具执行错误: {str(e)}"

        # Recurse for next round
        return self._execute_round(
            updated_messages, None, tool_manager, round_num + 1, max_rounds
        )

    def _build_initial_messages(
        self, query: str, conversation_history: Optional[str] = None
    ) -> List[Dict]:
        """
        Build initial message array with system prompt and user query.

        Args:
            query: User's query
            conversation_history: Previous conversation context

        Returns:
            List of message dictionaries
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

    def _build_api_params(
        self, messages: List[Dict], tools: Optional[List], round_num: int
    ) -> Dict:
        """
        Build API parameters for the current round.

        Args:
            messages: Message history
            tools: Available tools
            round_num: Current round number

        Returns:
            API parameters dictionary
        """
        api_params = {**self.base_params, "messages": messages}

        # Only include tools in first round
        if round_num == 0 and tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        return api_params

    def _make_api_call(self, api_params: Dict) -> Any:
        """
        Make API call to GLM-4.5.

        Args:
            api_params: API parameters

        Returns:
            API response
        """
        return self.client.chat.completions.create(**api_params)

    def _should_terminate(
        self, response, round_num: int, max_rounds: int, tool_manager
    ) -> bool:
        """
        Check if recursion should terminate.

        Args:
            response: API response
            round_num: Current round number
            max_rounds: Maximum allowed rounds
            tool_manager: Tool manager instance

        Returns:
            True if should terminate, False otherwise
        """
        # Reached maximum rounds
        if round_num >= max_rounds:
            return True

        # Response has no tool calls
        if response.choices[0].finish_reason != "tool_calls":
            return True

        # No tool manager available
        if not tool_manager:
            return True

        # No tool calls in response
        if not response.choices[0].message.tool_calls:
            return True

        return False

    def _execute_tools_and_update(
        self, response, messages: List[Dict], tool_manager
    ) -> List[Dict]:
        """
        Execute tool calls and update message history.

        Args:
            response: API response with tool calls
            messages: Current message history
            tool_manager: Tool manager instance

        Returns:
            Updated message history
        """
        # Copy messages to avoid modifying original
        updated_messages = messages.copy()

        # Add assistant message with tool calls
        assistant_message = {
            "role": "assistant",
            "content": response.choices[0].message.content,
            "tool_calls": response.choices[0].message.tool_calls,
        }
        updated_messages.append(assistant_message)

        # Execute all tool calls
        for tool_call in response.choices[0].message.tool_calls:
            tool_result = self._execute_single_tool(tool_call, tool_manager)

            # Add tool result message
            updated_messages.append(
                {"tool_call_id": tool_call.id, "role": "tool", "content": tool_result}
            )

        return updated_messages

    def _execute_single_tool(self, tool_call, tool_manager) -> str:
        """
        Execute a single tool call.

        Args:
            tool_call: Tool call object
            tool_manager: Tool manager instance

        Returns:
            Tool execution result
        """
        try:
            # Parse tool arguments
            tool_args = self._parse_tool_args(tool_call.function.arguments)

            # Execute tool
            result = tool_manager.execute_tool(tool_call.function.name, **tool_args)

            return result
        except Exception as e:
            return f"工具执行失败: {tool_call.function.name} - {str(e)}"

    def _parse_tool_args(self, arguments_str: str) -> Dict:
        """
        Parse tool arguments from string.

        Args:
            arguments_str: Arguments as JSON string

        Returns:
            Parsed arguments dictionary
        """
        try:
            import json

            return json.loads(arguments_str)
        except (json.JSONDecodeError, TypeError):
            # Fallback to eval if JSON parsing fails
            return eval(arguments_str)

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Legacy method for backward compatibility.

        Args:
            initial_response: Response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        assistant_message = {
            "role": "assistant",
            "content": initial_response.choices[0].message.content,
            "tool_calls": initial_response.choices[0].message.tool_calls,
        }
        messages.append(assistant_message)

        # Execute all tool calls and collect results
        tool_results = []
        for tool_call in initial_response.choices[0].message.tool_calls:
            # Parse tool arguments safely
            try:
                import json

                tool_args = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                # Fallback to eval if JSON parsing fails
                tool_args = eval(tool_call.function.arguments)

            tool_result = tool_manager.execute_tool(
                tool_call.function.name, **tool_args
            )

            tool_results.append(
                {"tool_call_id": tool_call.id, "role": "tool", "content": tool_result}
            )

        # Add tool results to messages
        messages.extend(tool_results)

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,  # 移除system参数，直接使用messages
        }

        # Get final response
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content
