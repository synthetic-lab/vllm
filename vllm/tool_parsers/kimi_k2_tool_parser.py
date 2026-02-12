# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

logger = init_logger(__name__)


class KimiK2ToolParser(ToolParser):
    """
    Tool parser for Kimi K2 models.

    Format:
    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.{name}:{index}<|tool_call_argument_begin|>{json}<|tool_call_end|>
    <|tool_calls_section_end|>
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        # Markers
        self.tool_calls_start_token: str = "<|tool_calls_section_begin|>"
        self.tool_calls_end_token: str = "<|tool_calls_section_end|>"
        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"
        self.tool_call_arg_token: str = "<|tool_call_argument_begin|>"

        # Support both singular and plural variants for section markers
        self.tool_calls_start_token_variants: list[str] = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_section_begin|>",
        ]
        self.tool_calls_end_token_variants: list[str] = [
            "<|tool_calls_section_end|>",
            "<|tool_call_section_end|>",
        ]

        # Regex for non-streaming (complete tool call matching)
        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<]+:\d+)\s*"
            r"<\|tool_call_argument_begin\|>\s*"
            r"(?P<function_arguments>(?:(?!<\|tool_call_begin\|>).)*?)\s*"
            r"<\|tool_call_end\|>",
            re.DOTALL,
        )

        # Regex for streaming - includes start marker like sglang
        self.stream_tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<:\s]+:\d+)\s*"
            r"<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*)",
            re.DOTALL,
        )

        # Streaming state (public for backwards compatibility with tests)
        self.token_buffer: str = ""
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self.prev_tool_call_arr: list[dict] = []
        self.section_char_count: int = 0  # For backwards compatibility

        # Internal state
        self._last_arguments: str = ""

    @property
    def in_tool_section(self) -> bool:
        """Whether we're currently inside a tool call section."""
        return self._has_tool_call_markers(self.token_buffer)

    @in_tool_section.setter
    def in_tool_section(self, value: bool) -> None:
        """Setter for backwards compatibility - clears buffer if set to False."""
        if not value:
            self.token_buffer = ""

    def reset_streaming_state(self) -> None:
        """Reset all streaming state between requests."""
        self.token_buffer = ""
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []
        self.prev_tool_call_arr = []
        self.section_char_count = 0
        self._last_arguments = ""
        logger.debug("Streaming state reset")

    def _has_tool_call_markers(self, text: str) -> bool:
        """Check if text contains any tool call markers (string-based)."""
        # Check for section markers (any variant)
        for variant in self.tool_calls_start_token_variants:
            if variant in text:
                return True
        # Check for individual tool call marker
        if self.tool_call_start_token in text:
            return True
        return False

    def _strip_section_markers(self, text: str) -> str:
        """Strip section begin/end markers from text."""
        result = text
        for variant in self.tool_calls_start_token_variants:
            result = result.replace(variant, "")
        for variant in self.tool_calls_end_token_variants:
            result = result.replace(variant, "")
        return result

    def _extract_content_before_tools(self, text: str) -> str:
        """Extract content that appears before tool call section."""
        for variant in self.tool_calls_start_token_variants:
            if variant in text:
                return text[: text.find(variant)]
        return text

    def _parse_tool_id(self, tool_id: str) -> str:
        """Extract function name from tool_id like 'functions.get_weather:0'."""
        # Format: functions.name:index or name:index
        name_part = tool_id.split(":")[0]
        return name_part.split(".")[-1]

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output (non-streaming)."""
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            function_call_tuples = self.tool_call_regex.findall(model_output)
            logger.debug("function_call_tuples: %s", function_call_tuples)

            tool_calls = []
            for function_id, function_args in function_call_tuples:
                function_name = self._parse_tool_id(function_id)
                tool_calls.append(
                    ToolCall(
                        id=function_id,
                        type="function",
                        function=FunctionCall(
                            name=function_name, arguments=function_args
                        ),
                    )
                )

            content = self._extract_content_before_tools(model_output)
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _has_section_end(self, text: str) -> bool:
        """Check if text contains a section end marker."""
        for variant in self.tool_calls_end_token_variants:
            if variant in text:
                return True
        return False

    def _extract_post_section_content(self, text: str) -> str:
        """Extract any content that appears after the section end marker."""
        for variant in self.tool_calls_end_token_variants:
            if variant in text:
                parts = text.split(variant, 1)
                if len(parts) > 1:
                    return parts[1]
        return ""

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """
        Extract tool calls from streaming output.

        Uses simple string-based detection and regex parsing,
        following the sglang approach for reliability.
        """
        logger.debug("delta_text: %s", delta_text)

        # Accumulate text in buffer
        self.token_buffer += delta_text

        # Check for section end - if found, process any remaining tool calls
        # then clear buffer and return any post-section content
        if self._has_section_end(self.token_buffer):
            # Process any pending tool calls first
            result = None
            if self._has_tool_call_markers(self.token_buffer):
                try:
                    result = self._process_tool_call_buffer()
                except Exception:
                    logger.exception("Error in streaming tool call parsing")

            # Extract content after section end (if any)
            post_content = self._extract_post_section_content(self.token_buffer)

            # Clear the buffer - section is complete
            self.token_buffer = ""

            # If we got a tool call result, return it
            # Otherwise return empty or post-section content
            if result is not None:
                return result
            if post_content.strip():
                return DeltaMessage(content=post_content)
            return DeltaMessage(content="")

        # Check if we have any tool call markers in the accumulated buffer
        if not self._has_tool_call_markers(self.token_buffer):
            # No tool calls yet - return delta as regular content
            # Clear buffer since we know there's no partial marker
            # (we would have detected the start of one)
            self.token_buffer = ""
            # Strip any stray end markers that might appear
            clean_delta = delta_text
            for variant in self.tool_calls_end_token_variants:
                clean_delta = clean_delta.replace(variant, "")
            clean_delta = clean_delta.replace(self.tool_call_end_token, "")
            if clean_delta:
                return DeltaMessage(content=clean_delta)
            return None

        # We have tool call markers - process the buffer
        try:
            return self._process_tool_call_buffer()
        except Exception:
            logger.exception("Error in streaming tool call parsing")
            return None

    def _process_tool_call_buffer(self) -> DeltaMessage | None:
        """Process the accumulated buffer for tool calls."""
        # Strip section markers from buffer for cleaner parsing
        working_text = self._strip_section_markers(self.token_buffer)

        # Try to match a tool call in progress
        match = self.stream_tool_call_regex.search(working_text)

        if not match:
            # Have markers but no parseable tool call yet - suppress output
            # This handles the case between section_begin and first tool_call_begin
            return DeltaMessage(content="")

        tool_id = match.group("tool_call_id")
        function_args = match.group("function_arguments")
        function_name = self._parse_tool_id(tool_id)

        # Initialize state for first tool call
        if self.current_tool_id == -1:
            self.current_tool_id = 0
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = [""]
            self._last_arguments = ""

        # Ensure tracking arrays are large enough
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")

        # Case 1: Haven't sent tool name yet
        if not self.current_tool_name_sent:
            self.current_tool_name_sent = True
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": function_name,
                "arguments": "",
            }
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=tool_id,
                        function=DeltaFunctionCall(name=function_name).model_dump(
                            exclude_none=True
                        ),
                    )
                ]
            )

        # Case 2: Streaming arguments
        # Get the arguments portion, stopping at tool_call_end if present
        args_to_parse = function_args.split(self.tool_call_end_token, 1)[0]

        # Calculate diff from last streamed arguments
        if args_to_parse.startswith(self._last_arguments):
            argument_diff = args_to_parse[len(self._last_arguments) :]
        else:
            argument_diff = args_to_parse

        result: DeltaMessage | None = None

        if argument_diff:
            self._last_arguments = args_to_parse
            self.streamed_args_for_tool[self.current_tool_id] += argument_diff
            result = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        function=DeltaFunctionCall(arguments=argument_diff).model_dump(
                            exclude_none=True
                        ),
                    )
                ]
            )

        # Check if this tool call is complete
        if self.tool_call_end_token in function_args:
            # Tool call is complete - prepare for next one
            # Remove the completed tool call from buffer
            tool_call_end_pattern = r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>"
            end_match = re.search(tool_call_end_pattern, working_text, re.DOTALL)
            if end_match:
                # Keep anything after the completed tool call
                remaining = working_text[end_match.end() :]
                self.token_buffer = remaining
            else:
                self.token_buffer = ""

            # Reset for next tool call
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            self._last_arguments = ""

            # Ensure we return something even if no diff
            if result is None:
                result = DeltaMessage(content="")

        return result
