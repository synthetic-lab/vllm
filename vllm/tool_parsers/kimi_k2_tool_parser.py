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

# ResponsesRequest is needed by adjust_request so it can accept both
# ChatCompletionRequest and ResponsesRequest types.
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)

# partial_tag_overlap detects partial marker matches at the end of a string,
# e.g. text ending in "<|tool_call" when the full marker is "<|tool_call_end|>".
# This prevents leaking half-formed marker bytes into streamed content.
from vllm.tool_parsers.utils import partial_tag_overlap

logger = init_logger(__name__)


class KimiK2ToolParser(ToolParser):
    """
    Tool parser for Kimi K2 models.

    Format:
    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.{name}:{index}<|tool_call_argument_begin|>{json}<|tool_call_end|>
    <|tool_calls_section_end|>
    """

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

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
        # Tracks how many characters of pre-section content have already been
        # emitted in streaming.  Without this, content arriving in the same
        # chunk as the section-begin marker would be silently dropped because
        # the old buffer-based approach only emitted content *before* the
        # marker was seen, and never revisited earlier text.
        self._sent_content_idx: int = 0

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
        self._sent_content_idx = 0
        logger.debug("Streaming state reset")

    def _has_tool_call_markers(self, text: str) -> bool:
        """Check if text contains any tool call markers (string-based)."""
        # Check for section markers (any variant)
        for variant in self.tool_calls_start_token_variants:
            if variant in text:
                return True
        # Check for individual tool call marker
        return self.tool_call_start_token in text

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

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # The streaming parser needs the special-token markers (e.g.
            # <|tool_calls_section_begin|>) to appear as literal text in
            # current_text so it can detect section boundaries.  If
            # skip_special_tokens is True the tokenizer strips them out and
            # the parser never sees them, breaking tool-call extraction.
            request.skip_special_tokens = False
        return request

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

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent content before the tool-calls section, or None.

        Holds back any trailing suffix that partially matches
        ``<|tool_calls_section_begin|>`` to avoid leaking marker bytes.
        Uses ``_sent_content_idx`` so that content arriving in the same chunk
        as the section-begin marker is still emitted (the old buffer-based
        approach would drop it).
        """
        if self.tool_calls_start_token not in current_text:
            # No section marker yet — but the end of current_text might
            # partially match the start of the marker (e.g. "<|tool_call").
            # Hold back those bytes so they don't leak into content.
            overlap = partial_tag_overlap(current_text, self.tool_calls_start_token)
            sendable_idx = len(current_text) - overlap
        else:
            # Section marker found — everything before it is content.
            sendable_idx = current_text.index(self.tool_calls_start_token)

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx : sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

    def _extract_tool_calls(self, current_text: str) -> list[str]:
        """Extract raw tool-calls from ``<|tool_call_begin|>…<|tool_call_end|>`` blocks.

        Returns a list of the text between each begin/end pair (including the
        header and argument marker but not the outer begin/end tokens themselves).
        Handles truncated streams where the end marker hasn't arrived yet by
        trimming any partial overlap with the end marker.
        """
        if self.tool_calls_start_token not in current_text:
            return []
        results: list[str] = []
        pos = current_text.index(self.tool_calls_start_token)
        while True:
            start = current_text.find(self.tool_call_start_token, pos)
            if start == -1:
                break
            tc_start = start + len(self.tool_call_start_token)
            end = current_text.find(self.tool_call_end_token, tc_start)
            if end != -1:
                # Complete tool call — record it and advance past the end marker.
                tool_call = current_text[tc_start:end]
                pos = end + len(self.tool_call_end_token)
            else:
                # Truncated tool call (stream ended mid-call, e.g. max_tokens).
                # Trim any partial end-marker overlap so we don't leak
                # half-formed marker bytes into the arguments.
                tool_call = current_text[tc_start:]
                overlap = partial_tag_overlap(tool_call, self.tool_call_end_token)
                if overlap:
                    tool_call = tool_call[:-overlap]
                results.append(tool_call)
                break
            results.append(tool_call)
        return results

    @staticmethod
    def _extract_tool_id_and_name(
        header: str | None,
    ) -> tuple[str | None, str | None]:
        """Parse (tool_id, tool_name) from a header like ``"functions.get_weather:0"``.

        Returns ``(None, None)`` if the header doesn't match the expected
        ``name:index`` pattern (e.g. a malformed id like ``"invalid.0"`` with
        no colon+digit suffix).  This is used during streaming instead of the
        regex-based non-streaming path so that malformed tool calls are simply
        skipped rather than raising an exception.
        """
        if header is None:
            return None, None
        match = re.match(r"(.+:\d+)", header)
        if not match:
            return None, None
        tool_id = match.group(1).strip()
        # Strip any "functions." prefix to get the bare function name.
        tool_name = tool_id.split(":")[0].split(".")[-1]
        return tool_id, tool_name

    def _split_tool_call(self, tool_call: str) -> tuple[str | None, str | None]:
        """Split a tool-call body into ``(header, arguments)`` at the argument marker.

        Example::

            'get_weather:0 <|tool_call_argument_begin|>{"c'
            -> ("get_weather:0", '{"c')
        """
        arg_pos = tool_call.find(self.tool_call_arg_token)
        if arg_pos == -1:
            return None, None
        header = tool_call[:arg_pos].strip()
        tool_args = tool_call[arg_pos + len(self.tool_call_arg_token) :]
        return header, tool_args

    def _compute_args_diff(self, index: int, tool_args: str | None) -> str | None:
        """Return new argument text not yet sent for tool ``index``, or None.

        Each call compares the current full arguments string against what was
        previously streamed (stored in ``streamed_args_for_tool``) and emits
        only the suffix that hasn't been sent yet.
        """
        if tool_args is None:
            return None
        prev = self.streamed_args_for_tool[index]
        if len(tool_args) <= len(prev):
            # Nothing new to send (arguments may have been trimmed due to
            # partial end-marker overlap on a previous call).
            return None
        diff = tool_args[len(prev) :]
        self.streamed_args_for_tool[index] = tool_args
        self.prev_tool_call_arr[index]["arguments"] = tool_args
        return diff

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
        """Streaming tool-call extraction.

        Unlike the old buffer-based approach (which accumulated delta_text into
        token_buffer and processed it with regex), this method operates on the
        full current_text each call:

        1. ``_extract_content`` figures out what plain-text content before the
           section marker hasn't been emitted yet.  It uses _sent_content_idx
           to track progress, which fixes two bugs:
           - Content arriving in the same chunk as the section-begin marker was
             silently dropped (e.g. "Hi! <|tool_calls_section_begin|>…").
           - Content after the section-end marker was incorrectly streamed as
             plain text instead of being suppressed.

        2. ``_extract_tool_calls`` finds all complete or in-progress tool-call
           bodies between begin/end markers.

        3. For each tool call we check whether we've already sent its name/id
           (first time only) and compute the incremental argument diff.
        """
        try:
            # Step 1: emit any new pre-section content.
            content = self._extract_content(current_text)
            # Step 2: find all tool-call bodies in current_text.
            tool_calls = self._extract_tool_calls(current_text)
            tool_call_deltas: list[DeltaToolCall] = []

            for i, tool_call in enumerate(tool_calls):
                # First time seeing a tool call at index i — initialise its
                # streaming state slot.
                if i >= len(self.prev_tool_call_arr):
                    self.prev_tool_call_arr.append({})
                    self.streamed_args_for_tool.append("")

                header, tool_args = self._split_tool_call(tool_call)

                # Emit the tool name/id exactly once per tool call.
                if "name" not in self.prev_tool_call_arr[i]:
                    tool_id, tool_name = self._extract_tool_id_and_name(header)
                    # If the header is malformed (no "name:digit" pattern),
                    # skip this tool entirely rather than emitting bad data.
                    if not tool_name:
                        break
                    self.prev_tool_call_arr[i]["name"] = tool_name
                    self.prev_tool_call_arr[i]["id"] = tool_id
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            type="function",
                            id=tool_id,
                            function=DeltaFunctionCall(name=tool_name).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

                # Stream back new tool arguments by diffing against what was
                # already sent for this tool index.
                args_diff = self._compute_args_diff(i, tool_args)
                if args_diff:
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            function=DeltaFunctionCall(arguments=args_diff).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

            # Only emit a DeltaMessage if there's something to send.
            # Returning None means "nothing new this chunk" — the caller
            # should not emit a streaming chunk at all.
            if content or tool_call_deltas:
                return DeltaMessage(
                    content=content,
                    tool_calls=tool_call_deltas,
                )
            return None
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None
