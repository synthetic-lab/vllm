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
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
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

        self._sent_content_idx: int = 0
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

        self.tool_calls_start_token: str = "<|tool_calls_section_begin|>"
        self.tool_calls_end_token: str = "<|tool_calls_section_end|>"

        # Some Kimi K2 checkpoints emit the singular form instead of plural.
        self.tool_calls_start_token_variants: list[str] = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_section_begin|>",
        ]
        self.tool_calls_end_token_variants: list[str] = [
            "<|tool_calls_section_end|>",
            "<|tool_call_section_end|>",
        ]

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"
        self.tool_call_arg_token: str = "<|tool_call_argument_begin|>"

        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<]+:\d+)\s*"
            r"<\|tool_call_argument_begin\|>\s*"
            r"(?P<function_arguments>(?:(?!<\|tool_call_begin\|>).)*?)\s*"
            r"<\|tool_call_end\|>",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Ensure special-token markers appear as literal text in
            # current_text so we can do pure text-based parsing.
            request.skip_special_tokens = False
        return request

    def reset_streaming_state(self) -> None:
        self._sent_content_idx = 0
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []

    def _find_section_start(self, text: str) -> int:
        """Return the earliest index of any section-start variant, or -1."""
        best = -1
        for variant in self.tool_calls_start_token_variants:
            pos = text.find(variant)
            if pos != -1 and (best == -1 or pos < best):
                best = pos
        return best

    def _has_section_start(self, text: str) -> bool:
        return any(v in text for v in self.tool_calls_start_token_variants)

    @staticmethod
    def _parse_tool_id(tool_id: str) -> str:
        """Extract function name from a tool_id like 'functions.get_weather:0'."""
        return tool_id.split(":")[0].split(".")[-1]

    def _extract_content_before_tools(self, text: str) -> str:
        pos = self._find_section_start(text)
        return text[:pos] if pos != -1 else text

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if not self._has_section_start(model_output):
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

        Holds back any trailing suffix that partially matches a section-start
        marker (any variant) to avoid leaking marker bytes.
        """
        section_start = self._find_section_start(current_text)
        if section_start == -1:
            max_overlap = 0
            for variant in self.tool_calls_start_token_variants:
                overlap = partial_tag_overlap(current_text, variant)
                if overlap > max_overlap:
                    max_overlap = overlap
            sendable_idx = len(current_text) - max_overlap
        else:
            sendable_idx = section_start

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx : sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

    def _extract_tool_calls(self, current_text: str) -> list[str]:
        """Extract raw bodies from <|tool_call_begin|>...<|tool_call_end|> blocks."""
        section_start = self._find_section_start(current_text)
        if section_start == -1:
            return []

        results: list[str] = []
        pos = section_start
        while True:
            start = current_text.find(self.tool_call_start_token, pos)
            if start == -1:
                break
            tc_start = start + len(self.tool_call_start_token)
            end = current_text.find(self.tool_call_end_token, tc_start)

            if end != -1:
                tool_call = current_text[tc_start:end]
                pos = end + len(self.tool_call_end_token)
            else:
                tool_call = current_text[tc_start:]
                overlap = partial_tag_overlap(tool_call, self.tool_call_end_token)
                if overlap:
                    tool_call = tool_call[:-overlap]

            results.append(tool_call)

            if end == -1:
                break
        return results

    @staticmethod
    def _extract_tool_id_and_name(
        header: str | None,
    ) -> tuple[str | None, str | None]:
        if header is None:
            return None, None
        match = re.match(r"(.+:\d+)", header)
        if not match:
            return None, None

        tool_id = match.group(1).strip()
        tool_name = tool_id.split(":")[0].split(".")[-1]
        return tool_id, tool_name

    def _split_tool_call(self, tool_call: str) -> tuple[str | None, str | None]:
        arg_pos = tool_call.find(self.tool_call_arg_token)
        if arg_pos == -1:
            return None, None
        header = tool_call[:arg_pos].strip()
        tool_args = tool_call[arg_pos + len(self.tool_call_arg_token) :]
        return header, tool_args

    def _compute_args_diff(self, index: int, tool_args: str | None) -> str | None:
        if tool_args is None:
            return None
        prev = self.streamed_args_for_tool[index]
        if len(tool_args) <= len(prev):
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
        try:
            content = self._extract_content(current_text)
            tool_calls = self._extract_tool_calls(current_text)
            tool_call_deltas: list[DeltaToolCall] = []

            for i, tool_call in enumerate(tool_calls):
                if i >= len(self.prev_tool_call_arr):
                    self.prev_tool_call_arr.append({})
                    self.streamed_args_for_tool.append("")

                header, tool_args = self._split_tool_call(tool_call)

                if "name" not in self.prev_tool_call_arr[i]:
                    tool_id, tool_name = self._extract_tool_id_and_name(header)
                    if not tool_name:
                        # Tool i not ready yet; can't skip ahead to i+1.
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

            if content or tool_call_deltas:
                return DeltaMessage(
                    content=content,
                    tool_calls=tool_call_deltas,
                )
            return None

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None
