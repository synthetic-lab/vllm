# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import (
        ResponsesRequest,
    )
else:
    ChatCompletionRequest = Any
    ResponsesRequest = Any


class BaseThinkingReasoningParser(ReasoningParser):
    """
    Base class for reasoning parsers that use thinking tokens.

    This class provides common functionality for parsers that use start and end
    tokens to delimit reasoning content (e.g., think tags).

    Subclasses must implement the start and end tokens via abstract
    properties. Optionally, subclasses can also define tool_start_token
    which signals the end of reasoning but is NOT stripped from content
    (useful for tool calls that should be parsed by downstream parsers).
    """

    @property
    @abstractmethod
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        raise NotImplementedError

    @property
    @abstractmethod
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        raise NotImplementedError

    @property
    def tool_start_token(self) -> str | None:
        """
        Optional token that signals tool calls and ends reasoning.

        Unlike end_token, this token is NOT stripped from the content
        and will be included in subsequent parsing. Returns None by
        default; subclasses can override to provide a token.
        """
        return None

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        if not self.start_token or not self.end_token:
            raise ValueError("start_token and end_token must be defined in subclasses")

        start_token_id = self.vocab.get(self.start_token)
        end_token_id = self.vocab.get(self.end_token)
        if start_token_id is None or end_token_id is None:
            raise RuntimeError(
                f"{self.__class__.__name__} reasoning parser could not locate "
                "think start/end tokens in the tokenizer!"
            )
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id

        # Optional tool start token - not required
        self.tool_start_token_id: int | None = None
        if self.tool_start_token:
            tool_token_id = self.vocab.get(self.tool_start_token)
            if tool_token_id is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} reasoning parser could not locate "
                    "tool_start_token in the tokenizer!"
                )
            self.tool_start_token_id = tool_token_id

    def _split_reasoning_from_content(
        self,
        text: str,
        start_offset: int = 0,
    ) -> tuple[str, str | None]:
        """
        Split text into reasoning and content based on which end token comes first.

        Args:
            text: The text to split
            start_offset: Offset to start searching from (for stripping start_token)

        Returns:
            Tuple of (reasoning, content). Content is None if no end token found.
            If tool_start_token comes first, it's preserved in content.
            If end_token comes first, it's stripped from content.
        """
        search_text = text[start_offset:]

        end_index = search_text.find(self.end_token)
        tool_index = search_text.find(self.tool_start_token) if self.tool_start_token else -1

        if end_index != -1 and (tool_index == -1 or end_index < tool_index):
            # end_token comes first
            reasoning = search_text[:end_index]
            content = search_text[end_index + len(self.end_token):]
            return reasoning, content if content else None
        elif tool_index != -1:
            # tool_start_token comes first - preserve it in content
            reasoning = search_text[:tool_index]
            content = search_text[tool_index:]
            return reasoning, content if content else None
        else:
            # No end token found
            return search_text, None

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        tool_start_token_id = self.tool_start_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == end_token_id:
                return True
            if tool_start_token_id is not None and input_ids[i] == tool_start_token_id:
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        end_token_id = self.end_token_id
        tool_start_token_id = self.tool_start_token_id

        if end_token_id in delta_ids:
            return True
        if tool_start_token_id is not None and tool_start_token_id in delta_ids:
            return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content after the end tokens.

        Note: tool_start_token is NOT stripped from content since it
        needs to be parsed by downstream parsers.
        """
        end_token_id = self.end_token_id
        tool_start_token_id = self.tool_start_token_id

        # Check for end_token in the middle of input_ids
        if end_token_id in input_ids[:-1]:
            idx = input_ids.index(end_token_id)
            return input_ids[idx + 1 :]

        # Check for tool_start_token - returns content INCLUDING the token
        if tool_start_token_id is not None and tool_start_token_id in input_ids[:-1]:
            idx = input_ids.index(tool_start_token_id)
            return input_ids[idx :]

        return []

    def _end_token_present(
        self, delta_token_ids: Sequence[int]
    ) -> tuple[bool, bool]:
        """Check if end_token or tool_start_token is present in delta."""
        end_present = self.end_token_id in delta_token_ids
        tool_present = (
            self.tool_start_token_id is not None
            and self.tool_start_token_id in delta_token_ids
        )
        return end_present, tool_present

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a delta message.
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.
        """
        end_token_id = self.end_token_id
        tool_start_token_id = self.tool_start_token_id
        end_tokens_to_skip: list[int] = [self.start_token_id, end_token_id]
        if tool_start_token_id is not None:
            end_tokens_to_skip.append(tool_start_token_id)

        # Skip single special tokens
        if len(delta_token_ids) == 1 and delta_token_ids[0] in end_tokens_to_skip:
            return None

        # Check if start token is present in previous or delta.
        # Keep compatibility with models that don't generate start tokens.
        if self.start_token_id in previous_token_ids:
            end_present, tool_present = self._end_token_present(delta_token_ids)
            if end_present or tool_present:
                # An end token is in delta, use helper to split
                reasoning, content = self._split_reasoning_from_content(delta_text)
                return DeltaMessage(
                    reasoning=reasoning, content=content
                )
            elif end_token_id in previous_token_ids:
                # start token in previous, end token in previous,
                # reasoning content continues
                return DeltaMessage(content=delta_text)
            elif tool_start_token_id is not None and tool_start_token_id in previous_token_ids:
                # start token in previous, tool_start in previous,
                # reasoning ended, content continues (including tool_start)
                return DeltaMessage(content=delta_text)
            else:
                # start token in previous, no end token in previous or delta,
                # reasoning content continues
                return DeltaMessage(reasoning=delta_text)
        elif self.start_token_id in delta_token_ids:
            end_present, tool_present = self._end_token_present(delta_token_ids)
            if end_present or tool_present:
                # Both start and an end token in delta, find start then split
                start_index = delta_text.find(self.start_token)
                after_start = start_index + len(self.start_token)
                reasoning, content = self._split_reasoning_from_content(
                    delta_text, start_offset=after_start
                )
                return DeltaMessage(
                    reasoning=reasoning, content=content
                )
            else:
                # start token in delta, no end token in delta,
                # reasoning content continues
                return DeltaMessage(reasoning=delta_text)
        else:
            # not find thinking start token
            return DeltaMessage(content=delta_text)

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        This is the base implementation that works for most models.
        Subclasses can override this method for specific behavior.

        Note: tool_start_token ends reasoning but is preserved in content
        for downstream parsers to handle.
        """
        # Check if the start token is present in the model output, remove it
        # if it is present.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        # Use helper to split reasoning from content
        reasoning, content = self._split_reasoning_from_content(model_output)
        return reasoning, content
