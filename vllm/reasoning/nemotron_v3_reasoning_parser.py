# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike


class NemotronV3ReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for Nemotron V3 models.
    """

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking_enabled = chat_kwargs.get("enable_thinking", True)
        force_nonempty_content = chat_kwargs.get("force_nonempty_content", False)

        self._identity_parser: IdentityReasoningParser | None
        if thinking_enabled is False or force_nonempty_content is True:
            self._identity_parser = IdentityReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._identity_parser = None

        self._tool_call_tag = "<tool_call>"
        self._tool_call_token_id = self.vocab.get(self._tool_call_tag)
        self._tool_call_end_tag = "</tool_call>"
        self._tool_call_end_token_id = self.vocab.get(self._tool_call_end_tag)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._identity_parser is not None:
            return self._identity_parser.is_reasoning_end(input_ids)

        if super().is_reasoning_end(input_ids):
            return True

        if self._tool_call_token_id is None:
            return False

        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == self.start_token_id:
                return False
            if input_ids[i] == self._tool_call_token_id:
                # Paired tool-call tags can appear in prompt examples. Only
                # an opening tag without a later close tag is an implicit
                # transition from reasoning to content/tool parsing.
                if self._tool_call_end_token_id is not None and any(
                    input_ids[j] == self._tool_call_end_token_id
                    for j in range(i + 1, len(input_ids))
                ):
                    continue
                return True

        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self._identity_parser is not None:
            return self._identity_parser.is_reasoning_end_streaming(
                input_ids, delta_ids
            )

        if super().is_reasoning_end_streaming(input_ids, delta_ids):
            return True

        return (
            self._tool_call_token_id is not None
            and self._tool_call_token_id in delta_ids
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self._identity_parser is not None:
            return self._identity_parser.extract_content_ids(input_ids)

        result = super().extract_content_ids(input_ids)
        if result:
            return result

        if (
            self._tool_call_token_id is not None
            and self._tool_call_token_id in input_ids
        ):
            tool_call_index = (
                len(input_ids) - 1 - input_ids[::-1].index(self._tool_call_token_id)
            )
            return input_ids[tool_call_index:]

        return []

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        if self._identity_parser is not None:
            return self._identity_parser.extract_reasoning(model_output, request)

        reasoning, final_content = super().extract_reasoning(model_output, request)
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None)

        if (
            chat_template_kwargs
            and (
                chat_template_kwargs.get("enable_thinking") is False
                or chat_template_kwargs.get("force_nonempty_content") is True
            )
            and final_content is None
        ):
            reasoning, final_content = final_content, reasoning

        return reasoning, final_content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self._identity_parser is not None:
            return self._identity_parser.extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
