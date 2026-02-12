#!/usr/bin/env python3
"""
Standalone tests for KimiK2ToolParser that don't require CUDA.
Mocks the vllm dependencies to test pure string parsing logic.
"""

import json
import sys
from unittest.mock import MagicMock

# Mock vllm modules before importing the parser
sys.modules['vllm'] = MagicMock()
sys.modules['vllm.logger'] = MagicMock()
sys.modules['vllm.logger'].init_logger = lambda x: MagicMock()
sys.modules['vllm.tokenizers'] = MagicMock()
sys.modules['vllm.entrypoints'] = MagicMock()
sys.modules['vllm.entrypoints.openai'] = MagicMock()
sys.modules['vllm.entrypoints.openai.chat_completion'] = MagicMock()
sys.modules['vllm.entrypoints.openai.chat_completion.protocol'] = MagicMock()
sys.modules['vllm.entrypoints.openai.engine'] = MagicMock()
sys.modules['vllm.entrypoints.openai.engine.protocol'] = MagicMock()
sys.modules['vllm.tool_parsers'] = MagicMock()
sys.modules['vllm.tool_parsers.abstract_tool_parser'] = MagicMock()

# Create mock protocol classes
class DeltaFunctionCall:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments
    def model_dump(self, exclude_none=False):
        result = {}
        if self.name is not None:
            result['name'] = self.name
        if self.arguments is not None:
            result['arguments'] = self.arguments
        return result

class DeltaToolCall:
    def __init__(self, index=None, type=None, id=None, function=None):
        self.index = index
        self.type = type
        self.id = id
        self.function = function

class DeltaMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

class FunctionCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments
    def __eq__(self, other):
        return self.name == other.name and self.arguments == other.arguments

class ToolCall:
    def __init__(self, id, type, function):
        self.id = id
        self.type = type
        self.function = function

class ExtractedToolCallInformation:
    def __init__(self, tools_called, tool_calls, content):
        self.tools_called = tools_called
        self.tool_calls = tool_calls
        self.content = content

# Inject mocks into the protocol module
protocol_mock = sys.modules['vllm.entrypoints.openai.engine.protocol']
protocol_mock.DeltaFunctionCall = DeltaFunctionCall
protocol_mock.DeltaToolCall = DeltaToolCall
protocol_mock.DeltaMessage = DeltaMessage
protocol_mock.FunctionCall = FunctionCall
protocol_mock.ToolCall = ToolCall
protocol_mock.ExtractedToolCallInformation = ExtractedToolCallInformation

# Mock base ToolParser
class MockToolParser:
    def __init__(self, tokenizer):
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []
        self.model_tokenizer = tokenizer

    def get_vocab(self):
        return self.model_tokenizer.get_vocab()

sys.modules['vllm.tool_parsers.abstract_tool_parser'].ToolParser = MockToolParser

# Now import the actual parser
import regex as re
from pathlib import Path

# Read and exec the parser file to get the class
parser_path = Path(__file__).parent.parent.parent / "vllm" / "tool_parsers" / "kimi_k2_tool_parser.py"
parser_code = parser_path.read_text()

# Execute in a namespace with our mocks
namespace = {
    're': re,
    'Sequence': list,
    'ToolParser': MockToolParser,
    'DeltaFunctionCall': DeltaFunctionCall,
    'DeltaToolCall': DeltaToolCall,
    'DeltaMessage': DeltaMessage,
    'FunctionCall': FunctionCall,
    'ToolCall': ToolCall,
    'ExtractedToolCallInformation': ExtractedToolCallInformation,
    'ChatCompletionRequest': MagicMock,
    'TokenizerLike': MagicMock,
    'init_logger': lambda x: MagicMock(),
    'logger': MagicMock(),
}

exec(compile(parser_code, parser_path, 'exec'), namespace)
KimiK2ToolParser = namespace['KimiK2ToolParser']


# Mock tokenizer with vocab
class MockTokenizer:
    def get_vocab(self):
        return {
            "<|tool_calls_section_begin|>": 100,
            "<|tool_calls_section_end|>": 101,
            "<|tool_call_section_begin|>": 102,
            "<|tool_call_section_end|>": 103,
            "<|tool_call_begin|>": 104,
            "<|tool_call_end|>": 105,
            "<|tool_call_argument_begin|>": 106,
        }


def run_streaming_sequence(parser, deltas):
    """Helper to simulate a streaming sequence."""
    previous_text = ""
    previous_token_ids = []
    results = []

    for delta_text, delta_token_ids in deltas:
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + delta_token_ids

        result = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=None,
        )
        results.append(result)

        previous_text = current_text
        previous_token_ids = current_token_ids

    return results


def test_extract_tool_calls_no_tools():
    """Test non-streaming with no tool calls."""
    parser = KimiK2ToolParser(MockTokenizer())
    model_output = "This is a test"
    result = parser.extract_tool_calls(model_output, request=None)

    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == model_output
    print("✓ test_extract_tool_calls_no_tools")


def test_extract_tool_calls_single():
    """Test non-streaming with single tool call."""
    parser = KimiK2ToolParser(MockTokenizer())
    model_output = """I'll help you. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Beijing"}<|tool_call_end|><|tool_calls_section_end|>"""

    result = parser.extract_tool_calls(model_output, request=None)

    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.tool_calls[0].function.arguments == '{"city": "Beijing"}'
    assert result.content == "I'll help you. "
    print("✓ test_extract_tool_calls_single")


def test_extract_tool_calls_multiple():
    """Test non-streaming with multiple tool calls."""
    parser = KimiK2ToolParser(MockTokenizer())
    model_output = """Check both. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Tokyo"}<|tool_call_end|><|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{"city": "NYC"}<|tool_call_end|><|tool_calls_section_end|>"""

    result = parser.extract_tool_calls(model_output, request=None)

    assert result.tools_called
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.tool_calls[1].function.name == "get_weather"
    print("✓ test_extract_tool_calls_multiple")


def test_streaming_no_tool_calls():
    """Test streaming with no tool calls."""
    parser = KimiK2ToolParser(MockTokenizer())

    result = parser.extract_tool_calls_streaming(
        previous_text="Hello",
        current_text="Hello world",
        delta_text=" world",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    assert result is not None
    assert result.content == " world"
    print("✓ test_streaming_no_tool_calls")


def test_streaming_tool_call_not_leaked():
    """CRITICAL: Test that tool call markers don't leak into content."""
    parser = KimiK2ToolParser(MockTokenizer())

    deltas = [
        ("I'll help. ", [1, 2]),
        ("<|tool_calls_section_begin|>", [100]),
        ("<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>", [104, 3, 106]),
        ('{"city": "Paris"}', [4, 5]),
        ("<|tool_call_end|}><|tool_calls_section_end|>", [105, 101]),
        (" Done.", [6]),
    ]

    results = run_streaming_sequence(parser, deltas)

    # Collect all content
    all_content = []
    for r in results:
        if r and r.content:
            all_content.append(r.content)

    full_content = "".join(all_content)

    # Check no markers leaked
    assert "<|tool_call" not in full_content, f"Marker leaked: {full_content}"
    assert "<|tool_calls_section" not in full_content, f"Marker leaked: {full_content}"
    assert "get_weather" not in full_content, f"Tool name leaked: {full_content}"
    assert "Paris" not in full_content, f"Tool arg leaked: {full_content}"

    print("✓ test_streaming_tool_call_not_leaked")


def test_streaming_reentry_after_section():
    """Test that content after tool section streams correctly."""
    parser = KimiK2ToolParser(MockTokenizer())

    deltas = [
        ("<|tool_calls_section_begin|>", [100]),
        ("<|tool_calls_section_end|>", [101]),
        (" More text", [7, 8]),
    ]

    results = run_streaming_sequence(parser, deltas)

    assert parser.in_tool_section is False
    assert results[2] is not None
    assert results[2].content == " More text"
    print("✓ test_streaming_reentry_after_section")


def test_streaming_same_chunk_begin_end():
    """Test begin and end markers in same chunk."""
    parser = KimiK2ToolParser(MockTokenizer())

    result = parser.extract_tool_calls_streaming(
        previous_text="Reasoning ",
        current_text="Reasoning <|tool_calls_section_begin|><|tool_calls_section_end|>",
        delta_text="<|tool_calls_section_begin|><|tool_calls_section_end|>",
        previous_token_ids=[1],
        current_token_ids=[1, 100, 101],
        delta_token_ids=[100, 101],
        request=None,
    )

    assert parser.in_tool_section is False
    print("✓ test_streaming_same_chunk_begin_end")


def test_state_reset():
    """Test reset_streaming_state clears all state."""
    parser = KimiK2ToolParser(MockTokenizer())

    # Put parser in complex state
    parser.token_buffer = "some buffer"
    parser.current_tool_id = 5
    parser.prev_tool_call_arr = [{"id": "test"}]
    parser.section_char_count = 1000

    parser.reset_streaming_state()

    assert parser.in_tool_section is False
    assert parser.token_buffer == ""
    assert parser.current_tool_id == -1
    assert parser.prev_tool_call_arr == []
    assert parser.section_char_count == 0
    print("✓ test_state_reset")


def test_suppress_noise_between_markers():
    """Test that text between section_begin and tool_call_begin is suppressed."""
    parser = KimiK2ToolParser(MockTokenizer())

    deltas = [
        ("I'll help. ", [1, 2]),
        ("<|tool_calls_section_begin|>", [100]),
        (" spurious noise ", [3, 4]),
        ("<|tool_call_begin|>", [104]),
    ]

    results = run_streaming_sequence(parser, deltas)

    # First delta should be content
    assert results[0].content == "I'll help. "

    # Noise should be suppressed (empty or None)
    assert results[2] is None or results[2].content == "" or results[2].content is None
    print("✓ test_suppress_noise_between_markers")


def test_the_failing_scenario():
    """
    Test the exact scenario from the bug report:
    Tool calls 5 and 6 were being output as raw content.
    """
    parser = KimiK2ToolParser(MockTokenizer())

    # Simulate what might cause tools 5 and 6 to leak
    deltas = [
        ("<|tool_calls_section_begin|>", [100]),
        ('<|tool_call_begin|>functions.file_read:5<|tool_call_argument_begin|>{"path": "/test.md"}', [104, 1, 106, 2]),
        ("<|tool_call_end|>", [105]),
        ('<|tool_call_begin|>functions.file_read:6<|tool_call_argument_begin|>{"path": "/test2.md"}', [104, 3, 106, 4]),
        ("<|tool_call_end|>", [105]),
        ("<|tool_calls_section_end|>", [101]),
    ]

    results = run_streaming_sequence(parser, deltas)

    # Collect all content
    all_content = []
    for r in results:
        if r and r.content:
            all_content.append(r.content)

    full_content = "".join(all_content)

    # The bug was that this content leaked:
    # " <|tool_call_begin|> functions.file_read:5 ..."
    assert "file_read:5" not in full_content, f"Tool 5 leaked: {full_content}"
    assert "file_read:6" not in full_content, f"Tool 6 leaked: {full_content}"
    assert "<|tool_call_begin|>" not in full_content, f"Marker leaked: {full_content}"

    print("✓ test_the_failing_scenario")


if __name__ == "__main__":
    print("Running standalone Kimi K2 tool parser tests...\n")

    test_extract_tool_calls_no_tools()
    test_extract_tool_calls_single()
    test_extract_tool_calls_multiple()
    test_streaming_no_tool_calls()
    test_streaming_tool_call_not_leaked()
    test_streaming_reentry_after_section()
    test_streaming_same_chunk_begin_end()
    test_state_reset()
    test_suppress_noise_between_markers()
    test_the_failing_scenario()

    print("\n✓ All tests passed!")
