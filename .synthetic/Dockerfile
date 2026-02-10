# Dockerfile to patch vLLM with custom reasoning parser changes
# Based on vllm/vllm-openai:v0.15.0
# Build with: docker build -f .synththetic/Dockerfile -t syntheticdreamlabs/synthetic-vllm:<VERSION> .

FROM vllm/vllm-openai:v0.15.0

# Copy the patched and new reasoning parser files into the container
# The vllm package is installed at /usr/local/lib/python3.12/dist-packages/vllm
COPY vllm/reasoning/__init__.py /usr/local/lib/python3.12/dist-packages/vllm/reasoning/__init__.py
COPY vllm/reasoning/basic_parsers.py /usr/local/lib/python3.12/dist-packages/vllm/reasoning/basic_parsers.py
COPY vllm/reasoning/kimi_k2_reasoning_parser.py /usr/local/lib/python3.12/dist-packages/vllm/reasoning/kimi_k2_reasoning_parser.py
COPY vllm/reasoning/deepseek_r1_reasoning_parser.py /usr/local/lib/python3.12/dist-packages/vllm/reasoning/deepseek_r1_reasoning_parser.py

# Copy patched files from https://github.com/synthetic-lab/vllm/pull/1 (Add --max-request-secs to vllm serve)
COPY vllm/entrypoints/openai/chat_completion/serving.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py
COPY vllm/entrypoints/openai/completion/serving.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/serving.py
COPY vllm/entrypoints/openai/engine/serving.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/engine/serving.py
COPY vllm/entrypoints/openai/responses/serving.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/serving.py
COPY vllm/entrypoints/openai/api_server.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py
COPY vllm/entrypoints/openai/cli_args.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/cli_args.py

# The container will use the patched files when vLLM starts
