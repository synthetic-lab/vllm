# Dockerfile to patch vLLM with custom reasoning parser changes
# Based on vllm/vllm-openai:v0.15.0
# Build with: docker build -f .scrapy/Dockerfile -t synthetic-vllm:<VERSION> .

FROM vllm/vllm-openai:v0.15.0

# Copy the patched and new reasoning parser files into the container
# The vllm package is installed at /usr/local/lib/python3.12/dist-packages/vllm
COPY vllm/reasoning/__init__.py /usr/local/lib/python3.12/dist-packages/vllm/reasoning/__init__.py
COPY vllm/reasoning/basic_parsers.py /usr/local/lib/python3.12/dist-packages/vllm/reasoning/basic_parsers.py
COPY vllm/reasoning/kimi_k2_reasoning_parser.py /usr/local/lib/python3.12/dist-packages/vllm/reasoning/kimi_k2_reasoning_parser.py

# The container will use the patched files when vLLM starts
