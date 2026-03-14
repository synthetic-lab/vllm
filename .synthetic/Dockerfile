# Dockerfile to patch vLLM with custom changes
# Based on vllm/vllm-openai:v0.15.0
# Build with: docker build -f .synthetic/Dockerfile -t syntheticdreamlabs/synthetic-vllm:<VERSION> .

FROM vllm/vllm-openai:nightly-bc2c0c86efb28e77677a3cfb8687e976914a313a

# Copy the entire vllm codebase to replace the installed version
# The vllm package is installed at /usr/local/lib/python3.12/dist-packages/vllm
COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/

# Remove __pycache__ directories to ensure fresh bytecode compilation
RUN find /usr/local/lib/python3.12/dist-packages/vllm -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# The container will use the patched files when vLLM starts
