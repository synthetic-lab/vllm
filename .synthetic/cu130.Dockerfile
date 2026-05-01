# Dockerfile to patch vLLM with custom changes (CUDA 13.0 version)
# Build with: docker build -f .synthetic/cu130.Dockerfile -t syntheticdreamlabs/synthetic-vllm:<VERSION>-cu130 .

FROM vllm/vllm-openai:v0.20.0-cu130

# Preserve compiled extensions (eg. .so files) that may not be checked out.
RUN cp -r /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn /tmp/vllm_flash_attn_backup

# Copy the entire vllm codebase to replace the installed version
# The vllm package is installed at /usr/local/lib/python3.12/dist-packages/vllm
COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/

RUN cp -r /tmp/vllm_flash_attn_backup/* /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/ && \
    rm -rf /tmp/vllm_flash_attn_backup

# Remove __pycache__ directories to ensure fresh bytecode compilation
RUN find /usr/local/lib/python3.12/dist-packages/vllm -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# The container will use the patched files when vLLM starts
