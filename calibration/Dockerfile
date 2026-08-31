ARG BASE_IMAGE=nvcr.io/nvidia/vllm@sha256:15f380ad9c32f0ac57ac16e4b778c6f733c88b9ffe3a936035d0a59ad17b1aab
FROM ${BASE_IMAGE}

COPY calibration/requirements-gh200.txt /tmp/requirements-gh200.txt

RUN python3 -m pip install \
        --no-cache-dir \
        --no-deps \
        --requirement /tmp/requirements-gh200.txt \
    && python3 -m pip check

LABEL org.opencontainers.image.title="Token-Energy Law FP8 KV calibration"
LABEL org.opencontainers.image.description="Offline calibration only; never used for paper energy measurement"
LABEL org.opencontainers.image.source="https://github.com/chenxuniu/active-bytes-law"
